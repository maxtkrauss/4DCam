#!/usr/bin/env python3
"""
System-stability runner for test-time input perturbation ablations.

What it does:
1) Creates noisy copies of validation thorlabs inputs for several noise ablations.
2) Runs test.py on each noisy split with an existing trained checkpoint.
3) Runs HSI_comparison_probabalistic.py on each run.
4) Computes input PSNR (clean vs noisy).
5) Produces:
   - summary CSV (PSNR + reconstruction metrics per ablation)
   - MAE/SSIM/uncertainty vs PSNR plot
   - per-noise MAE/SSIM/uncertainty vs PSNR plots
   - clean-input vs noisy-input variants figure
   - reconstruction visualization grid (optional; requires keeping generated images)

Storage layout:
- `results_root`: lightweight outputs (metrics CSVs, plots, summaries)
- `generated_root`: heavy test-time generated reconstructions (TIFFs/HTML from test.py)
"""

import argparse
import errno
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff


@dataclass(frozen=True)
class NoiseAblation:
    name: str
    noise_type: str
    level: float


DEFAULT_TEST_OPTS = [
    "--model", "pix2pix",
    "--input_nc", "1",
    "--output_nc", "212",
    "--netG", "unet_1024",
    "--netG_reps", "2",
    "--netD_mult", "0",
    "--norm_bitwise",
    "--use_nll",
    "--lambda_l1", "0",
    "--norm", "instance",
    "--no_dropout",
    "--eval",
]


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_tifs(path: str) -> List[str]:
    files = [f for f in os.listdir(path) if f.lower().endswith((".tif", ".tiff"))]
    files.sort(key=natural_sort_key)
    return files


def _format_level(level: float) -> str:
    return f"{level:.4f}".replace(".", "p")


def make_default_ablations(steps: int = 10) -> List[NoiseAblation]:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    # Diffuser-noise stress set:
    # - Gaussian std: low -> high
    # - Speckle std: low -> high
    # - Salt-pepper amount: low -> high
    # - Poisson peak: high -> low (lower peak => stronger noise)
    gaussian_levels = np.linspace(0.003, 0.040, steps)
    speckle_levels = np.linspace(0.003, 0.050, steps)
    saltpepper_levels = np.linspace(0.001, 0.020, steps)
    poisson_peaks = np.geomspace(1024.0, 32.0, steps)

    ablations: List[NoiseAblation] = [NoiseAblation("clean", "none", 0.0)]
    for i, v in enumerate(gaussian_levels, start=1):
        ablations.append(NoiseAblation(f"gaussian_s{i:02d}_{_format_level(float(v))}", "gaussian", float(v)))
    for i, v in enumerate(speckle_levels, start=1):
        ablations.append(NoiseAblation(f"speckle_s{i:02d}_{_format_level(float(v))}", "speckle", float(v)))
    for i, v in enumerate(saltpepper_levels, start=1):
        ablations.append(NoiseAblation(f"saltpepper_s{i:02d}_{_format_level(float(v))}", "saltpepper", float(v)))
    for i, peak in enumerate(poisson_peaks, start=1):
        ablations.append(NoiseAblation(f"poisson_s{i:02d}_peak{int(round(float(peak)))}", "poisson", float(peak)))
    return ablations


def make_poisson_peak_ablations(peaks: List[float]) -> List[NoiseAblation]:
    ablations: List[NoiseAblation] = [NoiseAblation("clean", "none", 0.0)]
    seen_names = set(["clean"])
    for i, peak in enumerate(peaks, start=1):
        peak = max(1.0, float(peak))
        peak_label = int(round(peak))
        name = f"poisson_s{i:02d}_peak{peak_label}"
        while name in seen_names:
            peak_label += 1
            name = f"poisson_s{i:02d}_peak{peak_label}"
        seen_names.add(name)
        ablations.append(NoiseAblation(name, "poisson", peak))
    return ablations


def compute_psnr(clean: np.ndarray, noisy: np.ndarray, max_val: float = 1.0, eps: float = 1e-12) -> float:
    mse = float(np.mean((clean - noisy) ** 2))
    if mse <= eps:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def estimate_poisson_psnr_for_peak(
    clean_dataroot: str,
    phase: str,
    bit_max: float,
    peak: float,
    seed: int,
    max_images: Optional[int],
    calibration_images: int,
) -> float:
    src_thorlabs = os.path.join(clean_dataroot, phase, "thorlabs")
    if not os.path.isdir(src_thorlabs):
        raise FileNotFoundError(f"Missing thorlabs folder: {src_thorlabs}")

    tif_files = list_tifs(src_thorlabs)
    if max_images is not None:
        tif_files = tif_files[:max_images]
    tif_files = tif_files[:max(1, calibration_images)]
    if not tif_files:
        raise FileNotFoundError(f"No tif files found in {src_thorlabs}")

    rng = np.random.default_rng(seed)
    ablation = NoiseAblation(name=f"poisson_peak{int(round(float(peak)))}", noise_type="poisson", level=float(peak))
    psnr_vals: List[float] = []
    for fname in tif_files:
        arr = tiff.imread(os.path.join(src_thorlabs, fname)).astype(np.float32)
        clean = arr / bit_max
        noisy = apply_noise(clean, ablation, rng)
        psnr_vals.append(compute_psnr(clean, noisy, max_val=1.0))
    return float(np.mean(psnr_vals))


def build_poisson_psnr_sweep(
    clean_dataroot: str,
    phase: str,
    bit_max: float,
    seed: int,
    max_images: Optional[int],
    target_min_psnr: float,
    steps: int,
    calibration_images: int,
    start_peak: float,
    min_peak: float,
) -> List[NoiseAblation]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if target_min_psnr <= 0:
        raise ValueError("target_min_psnr must be > 0")

    peak = max(float(start_peak), float(min_peak))
    estimated_psnr = estimate_poisson_psnr_for_peak(
        clean_dataroot=clean_dataroot,
        phase=phase,
        bit_max=bit_max,
        peak=peak,
        seed=seed,
        max_images=max_images,
        calibration_images=calibration_images,
    )
    print(f"Poisson calibration: peak={peak:.3f} -> estimated input PSNR={estimated_psnr:.3f} dB")

    guard = 0
    while estimated_psnr > target_min_psnr and peak > min_peak and guard < 16:
        next_peak = max(float(min_peak), peak / 2.0)
        if np.isclose(next_peak, peak):
            break
        peak = next_peak
        estimated_psnr = estimate_poisson_psnr_for_peak(
            clean_dataroot=clean_dataroot,
            phase=phase,
            bit_max=bit_max,
            peak=peak,
            seed=seed + guard + 1,
            max_images=max_images,
            calibration_images=calibration_images,
        )
        print(f"Poisson calibration: peak={peak:.3f} -> estimated input PSNR={estimated_psnr:.3f} dB")
        guard += 1

    final_peak = max(float(min_peak), peak)
    peaks = np.geomspace(float(start_peak), final_peak, num=steps)
    peaks = [max(float(min_peak), float(x)) for x in peaks]
    peaks = sorted(set(int(round(x)) for x in peaks), reverse=True)
    if not peaks:
        peaks = [int(round(max(float(min_peak), float(start_peak))))]

    print(f"Selected Poisson peaks for sweep: {peaks}")
    return make_poisson_peak_ablations([float(x) for x in peaks])


def build_ablations(args) -> List[NoiseAblation]:
    if args.ablation_mode == "classic":
        return make_default_ablations(args.noise_steps)
    if args.ablation_mode == "poisson_psnr_sweep":
        return build_poisson_psnr_sweep(
            clean_dataroot=args.dataroot,
            phase=args.phase,
            bit_max=args.bit_max,
            seed=args.seed,
            max_images=args.max_images,
            target_min_psnr=args.target_min_psnr,
            steps=args.noise_steps,
            calibration_images=args.calibration_images,
            start_peak=args.poisson_start_peak,
            min_peak=args.poisson_min_peak,
        )
    raise ValueError(f"Unsupported ablation_mode: {args.ablation_mode}")


def apply_noise(x: np.ndarray, ablation: NoiseAblation, rng: np.random.Generator) -> np.ndarray:
    if ablation.noise_type == "none":
        y = x.copy()
    elif ablation.noise_type == "gaussian":
        y = x + rng.normal(loc=0.0, scale=ablation.level, size=x.shape).astype(np.float32)
    elif ablation.noise_type == "speckle":
        y = x + x * rng.normal(loc=0.0, scale=ablation.level, size=x.shape).astype(np.float32)
    elif ablation.noise_type == "saltpepper":
        y = x.copy()
        amount = float(ablation.level)
        if amount > 0:
            rnd = rng.random(size=x.shape)
            y[rnd < (amount / 2.0)] = 0.0
            y[rnd > (1.0 - amount / 2.0)] = 1.0
    elif ablation.noise_type == "poisson":
        peak = max(1.0, float(ablation.level))
        y = rng.poisson(np.clip(x, 0.0, 1.0) * peak).astype(np.float32) / peak
    else:
        raise ValueError(f"Unsupported noise type: {ablation.noise_type}")
    return np.clip(y, 0.0, 1.0)


def symlink_or_copytree(src: str, dst: str) -> None:
    if os.path.exists(dst):
        safe_remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def safe_remove_path(path: str, retries: int = 5, retry_sleep_s: float = 0.5) -> None:
    """Robust path removal for NFS-like filesystems (handles transient ENOTEMPTY)."""
    if not os.path.exists(path) and not os.path.islink(path):
        return

    # Handle symlink directly
    if os.path.islink(path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return

    # Retry shutil.rmtree first
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EBUSY):
                time.sleep(retry_sleep_s)
                continue
            if attempt == retries - 1:
                break
            time.sleep(retry_sleep_s)

    # Fallback to system rm -rf (common on Linux HPC)
    subprocess.run(["rm", "-rf", path], check=False)
    if os.path.exists(path):
        raise RuntimeError(f"Failed to remove path after retries: {path}")


def create_noisy_split(
    clean_dataroot: str,
    phase: str,
    out_root: str,
    ablation: NoiseAblation,
    bit_max: float,
    seed: int,
    max_images: Optional[int],
) -> Tuple[str, pd.DataFrame]:
    src_thorlabs = os.path.join(clean_dataroot, phase, "thorlabs")
    src_cubert = os.path.join(clean_dataroot, phase, "cubert")
    if not os.path.isdir(src_thorlabs):
        raise FileNotFoundError(f"Missing thorlabs folder: {src_thorlabs}")
    if not os.path.isdir(src_cubert):
        raise FileNotFoundError(f"Missing cubert folder: {src_cubert}")

    run_root = os.path.join(out_root, ablation.name)
    dst_phase = os.path.join(run_root, phase)
    dst_thorlabs = os.path.join(dst_phase, "thorlabs")
    dst_cubert = os.path.join(dst_phase, "cubert")

    if os.path.isdir(run_root):
        safe_remove_path(run_root)
    os.makedirs(dst_thorlabs, exist_ok=True)
    os.makedirs(dst_phase, exist_ok=True)

    rng = np.random.default_rng(seed)
    tif_files = list_tifs(src_thorlabs)
    cubert_files = list_tifs(src_cubert)
    if max_images is not None:
        tif_files = tif_files[:max_images]
        cubert_files = cubert_files[:max_images]

    # Keep paired cardinality stable to avoid repeated modulo sampling in aligned_dataset.
    if max_images is None:
        symlink_or_copytree(src_cubert, dst_cubert)
    else:
        os.makedirs(dst_cubert, exist_ok=True)
        for fname in cubert_files:
            shutil.copy2(os.path.join(src_cubert, fname), os.path.join(dst_cubert, fname))

    rows: List[Dict[str, float]] = []
    for fname in tif_files:
        src_path = os.path.join(src_thorlabs, fname)
        dst_path = os.path.join(dst_thorlabs, fname)
        arr = tiff.imread(src_path)
        arr_dtype = arr.dtype
        clean = arr.astype(np.float32) / bit_max
        noisy = apply_noise(clean, ablation, rng)
        noisy_arr = np.clip(noisy * bit_max, 0.0, bit_max).astype(arr_dtype)
        tiff.imwrite(dst_path, noisy_arr)
        rows.append({
            "file": fname,
            "input_psnr": compute_psnr(clean, noisy, max_val=1.0),
        })

    psnr_df = pd.DataFrame(rows)
    psnr_csv = os.path.join(run_root, "input_psnr.csv")
    psnr_df.to_csv(psnr_csv, index=False)
    print(f"Input PSNR table written to {psnr_csv}")
    return run_root, psnr_df


def find_eval_image_dir(results_dir: str, model_name: str, phase: str, epoch: str) -> str:
    candidate = os.path.join(results_dir, model_name, f"{phase}_{epoch}", "images")
    if os.path.isdir(candidate):
        return candidate
    fallback = os.path.join(results_dir, model_name, f"{phase}_latest", "images")
    if os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError(
        f"Could not locate results images for model={model_name} under {results_dir} "
        f"(checked {candidate} and {fallback})"
    )


def read_single_row_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df.iloc[0].to_dict()


def plot_metric_vs_psnr(summary_df: pd.DataFrame, out_png: str) -> None:
    plot_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["input_psnr_mean", "avg_mae", "avg_ssim_3d", "avg_sigma_mean"]
    )
    plot_df = plot_df.sort_values("input_psnr_mean", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for noise_type, g in plot_df.groupby("noise_type"):
        g = g.sort_values("input_psnr_mean", ascending=False)
        axes[0].plot(g["input_psnr_mean"], g["avg_mae"], marker="o", label=noise_type)
        axes[1].plot(g["input_psnr_mean"], g["avg_ssim_3d"], marker="o", label=noise_type)
        axes[2].plot(g["input_psnr_mean"], g["avg_sigma_mean"], marker="o", label=noise_type)

    axes[0].set_title("MAE vs Input PSNR")
    axes[0].set_xlabel("Input PSNR (dB)")
    axes[0].set_ylabel("Avg MAE")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("SSIM vs Input PSNR")
    axes[1].set_xlabel("Input PSNR (dB)")
    axes[1].set_ylabel("Avg SSIM (3D)")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Avg Uncertainty vs Input PSNR")
    axes[2].set_xlabel("Input PSNR (dB)")
    axes[2].set_ylabel("Avg Sigma Mean")
    axes[2].grid(alpha=0.25)

    for _, row in plot_df.iterrows():
        axes[0].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_mae"]), fontsize=7, alpha=0.85)
        axes[1].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_ssim_3d"]), fontsize=7, alpha=0.85)
        axes[2].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_sigma_mean"]), fontsize=7, alpha=0.85)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)
    print(f"Saved metric-vs-PSNR plots to {out_png}")


def plot_metric_vs_psnr_per_noise(summary_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["input_psnr_mean", "avg_mae", "avg_ssim_3d", "avg_sigma_mean"]
    ).copy()

    # Skip clean baseline for per-noise trend figures.
    plot_df = plot_df[plot_df["noise_type"] != "none"]
    if plot_df.empty:
        print("No per-noise rows found for chart generation.")
        return

    for noise_type, g in plot_df.groupby("noise_type"):
        g = g.sort_values("input_psnr_mean", ascending=False)
        out_png = os.path.join(out_dir, f"{noise_type}_mae_ssim_sigma_vs_psnr.png")

        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))
        axes[0].plot(g["input_psnr_mean"], g["avg_mae"], marker="o", color="#1f77b4")
        axes[1].plot(g["input_psnr_mean"], g["avg_ssim_3d"], marker="o", color="#d62728")
        axes[2].plot(g["input_psnr_mean"], g["avg_sigma_mean"], marker="o", color="#2ca02c")

        axes[0].set_title(f"{noise_type}: MAE vs Input PSNR")
        axes[0].set_xlabel("Input PSNR (dB)")
        axes[0].set_ylabel("Avg MAE")
        axes[0].grid(alpha=0.25)

        axes[1].set_title(f"{noise_type}: SSIM vs Input PSNR")
        axes[1].set_xlabel("Input PSNR (dB)")
        axes[1].set_ylabel("Avg SSIM (3D)")
        axes[1].grid(alpha=0.25)

        axes[2].set_title(f"{noise_type}: Avg Uncertainty vs Input PSNR")
        axes[2].set_xlabel("Input PSNR (dB)")
        axes[2].set_ylabel("Avg Sigma Mean")
        axes[2].grid(alpha=0.25)

        for _, row in g.iterrows():
            axes[0].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_mae"]), fontsize=7, alpha=0.85)
            axes[1].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_ssim_3d"]), fontsize=7, alpha=0.85)
            axes[2].annotate(row["ablation"], (row["input_psnr_mean"], row["avg_sigma_mean"]), fontsize=7, alpha=0.85)

        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
        plt.close(fig)
        print(f"Saved per-noise chart: {out_png}")


def load_gt_pred_sigma(eval_img_dir: str, image_index: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gt_files = [f for f in os.listdir(eval_img_dir) if f.startswith("cb_raw_") and f.lower().endswith((".tif", ".tiff"))]
    pred_files = [f for f in os.listdir(eval_img_dir) if f.startswith("tl_gen_") and f.lower().endswith((".tif", ".tiff"))]
    gt_files.sort(key=natural_sort_key)
    pred_files.sort(key=natural_sort_key)
    if not gt_files or not pred_files:
        raise FileNotFoundError(f"No cb_raw_/tl_gen_ tif files found in {eval_img_dir}")
    idx = min(image_index, min(len(gt_files), len(pred_files)) - 1)
    gt = tiff.imread(os.path.join(eval_img_dir, gt_files[idx])).astype(np.float32)
    pred_full = tiff.imread(os.path.join(eval_img_dir, pred_files[idx])).astype(np.float32)
    pred = pred_full[:gt.shape[0]]
    sigma = pred_full[gt.shape[0]:] if pred_full.shape[0] > gt.shape[0] else None
    return gt, pred, sigma


def make_reconstruction_grid(
    summary_df: pd.DataFrame,
    eval_dir_map: Dict[str, str],
    out_png: str,
    image_index: int,
    max_rows: int,
) -> None:
    vis_df = summary_df.sort_values("input_psnr_mean", ascending=False).copy()
    if "clean" in vis_df["ablation"].values:
        clean_row = vis_df[vis_df["ablation"] == "clean"]
        rest = vis_df[vis_df["ablation"] != "clean"]
        vis_df = pd.concat([clean_row, rest], ignore_index=True)
    vis_df = vis_df.head(max_rows)

    n_rows = len(vis_df)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, max(3.0, 2.6 * n_rows)))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (_, row) in enumerate(vis_df.iterrows()):
        ablation = row["ablation"]
        gt, pred, sigma = load_gt_pred_sigma(eval_dir_map[ablation], image_index=image_index)
        gt_map = gt.mean(axis=0)
        pred_map = pred.mean(axis=0)
        err_map = np.mean(np.abs(gt - pred), axis=0)
        sigma_map = sigma.mean(axis=0) if sigma is not None and sigma.size > 0 else np.zeros_like(err_map)

        im0 = axes[r, 0].imshow(gt_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r, 0].set_title("GT (band-avg)")
        axes[r, 0].axis("off")
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.01)

        im1 = axes[r, 1].imshow(pred_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r, 1].set_title("Prediction (band-avg)")
        axes[r, 1].axis("off")
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.01)

        err_vmax = float(np.percentile(err_map, 99))
        im2 = axes[r, 2].imshow(err_map, cmap="magma", vmin=0.0, vmax=max(1e-6, err_vmax))
        axes[r, 2].set_title("Abs Error (band-avg)")
        axes[r, 2].axis("off")
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.01)

        sig_vmax = float(np.percentile(sigma_map, 99))
        im3 = axes[r, 3].imshow(sigma_map, cmap="plasma", vmin=0.0, vmax=max(1e-6, sig_vmax))
        axes[r, 3].set_title("Sigma (band-avg)")
        axes[r, 3].axis("off")
        fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.01)

        label = (
            f"{ablation}\n"
            f"PSNR={row['input_psnr_mean']:.2f} dB | "
            f"MAE={row['avg_mae']:.4f} | "
            f"SSIM={row['avg_ssim_3d']:.4f}"
        )
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=70, va="center", fontsize=9)

    fig.suptitle("Reconstruction Degradation Under Test-Time Input Perturbations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)
    print(f"Saved reconstruction grid to {out_png}")


def count_eval_pairs(eval_img_dir: str) -> int:
    return len([f for f in os.listdir(eval_img_dir) if f.startswith("cb_raw_") and f.lower().endswith((".tif", ".tiff"))])


def sanitize_tag(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s.strip("_") or "run"


def select_input_channel(img: np.ndarray, polarization: int) -> np.ndarray:
    # Expected thorlabs image shape: [C, H, W] with channels matching pol angles.
    if img.ndim == 2:
        return img
    if img.ndim != 3:
        raise ValueError(f"Unexpected input image shape: {img.shape}")

    pol_to_idx = {0: 0, 45: 1, 90: 2, 135: 3}
    c_idx = pol_to_idx.get(polarization, 0)
    c_idx = min(c_idx, img.shape[0] - 1)
    return img[c_idx]


def make_input_noise_variants_figure(
    clean_dataroot: str,
    phase: str,
    noisy_root_map: Dict[str, str],
    ablations: List[NoiseAblation],
    out_png: str,
    image_index: int,
    polarization: int,
    bit_max: float,
) -> None:
    clean_dir = os.path.join(clean_dataroot, phase, "thorlabs")
    clean_files = list_tifs(clean_dir)
    if not clean_files:
        raise FileNotFoundError(f"No tif files found in clean input dir: {clean_dir}")
    idx = min(image_index, len(clean_files) - 1)
    fname = clean_files[idx]

    clean_img_raw = tiff.imread(os.path.join(clean_dir, fname)).astype(np.float32)
    clean_cube = clean_img_raw / bit_max
    clean_2d = np.clip(select_input_channel(clean_cube, polarization), 0.0, 1.0)

    panel_entries = [("clean", clean_2d, float("inf"))]
    for ab in ablations:
        if ab.name == "clean":
            continue
        noisy_dir = os.path.join(noisy_root_map[ab.name], phase, "thorlabs")
        noisy_path = os.path.join(noisy_dir, fname)
        if not os.path.isfile(noisy_path):
            # Fallback to index-aligned file if names differ.
            noisy_files = list_tifs(noisy_dir)
            noisy_fname = noisy_files[min(idx, len(noisy_files) - 1)]
            noisy_path = os.path.join(noisy_dir, noisy_fname)
        noisy_img_raw = tiff.imread(noisy_path).astype(np.float32)
        noisy_cube = noisy_img_raw / bit_max
        noisy_2d = np.clip(select_input_channel(noisy_cube, polarization), 0.0, 1.0)
        psnr = compute_psnr(clean_2d, noisy_2d, max_val=1.0)
        panel_entries.append((ab.name, noisy_2d, psnr))

    n_panels = len(panel_entries)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (name, img2d, psnr) in enumerate(panel_entries):
        ax = axes[i]
        im = ax.imshow(img2d, cmap="gray", vmin=0.0, vmax=1.0)
        if np.isinf(psnr):
            title = f"{name}\nPSNR=inf"
        else:
            title = f"{name}\nPSNR={psnr:.2f} dB"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Input Noise Variants (phase={phase}, image={idx}, polarization={polarization} deg)",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)
    print(f"Saved input-noise variants figure to {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run test-time stability/noise ablations and summarize degradation.")
    parser.add_argument("--dataroot", type=str, required=True, help="Clean dataset root containing <phase>/thorlabs and <phase>/cubert")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Checkpoint root used by test.py")
    parser.add_argument("--model_name", type=str, required=True, help="Experiment name (must match trained checkpoint folder name)")
    parser.add_argument("--phase", type=str, default="validation", help="Dataset split/phase to evaluate")
    parser.add_argument("--epoch", type=str, default="latest", help="Checkpoint epoch to load")
    parser.add_argument("--polarization", type=int, default=0, help="Polarization channel selector (0/45/90/135)")

    parser.add_argument("--python_exe", type=str, default="python", help="Python executable to launch scripts")
    parser.add_argument("--test_script", type=str, default="test.py", help="Path to test.py")
    parser.add_argument("--eval_script", type=str, default="HSI_comparison_probabalistic.py", help="Path to eval script")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory for script execution")

    parser.add_argument("--results_root", type=str, default="results_stability", help="Base root for outputs")
    parser.add_argument("--generated_root", type=str, default=None, help="Base root for heavy generated test outputs (recommended: scratch)")
    parser.add_argument("--workspace_root", type=str, default=None, help="Base root for temporary noisy dataroots (default: <generated_root or results_root>/_workspace)")
    parser.add_argument("--run_tag", type=str, default=None, help="Unique namespace for this run (auto-generated if omitted)")
    parser.add_argument("--no_namespace", action="store_true", help="Disable run namespacing (legacy behavior; not safe for concurrent runs)")
    parser.add_argument("--bit_max", type=float, default=4095.0, help="Max sensor value used for normalization before adding noise")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for noise generation")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of validation images")
    parser.add_argument("--cleanup_noisy_inputs", action="store_true", help="Delete noisy copied datasets after run")
    parser.add_argument("--keep_generated_images", action="store_true", help="Keep test-time generated reconstruction folders (default deletes them after eval)")

    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="classic",
        choices=["classic", "poisson_psnr_sweep"],
        help="Use the legacy multi-noise sweep or an adaptive Poisson-only PSNR sweep",
    )
    parser.add_argument("--noise_steps", type=int, default=10, help="Number of severity steps per noise type")
    parser.add_argument("--target_min_psnr", type=float, default=10.0, help="For poisson_psnr_sweep: keep increasing noise until estimated mean PSNR reaches this threshold")
    parser.add_argument("--calibration_images", type=int, default=8, help="For poisson_psnr_sweep: number of images used to estimate PSNR levels")
    parser.add_argument("--poisson_start_peak", type=float, default=1024.0, help="For poisson_psnr_sweep: starting Poisson peak (high peak = mild noise)")
    parser.add_argument("--poisson_min_peak", type=float, default=1.0, help="For poisson_psnr_sweep: lowest Poisson peak allowed during calibration")
    parser.add_argument("--extra_test_opts", nargs="*", default=[], help="Additional args appended to test.py command")
    parser.add_argument("--no_default_test_opts", action="store_true", help="Disable built-in TEST_OPTS and only use extra opts")

    parser.add_argument("--viz_image_index", type=int, default=0, help="Which reconstructed sample index to visualize")
    parser.add_argument("--viz_max_rows", type=int, default=8, help="How many ablations to include in the reconstruction grid")
    parser.add_argument("--skip_reconstruction_grid", action="store_true", help="Skip reconstruction grid output")
    parser.add_argument("--input_noise_fig_index", type=int, default=0, help="Which clean/noisy input sample index to visualize")
    parser.add_argument("--skip_input_noise_figure", action="store_true", help="Do not generate input clean-vs-noisy figure")
    return parser.parse_args()


def main():
    args = parse_args()

    # Namespace paths by default so concurrent runs do not collide (e.g., both writing ablation 'clean').
    if args.no_namespace:
        results_root = args.results_root
        generated_root = args.generated_root or args.results_root
        workspace_root = args.workspace_root or os.path.join(generated_root, "_workspace")
        run_tag = "legacy"
    else:
        auto_tag = f"{sanitize_tag(args.model_name)}__{sanitize_tag(os.path.basename(os.path.normpath(args.dataroot)))}__{sanitize_tag(args.phase)}__pol{args.polarization}"
        run_tag = sanitize_tag(args.run_tag) if args.run_tag else auto_tag
        results_root = os.path.join(args.results_root, run_tag)
        generated_base = args.generated_root or args.results_root
        generated_root = os.path.join(generated_base, run_tag)
        workspace_base = args.workspace_root or os.path.join(generated_base, "_workspace")
        workspace_root = os.path.join(workspace_base, run_tag)

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(generated_root, exist_ok=True)
    os.makedirs(workspace_root, exist_ok=True)
    print(f"Run namespace: {run_tag}")
    print(f"Results root: {results_root}")
    print(f"Generated outputs root: {generated_root}")
    print(f"Workspace root: {workspace_root}")

    ablations = build_ablations(args)
    print("Ablations:", [a.name for a in ablations])

    summary_rows = []
    eval_dir_map: Dict[str, str] = {}
    noisy_root_map: Dict[str, str] = {}

    for i, ablation in enumerate(ablations):
        ablation_seed = args.seed + (i * 101)
        print(f"\n=== Ablation: {ablation.name} ({ablation.noise_type}, level={ablation.level}) ===")

        noisy_root, psnr_df = create_noisy_split(
            clean_dataroot=args.dataroot,
            phase=args.phase,
            out_root=workspace_root,
            ablation=ablation,
            bit_max=args.bit_max,
            seed=ablation_seed,
            max_images=args.max_images,
        )
        noisy_root_map[ablation.name] = noisy_root

        # test.py writes heavy TIFF outputs here (prefer scratch via --generated_root)
        ablation_generated_dir = os.path.join(generated_root, ablation.name)
        if os.path.isdir(ablation_generated_dir):
            safe_remove_path(ablation_generated_dir)
        os.makedirs(ablation_generated_dir, exist_ok=True)
        metrics_dir = os.path.join(results_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_csv = os.path.join(metrics_dir, f"metrics_{ablation.name}.csv")

        test_cmd = [
            args.python_exe, args.test_script,
            "--dataroot", noisy_root,
            "--name", args.model_name,
            "--checkpoints_dir", args.checkpoints_dir,
            "--phase", args.phase,
            "--epoch", args.epoch,
            "--polarization", str(args.polarization),
            "--results_dir", ablation_generated_dir,
        ]
        if not args.no_default_test_opts:
            test_cmd += DEFAULT_TEST_OPTS
        if args.extra_test_opts:
            test_cmd += args.extra_test_opts
        run_cmd(test_cmd, cwd=args.cwd)

        eval_img_dir = find_eval_image_dir(
            results_dir=ablation_generated_dir,
            model_name=args.model_name,
            phase=args.phase,
            epoch=args.epoch,
        )
        eval_dir_map[ablation.name] = eval_img_dir
        num_images = count_eval_pairs(eval_img_dir)
        if args.max_images is not None:
            num_images = min(num_images, args.max_images)

        eval_cmd = [
            args.python_exe, args.eval_script,
            "--results_dir", eval_img_dir,
            "--num_images", str(max(1, num_images)),
            "--metrics_csv", metrics_csv,
        ]
        run_cmd(eval_cmd, cwd=args.cwd)

        metrics = read_single_row_csv(metrics_csv)
        summary_rows.append({
            "ablation": ablation.name,
            "noise_type": ablation.noise_type,
            "noise_level": ablation.level,
            "model_name": args.model_name,
            "dataroot": args.dataroot,
            "phase": args.phase,
            "polarization": args.polarization,
            "run_tag": run_tag,
            "num_input_images": int(len(psnr_df)),
            "input_psnr_mean": float(psnr_df["input_psnr"].mean()),
            "input_psnr_std": float(psnr_df["input_psnr"].std(ddof=0)),
            **metrics,
        })

        # Free disk immediately after metrics are written.
        if not args.keep_generated_images:
            if os.path.isdir(ablation_generated_dir):
                safe_remove_path(ablation_generated_dir)
                print(f"Deleted generated reconstructions: {ablation_generated_dir}")

    summary_df = pd.DataFrame(summary_rows).sort_values("input_psnr_mean", ascending=False)
    summary_csv = os.path.join(results_root, "stability_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary written to {summary_csv}")

    metric_plot = os.path.join(results_root, "mae_ssim_vs_psnr.png")
    plot_metric_vs_psnr(summary_df, metric_plot)
    per_noise_plot_dir = os.path.join(results_root, "per_noise_charts")
    plot_metric_vs_psnr_per_noise(summary_df, per_noise_plot_dir)

    if not args.skip_input_noise_figure:
        input_noise_plot = os.path.join(results_root, "input_noise_variants.png")
        make_input_noise_variants_figure(
            clean_dataroot=args.dataroot,
            phase=args.phase,
            noisy_root_map=noisy_root_map,
            ablations=ablations,
            out_png=input_noise_plot,
            image_index=args.input_noise_fig_index,
            polarization=args.polarization,
            bit_max=args.bit_max,
        )

    if args.skip_reconstruction_grid:
        print("Skipping reconstruction grid (--skip_reconstruction_grid).")
    elif not args.keep_generated_images:
        print("Skipping reconstruction grid because generated images were deleted. Use --keep_generated_images to enable it.")
    else:
        recon_plot = os.path.join(results_root, "reconstruction_grid.png")
        make_reconstruction_grid(
            summary_df=summary_df,
            eval_dir_map=eval_dir_map,
            out_png=recon_plot,
            image_index=args.viz_image_index,
            max_rows=args.viz_max_rows,
        )

    if args.cleanup_noisy_inputs:
        safe_remove_path(workspace_root)
        print(f"Removed temporary noisy dataroots: {workspace_root}")


if __name__ == "__main__":
    main()
