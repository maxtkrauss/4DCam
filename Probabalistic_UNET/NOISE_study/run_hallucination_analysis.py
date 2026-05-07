#!/usr/bin/env python3
"""
Cross-dataset hallucination/robustness analysis runner.

Evaluates two trained models on both their own dataset and the other dataset:
  - model A on dataset A (in-distribution)
  - model A on dataset B (cross-distribution)
  - model B on dataset B (in-distribution)
  - model B on dataset A (cross-distribution)

Outputs:
  - per-run aggregate metrics CSVs
  - per-run per-image CSVs
  - combined summary CSV
  - delta summary CSV (cross vs in for each model)
  - simple explanatory plots:
      * 2x2 heatmaps (MAE, SSIM, sigma_mean)
      * in-vs-cross bar charts (MAE and sigma_mean)
      * per-image sigma_mean boxplot
      * qualitative panel (GT / prediction / error / sigma)
  - markdown report with key numeric deltas
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


@dataclass(frozen=True)
class EvalCase:
    model_label: str
    model_name: str
    checkpoints_dir: str
    dataset_label: str
    dataroot: str

    @property
    def run_tag(self) -> str:
        return f"{self.model_label}_on_{self.dataset_label}"


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def safe_remove_path(path: str, retries: int = 5, retry_sleep_s: float = 0.5) -> None:
    """Best-effort robust recursive delete for generated run folders."""
    if not os.path.exists(path) and not os.path.islink(path):
        return

    if os.path.islink(path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EBUSY, errno.EACCES) and attempt < retries - 1:
                time.sleep(retry_sleep_s)
                continue
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to remove path after retries: {path}") from e
            time.sleep(retry_sleep_s)


def count_eval_pairs(eval_img_dir: str) -> int:
    return len([f for f in os.listdir(eval_img_dir) if f.startswith("cb_raw_") and f.lower().endswith((".tif", ".tiff"))])


def find_eval_image_dir(results_dir: str, model_name: str, phase: str, epoch: str) -> str:
    c1 = os.path.join(results_dir, model_name, f"{phase}_{epoch}", "images")
    if os.path.isdir(c1):
        return c1
    c2 = os.path.join(results_dir, model_name, f"{phase}_latest", "images")
    if os.path.isdir(c2):
        return c2
    raise FileNotFoundError(
        f"Could not locate result images for model={model_name} under {results_dir} "
        f"(checked {c1} and {c2})"
    )


def read_single_row_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    return df.iloc[0].to_dict()


def load_gt_pred_sigma(eval_img_dir: str, image_index: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gt_files = [f for f in os.listdir(eval_img_dir) if f.startswith("cb_raw_") and f.lower().endswith((".tif", ".tiff"))]
    pred_files = [f for f in os.listdir(eval_img_dir) if f.startswith("tl_gen_") and f.lower().endswith((".tif", ".tiff"))]
    gt_files.sort(key=natural_sort_key)
    pred_files.sort(key=natural_sort_key)
    if not gt_files or not pred_files:
        raise FileNotFoundError(f"No cb_raw_/tl_gen_ tif files in {eval_img_dir}")
    idx = min(image_index, min(len(gt_files), len(pred_files)) - 1)
    gt = tiff.imread(os.path.join(eval_img_dir, gt_files[idx])).astype(np.float32)
    pred_full = tiff.imread(os.path.join(eval_img_dir, pred_files[idx])).astype(np.float32)
    pred = pred_full[:gt.shape[0]]
    sigma = pred_full[gt.shape[0]:] if pred_full.shape[0] > gt.shape[0] else None
    return gt, pred, sigma


def run_case(
    case: EvalCase,
    args,
    case_results_root: str,
    metrics_csv: str,
    per_image_csv: str,
) -> Tuple[Optional[str], Dict[str, float]]:
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(per_image_csv), exist_ok=True)

    # Ensure each case starts clean so stale images are never mixed into metrics.
    if not args.reuse_existing and os.path.isdir(case_results_root):
        safe_remove_path(case_results_root)
    os.makedirs(case_results_root, exist_ok=True)

    need_metrics = (not args.reuse_existing) or (not os.path.isfile(metrics_csv))
    need_per_image = (not args.reuse_existing) or (not os.path.isfile(per_image_csv))

    if not args.reuse_existing:
        test_cmd = [
            args.python_exe, args.test_script,
            "--dataroot", case.dataroot,
            "--name", case.model_name,
            "--checkpoints_dir", case.checkpoints_dir,
            "--phase", args.phase,
            "--epoch", args.epoch,
            "--polarization", str(args.polarization),
            "--results_dir", case_results_root,
        ]
        if not args.no_default_test_opts:
            test_cmd += DEFAULT_TEST_OPTS
        if args.extra_test_opts:
            test_cmd += args.extra_test_opts
        run_cmd(test_cmd, cwd=args.cwd)

    eval_img_dir: Optional[str] = None
    if need_metrics or need_per_image:
        eval_img_dir = find_eval_image_dir(
            results_dir=case_results_root,
            model_name=case.model_name,
            phase=args.phase,
            epoch=args.epoch,
        )
        num_images = count_eval_pairs(eval_img_dir)
        if args.max_images is not None:
            num_images = min(num_images, args.max_images)
        num_images = max(1, num_images)

    if need_metrics:
        if eval_img_dir is None:
            raise RuntimeError(f"Cannot compute metrics for {case.run_tag}: missing eval image directory.")
        eval_cmd = [
            args.python_exe, args.eval_script,
            "--results_dir", eval_img_dir,
            "--num_images", str(num_images),
            "--metrics_csv", metrics_csv,
        ]
        run_cmd(eval_cmd, cwd=args.cwd)

    if need_per_image:
        if eval_img_dir is None:
            raise RuntimeError(f"Cannot compute per-image metrics for {case.run_tag}: missing eval image directory.")
        per_image_cmd = [
            args.python_exe, args.per_image_script,
            "--results_dir", eval_img_dir,
            "--num_images", str(num_images),
            "--per_image_csv", per_image_csv,
        ]
        run_cmd(per_image_cmd, cwd=args.cwd)
    elif args.reuse_existing:
        # Optional qualitative panel input when reusing prior CSVs.
        try:
            eval_img_dir = find_eval_image_dir(
                results_dir=case_results_root,
                model_name=case.model_name,
                phase=args.phase,
                epoch=args.epoch,
            )
        except FileNotFoundError:
            eval_img_dir = None

    metrics = read_single_row_csv(metrics_csv)
    return eval_img_dir, metrics


def save_heatmaps(summary_df: pd.DataFrame, labels: Tuple[str, str], out_png: str) -> None:
    model_labels = list(labels)
    dataset_labels = list(labels)
    metric_specs = [
        ("avg_mae", "MAE"),
        ("avg_ssim_3d", "SSIM (3D)"),
        ("avg_sigma_mean", "Sigma Mean"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, title) in zip(axes, metric_specs):
        grid = np.full((2, 2), np.nan, dtype=np.float32)
        for i, m in enumerate(model_labels):
            for j, d in enumerate(dataset_labels):
                row = summary_df[(summary_df["model_label"] == m) & (summary_df["dataset_label"] == d)]
                if not row.empty:
                    grid[i, j] = float(row.iloc[0][col])
        im = ax.imshow(grid, cmap="viridis")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(dataset_labels)
        ax.set_yticklabels(model_labels)
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Model")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                val = grid[i, j]
                txt = "nan" if np.isnan(val) else f"{val:.4f}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Hallucination Matrix: In-Distribution vs Cross-Dataset", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)


def save_in_vs_cross_bars(summary_df: pd.DataFrame, in_dataset_map: Dict[str, str], out_png: str) -> None:
    rows = []
    for m in sorted(summary_df["model_label"].unique()):
        md = summary_df[summary_df["model_label"] == m]
        in_label = in_dataset_map[m]
        in_row = md[md["dataset_label"] == in_label]
        out_row = md[md["dataset_label"] != in_label]
        if in_row.empty or out_row.empty:
            continue
        rows.append({
            "model_label": m,
            "in_mae": float(in_row.iloc[0]["avg_mae"]),
            "out_mae": float(out_row.iloc[0]["avg_mae"]),
            "in_sigma": float(in_row.iloc[0]["avg_sigma_mean"]),
            "out_sigma": float(out_row.iloc[0]["avg_sigma_mean"]),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(x - w / 2, df["in_mae"], width=w, label="In-Distribution")
    axes[0].bar(x + w / 2, df["out_mae"], width=w, label="Cross-Dataset")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model_label"])
    axes[0].set_ylabel("Avg MAE")
    axes[0].set_title("MAE Shift")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - w / 2, df["in_sigma"], width=w, label="In-Distribution")
    axes[1].bar(x + w / 2, df["out_sigma"], width=w, label="Cross-Dataset")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model_label"])
    axes[1].set_ylabel("Avg Sigma Mean")
    axes[1].set_title("Uncertainty Shift")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("In vs Cross Dataset Comparison per Model", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)


def save_per_image_sigma_boxplot(per_image_df: pd.DataFrame, out_png: str) -> None:
    if per_image_df.empty or "sigma_mean" not in per_image_df.columns:
        return
    order = sorted(per_image_df["run_tag"].unique())
    data = [per_image_df[per_image_df["run_tag"] == tag]["sigma_mean"].astype(float).to_numpy() for tag in order]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Per-image Sigma Mean")
    ax.set_title("Uncertainty Distribution by Run")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)


def build_qualitative_maps(eval_img_dir: str, image_index: int) -> Dict[str, np.ndarray]:
    gt, pred, sigma = load_gt_pred_sigma(eval_img_dir, image_index=image_index)
    err_map = np.mean(np.abs(gt - pred), axis=0)
    return {
        "gt_map": np.mean(gt, axis=0),
        "pred_map": np.mean(pred, axis=0),
        "err_map": err_map,
        "sigma_map": np.mean(sigma, axis=0) if sigma is not None and sigma.size > 0 else np.zeros_like(err_map),
    }


def save_qualitative_panel(summary_df: pd.DataFrame, qual_maps: Dict[str, Dict[str, np.ndarray]], out_png: str) -> None:
    vis_df = summary_df[summary_df["run_tag"].isin(qual_maps.keys())].copy()
    if vis_df.empty:
        print("Skipping qualitative panel: no cached qualitative maps available.")
        return

    vis_df["sort_key"] = vis_df["is_in_distribution"].astype(int)
    vis_df = vis_df.sort_values(["sort_key", "model_label", "dataset_label"], ascending=[False, True, True]).drop(columns=["sort_key"])

    n_rows = len(vis_df)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, max(3.2, 2.8 * n_rows)))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (_, row) in enumerate(vis_df.iterrows()):
        tag = row["run_tag"]
        maps = qual_maps[tag]
        gt_map = maps["gt_map"]
        pred_map = maps["pred_map"]
        err_map = maps["err_map"]
        sigma_map = maps["sigma_map"]

        im0 = axes[r, 0].imshow(gt_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r, 0].set_title("GT (band-avg)")
        axes[r, 0].axis("off")
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.01)

        im1 = axes[r, 1].imshow(pred_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r, 1].set_title("Prediction (band-avg)")
        axes[r, 1].axis("off")
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.01)

        emax = float(np.percentile(err_map, 99))
        im2 = axes[r, 2].imshow(err_map, cmap="magma", vmin=0.0, vmax=max(1e-6, emax))
        axes[r, 2].set_title("Abs Error (band-avg)")
        axes[r, 2].axis("off")
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.01)

        smax = float(np.percentile(sigma_map, 99))
        im3 = axes[r, 3].imshow(sigma_map, cmap="plasma", vmin=0.0, vmax=max(1e-6, smax))
        axes[r, 3].set_title("Sigma (band-avg)")
        axes[r, 3].axis("off")
        fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.01)

        mode = "IN" if bool(row["is_in_distribution"]) else "CROSS"
        label = (
            f"{tag} [{mode}]\n"
            f"MAE={row['avg_mae']:.4f}, SSIM={row['avg_ssim_3d']:.4f}, "
            f"SIG={row['avg_sigma_mean']:.4f}"
        )
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=92, va="center", fontsize=8.5)

    fig.suptitle("Qualitative Hallucination Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".svg"), dpi=300)
    plt.close(fig)


def build_delta_table(summary_df: pd.DataFrame, in_dataset_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for model_label in sorted(summary_df["model_label"].unique()):
        md = summary_df[summary_df["model_label"] == model_label]
        in_label = in_dataset_map[model_label]
        in_row = md[md["dataset_label"] == in_label]
        out_row = md[md["dataset_label"] != in_label]
        if in_row.empty or out_row.empty:
            continue
        a = in_row.iloc[0]
        b = out_row.iloc[0]
        rows.append({
            "model_label": model_label,
            "in_dataset": in_label,
            "cross_dataset": str(b["dataset_label"]),
            "in_mae": float(a["avg_mae"]),
            "cross_mae": float(b["avg_mae"]),
            "delta_mae": float(b["avg_mae"] - a["avg_mae"]),
            "in_ssim_3d": float(a["avg_ssim_3d"]),
            "cross_ssim_3d": float(b["avg_ssim_3d"]),
            "delta_ssim_3d": float(b["avg_ssim_3d"] - a["avg_ssim_3d"]),
            "in_sigma_mean": float(a["avg_sigma_mean"]),
            "cross_sigma_mean": float(b["avg_sigma_mean"]),
            "delta_sigma_mean": float(b["avg_sigma_mean"] - a["avg_sigma_mean"]),
            "sigma_ratio_cross_over_in": float((b["avg_sigma_mean"] + 1e-12) / (a["avg_sigma_mean"] + 1e-12)),
        })
    return pd.DataFrame(rows)


def write_markdown_report(path: str, summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Hallucination Test Report")
    lines.append("")
    lines.append("This compares in-distribution vs cross-dataset reconstruction for two trained models.")
    lines.append("")
    lines.append("## Per-run Summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Cross-vs-In Deltas (per model)")
    lines.append("")
    if delta_df.empty:
        lines.append("No delta rows available.")
    else:
        lines.append(delta_df.to_markdown(index=False))
        lines.append("")
        lines.append("### Key Interpretation")
        for _, r in delta_df.iterrows():
            lines.append(
                f"- `{r['model_label']}` on cross dataset `{r['cross_dataset']}`: "
                f"MAE Δ={r['delta_mae']:+.4f}, SSIM Δ={r['delta_ssim_3d']:+.4f}, "
                f"Sigma Δ={r['delta_sigma_mean']:+.4f}, Sigma ratio={r['sigma_ratio_cross_over_in']:.2f}x."
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description="Run cross-dataset hallucination analysis between two trained models.")
    p.add_argument("--model_a_label", type=str, default="banknotes_model", help="Short label for model A")
    p.add_argument("--model_a_name", type=str, required=True, help="Experiment name for model A checkpoints")
    p.add_argument("--model_a_checkpoints_dir", type=str, required=True, help="checkpoints_dir passed to test.py for model A")
    p.add_argument("--dataset_a_label", type=str, default="banknotes", help="Short label for dataset A")
    p.add_argument("--dataset_a_dataroot", type=str, required=True, help="Dataroot for dataset A")

    p.add_argument("--model_b_label", type=str, default="produce_model", help="Short label for model B")
    p.add_argument("--model_b_name", type=str, required=True, help="Experiment name for model B checkpoints")
    p.add_argument("--model_b_checkpoints_dir", type=str, required=True, help="checkpoints_dir passed to test.py for model B")
    p.add_argument("--dataset_b_label", type=str, default="produce", help="Short label for dataset B")
    p.add_argument("--dataset_b_dataroot", type=str, required=True, help="Dataroot for dataset B")

    p.add_argument("--phase", type=str, default="validation", help="Split/phase used for testing")
    p.add_argument("--epoch", type=str, default="latest", help="Checkpoint epoch")
    p.add_argument("--polarization", type=int, default=0, help="Polarization channel")
    p.add_argument("--max_images", type=int, default=None, help="Optional cap on evaluated images")

    p.add_argument("--python_exe", type=str, default="python", help="Python executable")
    p.add_argument("--test_script", type=str, default="test.py", help="Path to test.py")
    p.add_argument("--eval_script", type=str, default="HSI_comparison_probabalistic.py", help="Path to aggregate eval script")
    p.add_argument("--per_image_script", type=str, default="HSI_comparison_probabalistic_per_image.py", help="Path to per-image eval script")
    p.add_argument("--cwd", type=str, default=".", help="Working directory for script execution")
    p.add_argument("--extra_test_opts", nargs="*", default=[], help="Additional args to append to test.py")
    p.add_argument("--no_default_test_opts", action="store_true", help="Disable built-in test opts")
    p.add_argument("--reuse_existing", action="store_true", help="Skip rerunning test/eval if CSVs already exist")
    p.add_argument("--keep_generated_images", action="store_true", help="Keep per-case generated images under results_root/runs (default deletes them after each case)")

    p.add_argument("--results_root", type=str, default="hallucination_analysis_results", help="Output root directory")
    p.add_argument("--qual_image_index", type=int, default=0, help="Image index for qualitative panel")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    runs_root = os.path.join(args.results_root, "runs")
    metrics_root = os.path.join(args.results_root, "metrics")
    per_image_root = os.path.join(args.results_root, "per_image")
    plots_root = os.path.join(args.results_root, "plots")
    for d in [runs_root, metrics_root, per_image_root, plots_root]:
        os.makedirs(d, exist_ok=True)

    case_aa = EvalCase(
        model_label=args.model_a_label,
        model_name=args.model_a_name,
        checkpoints_dir=args.model_a_checkpoints_dir,
        dataset_label=args.dataset_a_label,
        dataroot=args.dataset_a_dataroot,
    )
    case_ab = EvalCase(
        model_label=args.model_a_label,
        model_name=args.model_a_name,
        checkpoints_dir=args.model_a_checkpoints_dir,
        dataset_label=args.dataset_b_label,
        dataroot=args.dataset_b_dataroot,
    )
    case_bb = EvalCase(
        model_label=args.model_b_label,
        model_name=args.model_b_name,
        checkpoints_dir=args.model_b_checkpoints_dir,
        dataset_label=args.dataset_b_label,
        dataroot=args.dataset_b_dataroot,
    )
    case_ba = EvalCase(
        model_label=args.model_b_label,
        model_name=args.model_b_name,
        checkpoints_dir=args.model_b_checkpoints_dir,
        dataset_label=args.dataset_a_label,
        dataroot=args.dataset_a_dataroot,
    )
    cases = [case_aa, case_ab, case_bb, case_ba]

    in_dataset_map = {
        args.model_a_label: args.dataset_a_label,
        args.model_b_label: args.dataset_b_label,
    }

    summary_rows: List[Dict[str, float]] = []
    per_image_rows: List[pd.DataFrame] = []
    qual_maps: Dict[str, Dict[str, np.ndarray]] = {}

    for case in cases:
        print(f"\n=== Case: {case.run_tag} ===")
        case_results_root = os.path.join(runs_root, case.run_tag)
        metrics_csv = os.path.join(metrics_root, f"{case.run_tag}.csv")
        per_image_csv = os.path.join(per_image_root, f"{case.run_tag}.csv")

        eval_dir, metrics = run_case(
            case=case,
            args=args,
            case_results_root=case_results_root,
            metrics_csv=metrics_csv,
            per_image_csv=per_image_csv,
        )
        if eval_dir is not None:
            try:
                qual_maps[case.run_tag] = build_qualitative_maps(eval_dir, image_index=args.qual_image_index)
            except Exception as e:
                print(f"Warning: Could not cache qualitative map for {case.run_tag}: {e}")
        else:
            print(f"Warning: No generated image directory available for qualitative panel in {case.run_tag}.")

        is_in = int(case.dataset_label == in_dataset_map[case.model_label])
        row = {
            "run_tag": case.run_tag,
            "model_label": case.model_label,
            "model_name": case.model_name,
            "dataset_label": case.dataset_label,
            "dataroot": case.dataroot,
            "is_in_distribution": is_in,
            **metrics,
        }
        summary_rows.append(row)

        if os.path.isfile(per_image_csv):
            pi = pd.read_csv(per_image_csv)
            pi["run_tag"] = case.run_tag
            pi["model_label"] = case.model_label
            pi["dataset_label"] = case.dataset_label
            pi["is_in_distribution"] = is_in
            per_image_rows.append(pi)

        if not args.keep_generated_images and os.path.isdir(case_results_root):
            safe_remove_path(case_results_root)
            print(f"Deleted generated reconstructions: {case_results_root}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.results_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote summary: {summary_csv}")

    per_image_df = pd.concat(per_image_rows, ignore_index=True) if per_image_rows else pd.DataFrame()
    per_image_all_csv = os.path.join(args.results_root, "per_image_all.csv")
    per_image_df.to_csv(per_image_all_csv, index=False)
    print(f"Wrote per-image aggregate: {per_image_all_csv}")

    delta_df = build_delta_table(summary_df, in_dataset_map=in_dataset_map)
    delta_csv = os.path.join(args.results_root, "delta_cross_vs_in.csv")
    delta_df.to_csv(delta_csv, index=False)
    print(f"Wrote delta table: {delta_csv}")

    heatmap_png = os.path.join(plots_root, "hallucination_heatmaps.png")
    save_heatmaps(
        summary_df=summary_df,
        labels=(args.dataset_a_label, args.dataset_b_label),
        out_png=heatmap_png,
    )
    bars_png = os.path.join(plots_root, "in_vs_cross_bars.png")
    save_in_vs_cross_bars(summary_df=summary_df, in_dataset_map=in_dataset_map, out_png=bars_png)
    sigma_box_png = os.path.join(plots_root, "per_image_sigma_boxplot.png")
    save_per_image_sigma_boxplot(per_image_df=per_image_df, out_png=sigma_box_png)
    qual_png = os.path.join(plots_root, "qualitative_panel.png")
    save_qualitative_panel(summary_df=summary_df, qual_maps=qual_maps, out_png=qual_png)

    report_md = os.path.join(args.results_root, "report.md")
    write_markdown_report(path=report_md, summary_df=summary_df, delta_df=delta_df)
    print(f"Wrote markdown report: {report_md}")


if __name__ == "__main__":
    main()
