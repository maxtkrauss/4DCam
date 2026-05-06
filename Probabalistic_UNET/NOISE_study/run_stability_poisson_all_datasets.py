#!/usr/bin/env python3
"""
Launch adaptive Poisson-only stability ablations across the current trained dataset models.

Outputs:
  - per-dataset ablation folders written by run_stability_noise_ablation.py
  - aggregate CSV across all datasets
  - aggregate MAE/SSIM/uncertainty vs PSNR plots with one curve per dataset
"""

import argparse
import os
import subprocess
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS: List[Dict[str, str]] = [
    {
        "name": "banknotes_augmented",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/banknotes",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_banknotes_augmented_prob_retrain",
    },
    {
        "name": "fossils&fauna",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/fossils&fauna",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_fossils&fauna_prob_retrain",
    },
    {
        "name": "produce_augmented",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/produce",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/Diff_nll_modelsmodels/checkpoints_produce_augmented_prob_retrain",
    },
    {
        "name": "invertebrates_augmented",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/invertebrates",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_invertebrates_augmented_prob_retrain",
    },
    {
        "name": "rescharts_augmented",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/rescharts",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_rescharts_augmented_prob_retrain",
    },
    {
        "name": "cumulative_augmented",
        "dataroot": "/scratch/general/nfs1/u1528328/datasets/cumulative_split",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_cumulative_augmented_prob_retrain",
    },
]


def sanitize_tag(s: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s.strip())
    return safe.strip("_") or "run"


def run_cmd(cmd: List[str], cwd: str) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def build_dataset_lookup(selected_names: List[str]) -> List[Dict[str, str]]:
    if not selected_names:
        return list(DATASETS)
    wanted = set(selected_names)
    selected = [ds for ds in DATASETS if ds["name"] in wanted]
    missing = sorted(wanted.difference(ds["name"] for ds in selected))
    if missing:
        raise ValueError(f"Unknown dataset names: {missing}")
    return selected


def save_aggregate_triplet(df: pd.DataFrame, out_png: str) -> None:
    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["input_psnr_mean", "avg_mae", "avg_ssim_3d", "avg_sigma_mean"]
    ).copy()
    plot_df = plot_df[plot_df["noise_type"].astype(str).str.lower() == "poisson"]
    if plot_df.empty:
        raise RuntimeError("No finite Poisson rows found for aggregate plotting.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), sharex=True)
    metric_specs = [
        ("avg_mae", "Avg MAE"),
        ("avg_ssim_3d", "Avg SSIM (3D)"),
        ("avg_sigma_mean", "Avg Sigma Mean"),
    ]

    for dataset_name, group in plot_df.groupby("dataset_name"):
        group = group.sort_values("input_psnr_mean", ascending=False)
        x = group["input_psnr_mean"].to_numpy(dtype=float)
        for ax, (metric_col, y_label) in zip(axes, metric_specs):
            y = group[metric_col].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.0, label=dataset_name)
            ax.set_xlabel("Input PSNR (dB)")
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs PSNR")
            ax.grid(alpha=0.25)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="best", fontsize=8, frameon=True, title="Dataset")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote aggregate plot: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive Poisson stability ablations for all configured datasets.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional subset of dataset names to run")
    parser.add_argument("--polarization", type=int, default=0, help="Polarization channel")
    parser.add_argument("--phase", type=str, default="validation", help="Dataset split/phase to evaluate")
    parser.add_argument("--epoch", type=str, default="latest", help="Checkpoint epoch")
    parser.add_argument("--python_exe", type=str, default="python", help="Python executable")
    parser.add_argument("--runner_script", type=str, default="run_stability_noise_ablation.py", help="Poisson ablation runner")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory for script execution")
    parser.add_argument("--results_root", type=str, default="results_stability_all_datasets", help="Base root for lightweight outputs")
    parser.add_argument("--generated_root", type=str, default=None, help="Base root for heavy generated outputs")
    parser.add_argument("--workspace_root", type=str, default=None, help="Base root for temporary noisy dataroots")
    parser.add_argument("--bit_max", type=float, default=4095.0, help="Sensor max used for normalization")
    parser.add_argument("--seed", type=int, default=13, help="Base seed")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of evaluated images")
    parser.add_argument("--noise_steps", type=int, default=10, help="Number of Poisson levels between clean and the target low-PSNR endpoint")
    parser.add_argument("--target_min_psnr", type=float, default=10.0, help="Continue increasing Poisson noise until estimated PSNR drops below this")
    parser.add_argument("--calibration_images", type=int, default=8, help="Number of calibration images used to estimate Poisson severity")
    parser.add_argument("--poisson_start_peak", type=float, default=1024.0, help="Starting Poisson peak")
    parser.add_argument("--poisson_min_peak", type=float, default=1.0, help="Smallest Poisson peak allowed")
    parser.add_argument("--skip_input_noise_figure", action="store_true", help="Skip per-dataset input noise overview figure")
    parser.add_argument("--extra_test_opts", nargs="*", default=[], help="Additional args appended to test.py")
    parser.add_argument("--no_default_test_opts", action="store_true", help="Disable built-in TEST_OPTS")
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse prior per-dataset stability_summary.csv if present")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_datasets = build_dataset_lookup(args.datasets)

    base_results_root = os.path.abspath(args.results_root)
    base_generated_root = os.path.abspath(args.generated_root or os.path.join(base_results_root, "generated"))
    base_workspace_root = os.path.abspath(args.workspace_root or os.path.join(base_generated_root, "_workspace"))
    os.makedirs(base_results_root, exist_ok=True)
    os.makedirs(base_generated_root, exist_ok=True)
    os.makedirs(base_workspace_root, exist_ok=True)

    aggregate_rows: List[pd.DataFrame] = []

    for ds_index, ds in enumerate(selected_datasets):
        dataset_name = ds["name"]
        model_name = f"{dataset_name}_pol{args.polarization}"
        checkpoints_dir = os.path.join(ds["checkpoints_dir"], f"pol{args.polarization}")
        run_tag = sanitize_tag(f"{dataset_name}__{args.phase}__pol{args.polarization}__poisson_lt{args.target_min_psnr:g}")
        run_results_root = os.path.join(base_results_root, "per_dataset")
        summary_csv = os.path.join(run_results_root, run_tag, "stability_summary.csv")

        if args.reuse_existing and os.path.isfile(summary_csv):
            print(f"Reusing existing summary for {dataset_name}: {summary_csv}")
        else:
            cmd = [
                args.python_exe,
                args.runner_script,
                "--dataroot", ds["dataroot"],
                "--checkpoints_dir", checkpoints_dir,
                "--model_name", model_name,
                "--phase", args.phase,
                "--epoch", args.epoch,
                "--polarization", str(args.polarization),
                "--results_root", run_results_root,
                "--generated_root", base_generated_root,
                "--workspace_root", base_workspace_root,
                "--run_tag", run_tag,
                "--bit_max", str(args.bit_max),
                "--seed", str(args.seed + ds_index * 1000),
                "--noise_steps", str(args.noise_steps),
                "--ablation_mode", "poisson_psnr_sweep",
                "--target_min_psnr", str(args.target_min_psnr),
                "--calibration_images", str(args.calibration_images),
                "--poisson_start_peak", str(args.poisson_start_peak),
                "--poisson_min_peak", str(args.poisson_min_peak),
                "--cleanup_noisy_inputs",
                "--skip_reconstruction_grid",
            ]
            if args.max_images is not None:
                cmd += ["--max_images", str(args.max_images)]
            if args.skip_input_noise_figure:
                cmd += ["--skip_input_noise_figure"]
            if args.no_default_test_opts:
                cmd += ["--no_default_test_opts"]
            if args.extra_test_opts:
                cmd += ["--extra_test_opts", *args.extra_test_opts]
            run_cmd(cmd, cwd=args.cwd)

        if not os.path.isfile(summary_csv):
            raise FileNotFoundError(f"Expected summary_csv not found for {dataset_name}: {summary_csv}")

        df = pd.read_csv(summary_csv)
        df["dataset_name"] = dataset_name
        df["dataset_dataroot"] = ds["dataroot"]
        df["dataset_checkpoints_dir"] = checkpoints_dir
        aggregate_rows.append(df)

    aggregate_df = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame()
    aggregate_csv = os.path.join(base_results_root, "all_datasets_poisson_summary.csv")
    aggregate_df.to_csv(aggregate_csv, index=False)
    print(f"Wrote aggregate summary: {aggregate_csv}")

    aggregate_plot = os.path.join(base_results_root, "all_datasets_mae_ssim_sigma_vs_psnr.png")
    save_aggregate_triplet(aggregate_df, aggregate_plot)


if __name__ == "__main__":
    main()
