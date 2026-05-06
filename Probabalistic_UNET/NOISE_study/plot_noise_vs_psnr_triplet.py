#!/usr/bin/env python3
"""
Plot MAE, SSIM, and SIGMA against PSNR for selected noise families.

Input:
  - stability_summary.csv from run_stability_noise_ablation.py

Output:
  - PNG (+ SVG) with three subplots:
      1) MAE vs PSNR
      2) SSIM vs PSNR
      3) Sigma Mean vs PSNR
"""

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLORS: Dict[str, str] = {
    "gaussian": "#1f77b4",     # blue
    "saltpepper": "#ff7f0e",   # orange
    "poisson": "#2ca02c",      # green
}

PRETTY_LABELS: Dict[str, str] = {
    "gaussian": "Gaussian",
    "saltpepper": "Salt-Pepper",
    "poisson": "Poisson",
}


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot MAE/SSIM/SIGMA vs PSNR for noise families.")
    p.add_argument("--summary_csv", type=str, required=True, help="Path to stability_summary.csv")
    p.add_argument("--out_png", type=str, default="", help="Output PNG path (default: alongside summary_csv)")
    p.add_argument(
        "--noise_types",
        type=str,
        default="gaussian,saltpepper,poisson",
        help="Comma-separated noise types to include",
    )
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    return p.parse_args()


def main():
    args = parse_args()
    summary_csv = os.path.abspath(args.summary_csv)
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"summary_csv not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    _require_cols(
        df,
        ["noise_type", "input_psnr_mean", "avg_mae", "avg_ssim_3d", "avg_sigma_mean"],
        "stability_summary.csv",
    )

    noise_types = [x.strip().lower() for x in args.noise_types.split(",") if x.strip()]
    if not noise_types:
        raise ValueError("No valid --noise_types provided.")

    # Keep finite PSNR rows only for plotting x-axis.
    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["input_psnr_mean"]).copy()
    plot_df["noise_type"] = plot_df["noise_type"].astype(str).str.lower()
    plot_df = plot_df[plot_df["noise_type"].isin(noise_types)].copy()
    if plot_df.empty:
        raise RuntimeError("No rows left after filtering by noise type and finite input_psnr_mean.")

    clean_rows = df[df["noise_type"].astype(str).str.lower() == "none"]
    clean = clean_rows.iloc[0] if not clean_rows.empty else None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True)

    metric_specs = [
        ("avg_mae", "MAE", True),
        ("avg_ssim_3d", "SSIM (3D)", False),
        ("avg_sigma_mean", "Sigma Mean (Uncertainty)", False),
    ]

    x_min = float(plot_df["input_psnr_mean"].min())
    x_max = float(plot_df["input_psnr_mean"].max())
    x_pad = max(0.15, 0.03 * (x_max - x_min))

    for ax, (metric_col, y_label, lower_is_better) in zip(axes, metric_specs):
        for nt in noise_types:
            g = plot_df[plot_df["noise_type"] == nt].sort_values("input_psnr_mean", ascending=True)
            if g.empty:
                continue
            color = DEFAULT_COLORS.get(nt, None)
            x_vals = g["input_psnr_mean"].to_numpy(dtype=float)
            y_vals = g[metric_col].to_numpy(dtype=float)
            label = PRETTY_LABELS.get(nt, nt)
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                alpha=0.95,
                label=label,
            )

        # Optional clean baseline (horizontal reference line).
        if clean is not None and metric_col in clean.index and pd.notna(clean[metric_col]):
            ax.axhline(
                float(clean[metric_col]),
                linestyle="--",
                linewidth=1.4,
                color="#555555",
                alpha=0.9,
                label="Clean baseline",
            )

        ax.set_xlabel("Input PSNR (dB)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs PSNR", fontsize=11.5)
        ax.grid(alpha=0.28)
        ax.set_axisbelow(True)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.legend(loc="best", fontsize=8, frameon=True, title="Noise Type")

    fig.tight_layout()

    out_png = args.out_png
    if not out_png:
        out_png = os.path.join(os.path.dirname(summary_csv), "mae_ssim_sigma_vs_psnr.png")
    out_png = os.path.abspath(out_png)
    out_svg = os.path.splitext(out_png)[0] + ".svg"

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_svg, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plot: {out_png}")
    print(f"Wrote plot: {out_svg}")


if __name__ == "__main__":
    main()
