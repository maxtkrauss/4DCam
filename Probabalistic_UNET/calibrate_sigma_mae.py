#!/usr/bin/env python3
"""
Combine and analyze sigma-MAE correlations across multiple probabilistic U-Net datasets.

Outputs:
- Per-polarization calibration plots
- Per-dataset combined calibration plots
- Global combined calibration plot
- A 2x3 summary grid across datasets

For every plot above, this script writes:
- a standard linear-axis version
- a companion `_log` version with log-scaled x/y axes
"""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "metrics_prob_nll"
DATA_DIR = DEFAULT_DATA_DIR

FILES = [
    DATA_DIR / "pixels_prob_banknotes_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_banknotes_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_banknotes_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_banknotes_augmented_pol135.npz",
    DATA_DIR / "pixels_prob_invertebrates_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_invertebrates_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_invertebrates_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_invertebrates_augmented_pol135.npz",
    DATA_DIR / "pixels_prob_produce_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_produce_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_produce_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_produce_augmented_pol135.npz",
    DATA_DIR / "pixels_prob_rescharts_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_rescharts_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_rescharts_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_rescharts_augmented_pol135.npz",
    DATA_DIR / "pixels_prob_william_summer_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_william_summer_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_william_summer_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_william_summer_augmented_pol135.npz",
    DATA_DIR / "pixels_prob_cumulative_augmented_pol0.npz",
    DATA_DIR / "pixels_prob_cumulative_augmented_pol45.npz",
    DATA_DIR / "pixels_prob_cumulative_augmented_pol90.npz",
    DATA_DIR / "pixels_prob_cumulative_augmented_pol135.npz",
]

DISPLAY_NAMES = {
    "banknotes": "Banknotes",
    "banknotes_augmented": "Banknotes",
    "invertebrates": "Invertebrates",
    "invertebrates_augmented": "Invertebrates",
    "produce": "Produce",
    "produce_augmented": "Produce",
    "rescharts": "Resolution Charts",
    "rescharts_augmented": "Resolution Charts",
    "william_summer": "Fossils/Flora",
    "fossils&fauna": "Fossils/Flora",
    "cumulative": "Unified",
    "cumulative_augmented": "Unified",
}


def fit_sigma_mae(sigma: np.ndarray, mae: np.ndarray):
    """Perform linear regression MAE = a*sigma + b and compute stats."""
    mask = np.isfinite(sigma) & np.isfinite(mae) & (sigma > 0) & (mae > 0)
    sigma = np.asarray(sigma[mask], dtype=np.float64)
    mae = np.asarray(mae[mask], dtype=np.float64)
    if len(sigma) < 10:
        return None, None

    model = LinearRegression().fit(sigma.reshape(-1, 1), mae)
    r2 = float(model.score(sigma.reshape(-1, 1), mae))
    r, _ = pearsonr(sigma, mae)
    stats = {
        "a": float(model.coef_[0]),
        "b": float(model.intercept_),
        "r2": r2,
        "pearson_r": float(r),
        "n": int(len(sigma)),
    }
    return stats, model


def _plot_regression_line(ax, sigma: np.ndarray, model: LinearRegression, use_log: bool) -> None:
    xfit = np.linspace(float(np.min(sigma)), float(np.max(sigma)), 200)
    yfit = model.predict(xfit.reshape(-1, 1))
    if use_log:
        yfit = np.clip(yfit, 1e-12, None)
    ax.plot(xfit, yfit, "k-", lw=2.0)
    ax.plot(xfit, yfit, "w-", lw=0.8, alpha=0.65)


def plot_fit(
    sigma: np.ndarray,
    mae: np.ndarray,
    model: LinearRegression,
    title: str,
    fname: str,
    stats: dict,
    out_dir: Path,
) -> None:
    """Save both linear-axis and log-axis versions of one calibration plot."""
    for use_log in [False, True]:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        hb = ax.hexbin(sigma, mae, gridsize=250, bins="log", cmap="plasma")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("log(pixel count)", fontsize=10)

        ax.set_title(f"{title}{' (log scales)' if use_log else ''}", fontsize=13, fontweight="semibold")
        ax.set_xlabel("Predicted Uncertainty (sigma)")
        ax.set_ylabel("Observed Mean Absolute Error (MAE)")

        _plot_regression_line(ax, sigma, model, use_log=use_log)

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        text = (
            f"MAE = {stats['a']:.3f}*sigma + {stats['b']:.4f}\n"
            f"R^2 = {stats['r2']:.3f},  r = {stats['pearson_r']:.3f}\n"
            f"n = {stats['n']:,}"
        )
        ax.text(
            0.97,
            0.93,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.4"),
        )
        fig.tight_layout()

        suffix = "_log" if use_log else ""
        fig.savefig(out_dir / f"{fname}{suffix}.png", dpi=450)
        fig.savefig(out_dir / f"{fname}{suffix}.svg", dpi=450)
        plt.close(fig)


def plot_summary_grid(datasets_grouped: dict, use_log: bool, out_dir: Path) -> Path:
    selected_datasets = [d for d in datasets_grouped.keys() if d in DISPLAY_NAMES][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    hb = None

    for i, dataset in enumerate(selected_datasets):
        sigma_all, mae_all = [], []
        name = DISPLAY_NAMES.get(dataset, dataset.title())
        for pol, path in sorted(datasets_grouped[dataset]):
            data = np.load(path)
            sigma_all.append(data["sigma"])
            mae_all.append(data["mae"])

        sigma_all = np.concatenate(sigma_all)
        mae_all = np.concatenate(mae_all)
        stats, model = fit_sigma_mae(sigma_all, mae_all)

        ax = axes[i]
        hb = ax.hexbin(sigma_all, mae_all, gridsize=200, bins="log", cmap="plasma")
        ax.set_title(name, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Predicted Uncertainty (sigma)")
        ax.set_ylabel("Observed MAE")

        if model is not None:
            _plot_regression_line(ax, sigma_all, model, use_log=use_log)
            ax.text(
                0.97,
                0.93,
                f"$r$ = {stats['pearson_r']:.2f}\n"
                f"$R^2$ = {stats['r2']:.2f}\n"
                f"$n$ = {stats['n']:,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            )

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

    for j in range(len(selected_datasets), len(axes)):
        fig.delaxes(axes[j])

    if hb is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig.colorbar(hb, cax=cbar_ax)
        cb.set_label("log(pixel count)", fontsize=10)

    fig.suptitle(
        f"Per-pixel sigma-MAE Correlation Across Datasets (All Polarizations){' - log scales' if use_log else ''}",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    suffix = "_log" if use_log else ""
    out_path = out_dir / f"sigma_mae_summary_grid{suffix}.png"
    fig.savefig(out_path, dpi=450)
    fig.savefig(out_path.with_suffix(".svg"), dpi=450)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine and analyze sigma-MAE pixel calibration NPZ files.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing pixels_prob_<dataset>_pol<angle>.npz files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data_dir>/combined_results.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir / "combined_results"
    summary_csv = out_dir / "sigma_mae_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    datasets_grouped = {}
    sigma_global = []
    mae_global = []

    files = sorted(data_dir.glob("pixels_prob_*_pol*.npz"))
    if not files:
        raise FileNotFoundError(f"No pixels_prob_*_pol*.npz files found in: {data_dir}")

    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing input file: {file}")
        m = re.search(r"pixels_prob_(.*?)_pol(\d+)\.npz", file.name)
        if not m:
            continue
        dataset, pol = m.group(1), int(m.group(2))
        datasets_grouped.setdefault(dataset, []).append((pol, file))

    for dataset, entries in datasets_grouped.items():
        sigma_all = []
        mae_all = []
        name = DISPLAY_NAMES.get(dataset, dataset.title())
        print(f"\nProcessing dataset: {name}")

        for pol, path in sorted(entries):
            data = np.load(path)
            sigma = data["sigma"]
            mae = data["mae"]
            sigma_all.append(sigma)
            mae_all.append(mae)
            sigma_global.append(sigma)
            mae_global.append(mae)

            stats, model = fit_sigma_mae(sigma, mae)
            if stats is not None:
                results.append({"dataset": name, "polarization": pol, **stats})
                title = f"{name} - sigma-MAE Calibration ({pol} deg)"
                plot_fit(sigma, mae, model, title, f"{dataset}_pol{pol}", stats, out_dir)
                print(f"  pol{pol}: r={stats['pearson_r']:.3f}, R^2={stats['r2']:.3f}, n={stats['n']:,}")

        sigma_all = np.concatenate(sigma_all)
        mae_all = np.concatenate(mae_all)
        stats, model = fit_sigma_mae(sigma_all, mae_all)
        if stats is not None:
            results.append({"dataset": name, "polarization": "avg", **stats})
            title = f"{name} - sigma-MAE Calibration (All Polarizations)"
            plot_fit(sigma_all, mae_all, model, title, f"{dataset}_avg", stats, out_dir)
            print(f"  Combined: r={stats['pearson_r']:.3f}, R^2={stats['r2']:.3f}, n={stats['n']:,}")

    sigma_global = np.concatenate(sigma_global)
    mae_global = np.concatenate(mae_global)
    global_stats, global_model = fit_sigma_mae(sigma_global, mae_global)
    if global_stats is not None:
        results.append({"dataset": "All Combined", "polarization": "all", **global_stats})
        title = "Global sigma-MAE Calibration (All Datasets and Polarizations)"
        plot_fit(sigma_global, mae_global, global_model, title, "global_all_combined", global_stats, out_dir)
        print("\n=== Global Combined Fit ===")
        print(
            f"a={global_stats['a']:.4f}, b={global_stats['b']:.4f}, "
            f"R^2={global_stats['r2']:.4f}, r={global_stats['pearson_r']:.4f}, n={global_stats['n']:,}"
        )

    for use_log in [False, True]:
        out_path = plot_summary_grid(datasets_grouped, use_log=use_log, out_dir=out_dir)
        print(f"\nSaved multi-panel sigma-MAE summary grid to:\n{out_path}")

    df = pd.DataFrame(results)
    df.to_csv(summary_csv, index=False)
    print("\n=== Summary saved ===")
    print(summary_csv)
    print(df.groupby("dataset")[["pearson_r", "r2"]].mean().round(3))


if __name__ == "__main__":
    main()
