#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine and analyze σ–MAE correlations across multiple probabilistic U-Net datasets.
Generates per-polarization fits, per-dataset combined fits, one global combined plot,
and a 2×3 grid summary of per-pixel σ–MAE correlations across datasets.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# ============================================================
# CONFIGURATION
# ============================================================
FILES = [
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_banknotes_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_banknotes_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_banknotes_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_banknotes_augmented_pol135.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_invertebrates_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_invertebrates_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_invertebrates_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_invertebrates_augmented_pol135.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_produce_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_produce_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_produce_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_produce_augmented_pol135.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_rescharts_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_rescharts_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_rescharts_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_rescharts_augmented_pol135.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_william_summer_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_william_summer_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_william_summer_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_william_summer_augmented_pol135.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_cumulative_augmented_pol0.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_cumulative_augmented_pol45.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_cumulative_augmented_pol90.npz",
    r"Probabalistic_UNET\metrics_prob_nll\pixels_prob_cumulative_augmented_pol135.npz",
]
OUT_DIR = r"Probabalistic_UNET\metrics_prob_nll\combined_results"
SUMMARY_CSV = os.path.join(OUT_DIR, "sigma_mae_summary.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Pretty display names
DISPLAY_NAMES = {
    "banknotes": "Banknotes",
    "invertebrates": "Invertebrates",
    "produce": "Produce",
    "rescharts": "Resolution Charts",
    "william_summer": "Fossils/Flora",
    "cumulative": "Unified"
}

# ============================================================
# FUNCTIONS
# ============================================================

def fit_sigma_mae(sigma, mae):
    """Perform linear regression MAE = aσ + b and compute stats."""
    mask = np.isfinite(sigma) & np.isfinite(mae) & (sigma > 0) & (mae > 0)
    sigma, mae = sigma[mask], mae[mask]
    if len(sigma) < 10:
        return None, None

    model = LinearRegression().fit(sigma.reshape(-1, 1), mae)
    r2 = model.score(sigma.reshape(-1, 1), mae)
    r, _ = pearsonr(sigma, mae)
    return {"a": model.coef_[0], "b": model.intercept_, "r2": r2, "pearson_r": r, "n": len(sigma)}, model


def plot_fit(sigma, mae, model, title, fname, stats):
    """Plot σ–MAE calibration with regression line and save to PNG+SVG."""
    plt.figure(figsize=(6.5, 5.5))
    hb = plt.hexbin(sigma, mae, gridsize=250, bins='log', cmap='plasma')
    cbar = plt.colorbar(hb)
    cbar.set_label("log(pixel count)", fontsize=10)

    plt.title(title, fontsize=13, fontweight='semibold')
    plt.xlabel("Predicted Uncertainty (σ)")
    plt.ylabel("Observed Mean Absolute Error (MAE)")

    xfit = np.linspace(np.min(sigma), np.max(sigma), 200)
    yfit = model.predict(xfit.reshape(-1, 1))
    plt.plot(xfit, yfit, "k-", lw=2.2)
    plt.plot(xfit, yfit, "w-", lw=0.8, alpha=0.7)

    text = (f"MAE = {stats['a']:.3f}σ + {stats['b']:.4f}\n"
            f"R² = {stats['r2']:.3f},  r = {stats['pearson_r']:.3f}\n"
            f"n = {stats['n']:,}")
    plt.gca().text(0.97, 0.93, text, transform=plt.gca().transAxes,
                   ha="right", va="top",
                   bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.4"))
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, f"{fname}.png"), dpi=450)
    plt.savefig(os.path.join(OUT_DIR, f"{fname}.svg"), dpi=450)
    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

results = []
datasets = {}
sigma_global, mae_global = [], []

# --- Group by dataset ---
for file in FILES:
    m = re.search(r"pixels_prob_(.*?)_augmented_pol(\d+)\.npz", os.path.basename(file))
    if not m:
        continue
    dataset, pol = m.group(1), int(m.group(2))
    datasets.setdefault(dataset, []).append((pol, file))

# --- Process each dataset ---
for dataset, entries in datasets.items():
    sigma_all, mae_all = [], []
    name = DISPLAY_NAMES.get(dataset, dataset.title())
    print(f"\nProcessing dataset: {name}")

    for pol, path in sorted(entries):
        data = np.load(path)
        sigma, mae = data["sigma"], data["mae"]
        sigma_all.append(sigma)
        mae_all.append(mae)
        sigma_global.append(sigma)
        mae_global.append(mae)

        stats, model = fit_sigma_mae(sigma, mae)
        if stats:
            results.append({"dataset": name, "polarization": pol, **stats})
            title = f"{name} — σ–MAE Calibration ({pol}°)"
            plot_fit(sigma, mae, model, title, f"{dataset}_pol{pol}", stats)
            print(f"  pol{pol}: r={stats['pearson_r']:.3f}, R²={stats['r2']:.3f}, n={stats['n']:,}")

    # Combined per-dataset fit
    sigma_all = np.concatenate(sigma_all)
    mae_all = np.concatenate(mae_all)
    stats, model = fit_sigma_mae(sigma_all, mae_all)
    if stats:
        results.append({"dataset": name, "polarization": "avg", **stats})
        title = f"{name} — σ–MAE Calibration (All Polarizations)"
        plot_fit(sigma_all, mae_all, model, title, f"{dataset}_avg", stats)
        print(f"  → Combined: r={stats['pearson_r']:.3f}, R²={stats['r2']:.3f}, n={stats['n']:,}")

# ============================================================
# GLOBAL COMBINED FIT (ALL DATASETS & POLARIZATIONS)
# ============================================================

sigma_global = np.concatenate(sigma_global)
mae_global = np.concatenate(mae_global)

global_stats, global_model = fit_sigma_mae(sigma_global, mae_global)
if global_stats:
    results.append({"dataset": "All Combined", "polarization": "all", **global_stats})
    title = "Global σ–MAE Calibration (All Datasets & Polarizations)"
    plot_fit(sigma_global, mae_global, global_model, title, "global_all_combined", global_stats)
    print("\n=== Global Combined Fit ===")
    print(f"a={global_stats['a']:.4f}, b={global_stats['b']:.4f}, R²={global_stats['r2']:.4f}, "
          f"r={global_stats['pearson_r']:.4f}, n={global_stats['n']:,}")

# ============================================================
# MULTI-PANEL SUMMARY FIGURE (2 ROWS × 3 COLUMNS)
# ============================================================

selected_datasets = [d for d in datasets.keys() if d in DISPLAY_NAMES][:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes = axes.flatten()

for i, dataset in enumerate(selected_datasets):
    sigma_all, mae_all = [], []
    name = DISPLAY_NAMES.get(dataset, dataset.title())
    for pol, path in sorted(datasets[dataset]):
        data = np.load(path)
        sigma_all.append(data["sigma"])
        mae_all.append(data["mae"])
    sigma_all = np.concatenate(sigma_all)
    mae_all = np.concatenate(mae_all)

    stats, model = fit_sigma_mae(sigma_all, mae_all)
    ax = axes[i]
    hb = ax.hexbin(sigma_all, mae_all, gridsize=200, bins='log', cmap='plasma')
    ax.set_title(name, fontsize=11, fontweight="semibold")
    ax.set_xlabel("Predicted Uncertainty (σ)")
    ax.set_ylabel("Observed MAE")

    if model:
        xfit = np.linspace(np.min(sigma_all), np.max(sigma_all), 200)
        yfit = model.predict(xfit.reshape(-1, 1))
        ax.plot(xfit, yfit, "k-", lw=2.0)
        ax.plot(xfit, yfit, "w-", lw=0.8, alpha=0.6)

        # Updated annotation with n included
        ax.text(
            0.97, 0.93,
            f"$r$ = {stats['pearson_r']:.2f}\n"
            f"$R^2$ = {stats['r2']:.2f}\n"
            f"$n$ = {stats['n']:,}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3")
        )

# Hide any unused subplots
for j in range(len(selected_datasets), len(axes)):
    fig.delaxes(axes[j])

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(hb, cax=cbar_ax)
cb.set_label("log(pixel count)", fontsize=10)

fig.suptitle("Per-pixel σ–MAE Correlation Across Datasets (All Polarizations)", fontsize=14, fontweight="semibold")
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

out_path = os.path.join(OUT_DIR, "sigma_mae_summary_grid.png")
plt.savefig(out_path, dpi=450)
plt.savefig(out_path.replace(".png", ".svg"), dpi=450)
plt.close()

print(f"\nSaved multi-panel σ–MAE summary grid to:\n{out_path}")


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

df = pd.DataFrame(results)
df.to_csv(SUMMARY_CSV, index=False)
print("\n=== Summary saved ===")
print(SUMMARY_CSV)
print(df.groupby("dataset")[["pearson_r", "r2"]].mean().round(3))
