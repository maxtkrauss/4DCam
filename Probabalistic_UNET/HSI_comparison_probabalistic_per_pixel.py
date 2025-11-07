#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tifffile as tiff
import argparse
import csv
import shutil
from tqdm import tqdm

def compute_metrics(gt, pred, sigma):
    """Compute pixelwise metrics."""
    abs_err = np.abs(gt - pred)       # same shape as cube
    mse = (gt - pred) ** 2
    mae_map = np.mean(abs_err, axis=0)   # average over spectral bands
    mse_map = np.mean(mse, axis=0)
    sigma_map = np.mean(sigma, axis=0)

    return mae_map, mse_map, sigma_map


def main():
    parser = argparse.ArgumentParser(description="Per-pixel Probabilistic HSI comparison")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--output_npz', type=str, required=True, help='Output .npz file path for per-pixel metrics')
    parser.add_argument('--delete_after', action='store_true', help='Delete recon images after evaluation to save space')
    parser.add_argument('--max_images', type=int, default=None, help='Limit number of images to process')
    args = parser.parse_args()

    image_dir = args.results_dir
    os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)

    gt_files = sorted([f for f in os.listdir(image_dir) if f.startswith("cb_raw_") and f.endswith(".tif")])
    pred_files = sorted([f for f in os.listdir(image_dir) if f.startswith("tl_gen_") and f.endswith(".tif")])
    limit = min(len(gt_files), len(pred_files))
    if args.max_images:
        limit = min(limit, args.max_images)

    all_sigma = []
    all_mae = []
    all_mse = []

    for idx in tqdm(range(limit), desc="Evaluating pixelwise metrics"):
        gt_path = os.path.join(image_dir, gt_files[idx])
        pred_path = os.path.join(image_dir, pred_files[idx])

        gt = tiff.imread(gt_path)
        pred_full = tiff.imread(pred_path)
        n_bands = gt.shape[0]
        pred = pred_full[:n_bands]
        sigma = pred_full[n_bands:]

        # Compute per-pixel maps
        mae_map, mse_map, sigma_map = compute_metrics(gt, pred, sigma)

        all_sigma.append(sigma_map.flatten())
        all_mae.append(mae_map.flatten())
        all_mse.append(mse_map.flatten())

        if args.delete_after:
            try:
                os.remove(pred_path)
            except Exception as e:
                print(f"Warning: Could not delete {pred_path}: {e}")

    # Concatenate all into single arrays
    all_sigma = np.concatenate(all_sigma)
    all_mae = np.concatenate(all_mae)
    all_mse = np.concatenate(all_mse)

    # Save compressed NPZ file
    np.savez_compressed(args.output_npz, sigma=all_sigma, mae=all_mae, mse=all_mse)
    print(f"Saved per-pixel σ–MAE database to {args.output_npz}")
    print(f"Total pixels: {len(all_sigma):,}")

if __name__ == "__main__":
    main()
