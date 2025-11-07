# HSI_comparison_prob_per_image.py
import os
import numpy as np
import tifffile as tiff
import argparse
import csv
from skimage.metrics import structural_similarity as ssim

# ----- helpers: copied exactly from the batch script -----
def compute_rase(gt, pred):
    N = gt.shape[0]
    mse_per_band = np.mean((gt - pred) ** 2, axis=(1, 2))
    mean_spectral_value = np.mean(gt)
    if mean_spectral_value == 0:
        return np.nan
    rase = (100 / N) * np.sqrt(np.sum(mse_per_band) / (mean_spectral_value ** 2))
    return rase

def spectral_fidelity(gt, pred):
    n_bands, height, width = gt.shape
    pixel_fidelity = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            gt_spectrum = gt[:, y, x]
            pred_spectrum = pred[:, y, x]
            if np.sum(gt_spectrum**2) == 0 or np.sum(pred_spectrum**2) == 0:
                pixel_fidelity[y, x] = 0
                continue
            dot_product = np.sum(gt_spectrum * pred_spectrum)
            gt_norm = np.sqrt(np.sum(gt_spectrum**2))
            pred_norm = np.sqrt(np.sum(pred_spectrum**2))
            pixel_fidelity[y, x] = dot_product / (gt_norm * pred_norm)
    return np.mean(pixel_fidelity)

def relative_spectral_error_l1(gt, pred):
    bands, height, width = gt.shape
    gt_flat = gt.reshape(bands, -1)
    pred_flat = pred.reshape(bands, -1)
    l1_diff = np.abs(gt_flat - pred_flat).sum(axis=0)
    l1_gt = np.abs(gt_flat).sum(axis=0)
    eps = 1e-8
    relative_errors = l1_diff / (l1_gt + eps)
    return relative_errors.mean()

def spectral_reconstruction_error_l1(gt, pred):
    return np.mean(np.abs(gt - pred))

def spectral_reconstruction_error_l2(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

def relative_spectral_reconstruction_error(gt, pred, norm='l1'):
    if norm == 'l1':
        sre = spectral_reconstruction_error_l1(gt, pred)
        gt_magnitude = np.mean(np.abs(gt))
    elif norm == 'l2':
        sre = spectral_reconstruction_error_l2(gt, pred)
        gt_magnitude = np.sqrt(np.mean(gt ** 2))
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
    if gt_magnitude == 0:
        return np.nan
    return (sre / gt_magnitude) * 100

def analyze_sigma(sigma):
    return {
        "sigma_mean": float(np.mean(sigma)),
        "sigma_std": float(np.std(sigma)),
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "sigma_median": float(np.median(sigma)),
    }

def main():
    parser = argparse.ArgumentParser(description="Per-image Probabilistic HSI comparison (mirrors batch script).")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to process (cap)')
    parser.add_argument('--per_image_csv', type=str, required=True, help='Path to output per-image CSV')
    args = parser.parse_args()

    image_dir = args.results_dir
    num_images = args.num_images
    os.makedirs(os.path.dirname(args.per_image_csv) or '.', exist_ok=True)

    # --- pairing: identical to batch script (sort + parallel index) ---
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    gt_files = [f for f in image_files if f.startswith("cb_raw_")]
    pred_files = [f for f in image_files if f.startswith("tl_gen_")]
    gt_files.sort()
    pred_files.sort()

    rows = []
    limit = min(num_images, len(gt_files), len(pred_files))

    for idx in range(limit):
        gt = tiff.imread(os.path.join(image_dir, gt_files[idx]))
        pred_full = tiff.imread(os.path.join(image_dir, pred_files[idx]))

        # Split mean and sigma exactly like batch script
        pred = pred_full[:gt.shape[0], :, :]
        sigma = pred_full[gt.shape[0]:, :, :]

        # Metrics (identical formulae and data_range)
        ssim_3d = ssim(gt, pred, data_range=gt.max() - gt.min(), multichannel=False)
        ssim_2d = ssim(gt[0], pred[0], data_range=gt[0].max() - gt[0].min())
        mse = np.mean((gt - pred) ** 2)
        mae = np.mean(np.abs(gt - pred))
        rase = compute_rase(gt, pred)
        fidelity = spectral_fidelity(gt, pred)
        RSE = relative_spectral_error_l1(gt, pred)
        sre_l1 = spectral_reconstruction_error_l1(gt, pred)
        sre_l2 = spectral_reconstruction_error_l2(gt, pred)
        rsre_l1 = relative_spectral_reconstruction_error(gt, pred, norm='l1')
        rsre_l2 = relative_spectral_reconstruction_error(gt, pred, norm='l2')

        sigma_stats = analyze_sigma(sigma)

        # Per-image row: same metric names as batch (minus the "avg_" prefix)
        rows.append({
            "image_index": idx,
            "gt_file": gt_files[idx],
            "pred_file": pred_files[idx],
            "ssim_3d": float(ssim_3d),
            "ssim_2d": float(ssim_2d),
            "mse": float(mse),
            "mae": float(mae),
            "rase": float(rase),
            "fidelity": float(fidelity),
            "RSE": float(RSE),
            "sre_l1": float(sre_l1),
            "sre_l2": float(sre_l2),
            "rsre_l1": float(rsre_l1),
            "rsre_l2": float(rsre_l2),
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
            "sigma_median": sigma_stats["sigma_median"],
        })

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.per_image_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.per_image_csv}")
    else:
        print("No valid image pairs found in the specified folder.")

if __name__ == "__main__":
    main()
