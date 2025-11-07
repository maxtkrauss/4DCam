import os
import numpy as np
import tifffile as tiff
import argparse
import csv
from skimage.metrics import structural_similarity as ssim
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import re


def compute_rase(gt, pred, eps=1e-12):
    """
    RASE = (100 / mean_intensity) * sqrt( mean_over_bands( MSE_band ) )
    gt, pred: [C, H, W]
    """
    C = gt.shape[0]
    mse_per_band = np.mean((gt - pred) ** 2, axis=(1, 2))      # [C]
    mu = float(np.mean(gt))
    return float((100.0 / (mu + eps)) * np.sqrt(np.mean(mse_per_band)))


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

def relative_spectral_error_l1(gt, pred, eps=1e-6):
    """
    Per-pixel relative spectral error (L1):
      RSE = mean_over_pixels(  ||gt - pred||_1  /  ||gt||_1  ),
    masking pixels whose GT spectral L1 is ~0 to avoid blow-ups.
    """
    C, H, W = gt.shape
    gt_flat   = gt.reshape(C, -1)          # [C, HW]
    pred_flat = pred.reshape(C, -1)        # [C, HW]

    denom = np.sum(np.abs(gt_flat), axis=0)           # ||gt||_1 per pixel
    num   = np.sum(np.abs(gt_flat - pred_flat), axis=0)  # ||gt - pred||_1 per pixel

    valid = denom > eps
    if not np.any(valid):
        return np.nan

    rse_pixels = num[valid] / denom[valid]
    return float(np.mean(rse_pixels))      # (×100 if you prefer percent)


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
    parser = argparse.ArgumentParser(description="Batch Probabilistic HSI comparison and metrics export.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to process')
    parser.add_argument('--metrics_csv', type=str, default=None, help='Path to output metrics CSV (default: results_dir/metrics.csv)')
    args = parser.parse_args()

    image_dir = args.results_dir
    num_images = args.num_images
    metrics_csv = args.metrics_csv or os.path.join(image_dir, 'metrics.csv')
    # Ensure output dir exists
    metrics_dir = os.path.dirname(metrics_csv)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    gt_files = [f for f in image_files if f.startswith("cb_raw_")]
    pred_files = [f for f in image_files if f.startswith("tl_gen_")]

    # Natural sort if needed
    gt_files.sort()
    pred_files.sort()

    ssim_3d_list, ssim_2d_list, mse_list, mae_list, rase_list, fidelity_list = [], [], [], [], [], []
    RSE_list, sre_l1_list, sre_l2_list, rsre_l1_list, rsre_l2_list = [], [], [], [], []
    sigma_means, sigma_stds, sigma_mins, sigma_maxs, sigma_medians = [], [], [], [], []

    for idx in range(min(num_images, len(gt_files), len(pred_files))):
        gt = tiff.imread(os.path.join(image_dir, gt_files[idx]))
        pred_full = tiff.imread(os.path.join(image_dir, pred_files[idx]))
        # Split mean and sigma
        pred = pred_full[:gt.shape[0], :, :]
        sigma = pred_full[gt.shape[0]:, :, :]

        # Metrics on mean prediction
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

        ssim_3d_list.append(ssim_3d)
        ssim_2d_list.append(ssim_2d)
        mse_list.append(mse)
        mae_list.append(mae)
        rase_list.append(rase)
        fidelity_list.append(fidelity)
        RSE_list.append(RSE)
        sre_l1_list.append(sre_l1)
        sre_l2_list.append(sre_l2)
        rsre_l1_list.append(rsre_l1)
        rsre_l2_list.append(rsre_l2)

        # Uncertainty analysis
        sigma_stats = analyze_sigma(sigma)
        sigma_means.append(sigma_stats["sigma_mean"])
        sigma_stds.append(sigma_stats["sigma_std"])
        sigma_mins.append(sigma_stats["sigma_min"])
        sigma_maxs.append(sigma_stats["sigma_max"])
        sigma_medians.append(sigma_stats["sigma_median"])

    # Aggregate metrics
    metrics = {
        "avg_ssim_3d": np.mean(ssim_3d_list),
        "avg_ssim_2d": np.mean(ssim_2d_list),
        "avg_mse": np.mean(mse_list),
        "avg_mae": np.mean(mae_list),
        "avg_rase": np.mean(rase_list),
        "avg_fidelity": np.mean(fidelity_list),
        "avg_RSE": np.mean(RSE_list),
        "avg_sre_l1": np.mean(sre_l1_list),
        "avg_sre_l2": np.mean(sre_l2_list),
        "avg_rsre_l1": np.mean(rsre_l1_list),
        "avg_rsre_l2": np.mean(rsre_l2_list),
        "avg_sigma_mean": np.mean(sigma_means),
        "avg_sigma_std": np.mean(sigma_stds),
        "avg_sigma_min": np.mean(sigma_mins),
        "avg_sigma_max": np.mean(sigma_maxs),
        "avg_sigma_median": np.mean(sigma_medians),
        "num_images": len(ssim_3d_list)
    }

    print(f"\nAverage Metrics Over {len(ssim_3d_list)} Valid Image Pairs:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Write metrics to CSV
    with open(metrics_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics written to {metrics_csv}")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import re

def natural_sort_key(s):
    """Sort strings that contain numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class InteractiveProbabilisticHSIPlot:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.image_files.sort(key=natural_sort_key)
        self.current_index = 0
        self.selected_points = []
        self.lines = []
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.color_index = 0
        self.view_mode = "Single Wavelength"
        self.current_wavelength_index = 0
        self.load_images()
        self.setup_plot()

    def load_images(self):
        # Find GT and prediction files for current index
        gt_file = next((f for f in self.image_files if f.startswith("cb_raw_") and f.endswith(f"_{self.current_index}.tif")), None)
        pred_file = next((f for f in self.image_files if f.startswith("tl_gen_") and f.endswith(f"_{self.current_index}.tif")), None)
        if not gt_file or not pred_file:
            print(f"Files for index {self.current_index} not found")
            return
        gt_path = os.path.join(self.image_dir, gt_file)
        pred_path = os.path.join(self.image_dir, pred_file)
        self.ground_truth = tiff.imread(gt_path)
        pred_full = tiff.imread(pred_path)
        self.mu = pred_full[:self.ground_truth.shape[0], :, :]
        self.sigma = pred_full[self.ground_truth.shape[0]:, :, :]
        # Normalize for visualization
        gt_min, gt_max = self.ground_truth.min(), self.ground_truth.max()
        self.ground_truth = np.clip((self.ground_truth - gt_min) / (gt_max - gt_min), 0, 1)
        self.mu = np.clip((self.mu - gt_min) / (gt_max - gt_min), 0, 1)
        self.sigma = np.clip(self.sigma, 0, None)
        # Generate wavelength array
        self.wavelengths = np.linspace(450, 850, self.ground_truth.shape[0])

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 7))
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1, 0.1])
        self.ax = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[0, 3])
        ]
        self.slider_ax = self.fig.add_subplot(gs[1, 1:3])
        self.wavelength_slider = Slider(
            self.slider_ax, 'Wavelength (nm)', 0, len(self.wavelengths) - 1, valinit=0, valstep=1
        )
        self.wavelength_slider.on_changed(self.update_wavelength)
        ax_radio = plt.axes([0, 0.10, 0.12, 0.15])
        self.radio = RadioButtons(ax_radio, ['Single Wavelength', 'Sigma View'])
        self.radio.on_clicked(self.change_view_mode)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        button_ax_next = plt.axes([0.4, 0.02, 0.15, 0.05])
        self.btn_next = Button(button_ax_next, "Next")
        self.btn_next.on_clicked(self.load_next_image)
        button_ax_clear = plt.axes([0.6, 0.02, 0.15, 0.05])
        self.btn_clear = Button(button_ax_clear, "Clear Spectra")
        self.btn_clear.on_clicked(self.clear_spectra)
        self.update_plot()
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def update_wavelength(self, val):
        self.current_wavelength_index = int(val)
        self.update_plot()

    def change_view_mode(self, label):
        self.view_mode = label
        self.update_plot()

    def update_plot(self):
        for a in self.ax[:3]:
            a.cla()
        idx = self.current_wavelength_index
        wavelength = self.wavelengths[idx]
        if self.view_mode == "Single Wavelength":
            self.ax[0].imshow(self.ground_truth[idx], cmap='viridis')
            self.ax[0].set_title(f"Ground Truth ({wavelength:.1f}nm)")
            self.ax[1].imshow(self.mu[idx], cmap='viridis')
            self.ax[1].set_title(f"Reconstructed μ ({wavelength:.1f}nm)")
            self.ax[2].imshow(self.sigma[idx], cmap='plasma')
            self.ax[2].set_title(f"Uncertainty σ ({wavelength:.1f}nm)")
        elif self.view_mode == "Sigma View":
            error_img = np.abs(self.ground_truth[idx] - self.mu[idx])
            self.ax[0].imshow(error_img, cmap='hot')
            self.ax[0].set_title(f"Absolute Error ({wavelength:.1f}nm)")
            self.ax[1].imshow(self.mu[idx], cmap='viridis')
            self.ax[1].set_title(f"Reconstructed μ ({wavelength:.1f}nm)")
            self.ax[2].imshow(self.sigma[idx], cmap='plasma')
            self.ax[2].set_title(f"Uncertainty σ ({wavelength:.1f}nm)")
            sigma_img = self.sigma[idx]
            self.ax[2].text(0.02, 0.98, f"Min σ: {sigma_img.min():.4f}\nMax σ: {sigma_img.max():.4f}",
                           transform=self.ax[2].transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        for a in self.ax[:3]:
            a.axis('off')
        self.ax[3].cla()
        self.ax[3].set_title("Spectral Comparison")
        self.ax[3].set_xlabel("Wavelength (nm)")
        self.ax[3].set_ylabel("Intensity")
        for point in self.selected_points:
            x, y, color = point
            gt_spectrum = self.ground_truth[:, y, x]
            mu_spectrum = self.mu[:, y, x]
            sigma_spectrum = self.sigma[:, y, x]
            self.ax[3].plot(self.wavelengths, gt_spectrum, color=color, label="Ground Truth")
            self.ax[3].plot(self.wavelengths, mu_spectrum, "--", color=color, label="Reconstructed (μ)")
            upper_bound = mu_spectrum + sigma_spectrum
            lower_bound = mu_spectrum - sigma_spectrum
            self.ax[3].fill_between(self.wavelengths, lower_bound, upper_bound, color=color, alpha=0.2)
        self.fig.suptitle(
            f"Image Analysis: Ground Truth vs Reconstruction (Index {self.current_index})",
            fontsize=16
        )
        if len(self.selected_points) > 0:
            self.ax[3].legend(loc='upper left')
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax[0]:
            x, y = int(event.xdata), int(event.ydata)
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            self.selected_points.append((x, y, color))
            self.ax[0].plot(x, y, "o", color=color, markersize=8)
            gt_spectrum = self.ground_truth[:, y, x]
            mu_spectrum = self.mu[:, y, x]
            sigma_spectrum = self.sigma[:, y, x]
            line_gt, = self.ax[3].plot(self.wavelengths, gt_spectrum, color=color, label="Ground Truth")
            line_mu, = self.ax[3].plot(self.wavelengths, mu_spectrum, "--", color=color, label="Reconstructed (μ)")
            upper_bound = mu_spectrum + sigma_spectrum
            lower_bound = mu_spectrum - sigma_spectrum
            uncertainty = self.ax[3].fill_between(self.wavelengths, lower_bound, upper_bound, color=color, alpha=0.2)
            self.lines.append((line_gt, line_mu, uncertainty))
            self.ax[3].legend()
            self.fig.canvas.draw()

    def clear_spectra(self, event):
        for line_info in self.lines:
            for item in line_info:
                try:
                    item.remove()
                except Exception:
                    pass
        self.lines = []
        self.selected_points = []
        self.update_plot()

    def load_next_image(self, event):
        self.current_index += 1
        if self.current_index >= len([f for f in self.image_files if f.startswith("cb_raw_")]):
            print("No more images.")
            return
        self.load_images()
        self.selected_points = []
        self.lines = []
        self.update_plot()

if __name__ == "__main__":
    main()
