#!/usr/bin/env python3
"""
Interactive Hyperspectral Reconstruction Viewer
------------------------------------------------
Displays:
  • Ground Truth, Reconstruction, Error, and Sigma (2×2 grid)
  • Spectral plot (μ ± σ) for clicked pixels
  • Metrics summary (MAE, MSE, SSIM, PSNR)

Use:
  - Slider for wavelength selection
  - Next/Prev/Clear buttons to navigate and clear selections
  - Click pixels on GT/Recon to plot spectra
"""

import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# ---------------------------- Config ----------------------------
IMAGE_DIR = r"/scratch/general/nfs1/u1528328/Umbrella_Video_Results/Umbrella_Video_aligned_eval_pol0/validation_latest/images"
CROP_SIZE = 410
NUM_BANDS = 106
WAVELENGTHS = np.linspace(450, 850, NUM_BANDS)

# ---------------------------- Utils -----------------------------
def center_crop(img, target_h, target_w):
    """Crop image symmetrically to target size."""
    if img.ndim == 3:
        _, h, w = img.shape
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[:, top:top+target_h, left:left+target_w]
    return img

def compute_metrics(gt, recon):
    """Compute MAE, MSE, SSIM, PSNR between GT and reconstruction."""
    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    rc_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    mae = np.mean(np.abs(gt_n - rc_n))
    mse = np.mean((gt_n - rc_n) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    try:
        s = ssim(gt_n, rc_n, data_range=1.0)
    except Exception:
        s = np.nan
    return mae, mse, s, psnr

# ---------------------------- Viewer ----------------------------
class HSIViewer:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.gt_files = sorted([f for f in os.listdir(image_dir) if f.startswith("cb_raw_")])
        self.pred_files = sorted([f for f in os.listdir(image_dir) if f.startswith("tl_gen_")])
        self.index, self.band = 0, 0
        self.selected_points, self.color_index = [], 0
        self.colors = plt.cm.tab10.colors
        self.load_pair()

        # Set modern style
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#f8f9fa')

        # Layout: 2x2 images + spectrum plot + slider + controls
        gs = self.fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.08, 0.08],
                                    width_ratios=[1, 1, 0.05, 1.2],
                                    hspace=0.3, wspace=0.3)
        
        self.ax_imgs = [
            self.fig.add_subplot(gs[0, 0]),  # GT
            self.fig.add_subplot(gs[0, 1]),  # Recon
            self.fig.add_subplot(gs[1, 0]),  # Error
            self.fig.add_subplot(gs[1, 1]),  # Sigma
        ]
        
        self.ax_spec = self.fig.add_subplot(gs[:2, 3])  # spectrum on right
        self.ax_slider = self.fig.add_subplot(gs[2, :3])  # wavelength slider
        
        # Metrics display in dedicated axis
        self.ax_metrics = self.fig.add_subplot(gs[2, 3])
        self.ax_metrics.axis('off')

        # Button layout (row 3)
        button_y = 0.04
        btn_width, btn_height = 0.08, 0.045
        
        self.btn_prev = Button(plt.axes([0.12, button_y, btn_width, btn_height]), 
                               'Prev', hovercolor='#e0e0e0')
        self.btn_next = Button(plt.axes([0.21, button_y, btn_width, btn_height]), 
                               'Next', hovercolor='#e0e0e0')
        self.btn_clear = Button(plt.axes([0.30, button_y, btn_width, btn_height]), 
                                'Clear', hovercolor='#ffcccc')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_clear.on_clicked(self.clear_selections)

        # Slider styling
        self.slider = Slider(self.ax_slider, 'Wavelength (nm)',
                             0, NUM_BANDS-1, valinit=0, valstep=1,
                             color='#2196F3', track_color='#e0e0e0')
        self.slider.on_changed(self.update_band)

        # Event handling
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

        self.update_images()
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.show()

    # ---------------- Load pair ----------------
    def load_pair(self):
        gt_name = self.gt_files[self.index]
        pred_name = self.pred_files[self.index]
        print(f"\n🖼️  Loading pair {self.index + 1}/{len(self.gt_files)}:")
        print(f"   ├─ Ground Truth: {gt_name}")
        print(f"   └─ Reconstruction: {pred_name}")

        gt = tiff.imread(self.image_dir / gt_name)
        pred_full = tiff.imread(self.image_dir / pred_name)
        mu, sigma = pred_full[:NUM_BANDS], pred_full[NUM_BANDS:]
        self.gt = gt
        self.mu = mu
        self.sigma = sigma


    # ---------------- Update images ----------------
    def update_images(self):
        wl = WAVELENGTHS[self.band]
        gt, mu = self.gt[self.band], self.mu[self.band]
        err, sig = np.abs(gt - mu), self.sigma[self.band]

        mae, mse, s, psnr = compute_metrics(gt, mu)

        panels = [
            (self.ax_imgs[0], gt, f"Ground Truth\n({wl:.0f} nm)", "viridis"),
            (self.ax_imgs[1], mu, "Reconstruction", "viridis"),
            (self.ax_imgs[2], err, "Absolute Error", "hot"),
            (self.ax_imgs[3], sig, "Uncertainty (σ)", "viridis"),
        ]

        for ax, data, title, cmap in panels:
            ax.cla()
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
            ax.axis("off")
            
            # Overlay selected points
            for (x, y, c) in self.selected_points:
                ax.plot(x, y, "o", color=c, markersize=6, markeredgecolor='white', 
                       markeredgewidth=1.5)

        # Metrics display
        self.ax_metrics.cla()
        self.ax_metrics.axis('off')
        metrics_text = (f"MAE: {mae:.4f}\n"
                       f"MSE: {mse:.4f}\n"
                       f"SSIM: {s:.3f}\n"
                       f"PSNR: {psnr:.2f} dB")
        self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                            fontsize=9, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='#ffffff', alpha=0.8, pad=0.5))

        self.update_spectrum_plot()
        self.fig.suptitle(f"Hyperspectral Reconstruction — Pair {self.index+1}/{len(self.gt_files)}",
                         fontsize=13, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()

    # ---------------- Update wavelength ----------------
    def update_band(self, val):
        self.band = int(val)
        self.update_images()

    # ---------------- Spectrum plot ----------------
    def update_spectrum_plot(self):
        self.ax_spec.cla()
        self.ax_spec.set_title("Spectral Profile", fontsize=10, fontweight='bold', pad=8)
        self.ax_spec.set_xlabel("Wavelength (nm)", fontsize=9)
        self.ax_spec.set_ylabel("Intensity", fontsize=9)
        self.ax_spec.set_facecolor('#fafafa')

        if not self.selected_points:
            self.ax_spec.text(0.5, 0.5, "Click pixels to plot\nspectra", 
                             ha="center", va="center", color="#999999", fontsize=9,
                             transform=self.ax_spec.transAxes)
        else:
            for (x, y, c) in self.selected_points:
                gt_spec = self.gt[:, y, x]
                mu_spec = self.mu[:, y, x]
                sigma_spec = self.sigma[:, y, x]
                self.ax_spec.plot(WAVELENGTHS, gt_spec, color=c, lw=2.2, 
                                 label=f"GT ({x},{y})", zorder=3)
                self.ax_spec.plot(WAVELENGTHS, mu_spec, "--", color=c, lw=1.8, 
                                 label="Recon", zorder=2, alpha=0.85)
                self.ax_spec.fill_between(WAVELENGTHS,
                                          mu_spec - sigma_spec, mu_spec + sigma_spec,
                                          color=c, alpha=0.15, zorder=1)
            self.ax_spec.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
        
        self.ax_spec.grid(alpha=0.25, linestyle='--')
        self.ax_spec.set_xlim(WAVELENGTHS[0], WAVELENGTHS[-1])

    # ---------------- Interaction ----------------
    def onclick(self, event):
        if event.inaxes in self.ax_imgs[:2] and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < CROP_SIZE and 0 <= y < CROP_SIZE:
                color = self.colors[self.color_index % len(self.colors)]
                self.color_index += 1
                self.selected_points.append((x, y, color))
                self.update_images()

    # ---------------- Clear selections ----------------
    def clear_selections(self, event):
        self.selected_points = []
        self.color_index = 0
        self.update_images()

    # ---------------- Navigation ----------------
    def next_image(self, event):
        self.index = (self.index + 1) % len(self.gt_files)
        self.selected_points, self.color_index = [], 0
        self.load_pair()
        self.update_images()

    def prev_image(self, event):
        self.index = (self.index - 1) % len(self.gt_files)
        self.selected_points, self.color_index = [], 0
        self.load_pair()
        self.update_images()


# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    HSIViewer(IMAGE_DIR)