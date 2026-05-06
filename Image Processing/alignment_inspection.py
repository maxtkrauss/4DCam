import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re

# ---------------------------------------------------------------
# Directories
# ---------------------------------------------------------------
tl_dir       = r"D:\banknotes_4-15\original\thorlabs"
orig_cb_dir  = r"D:\banknotes_4-15\original\cubert"
reg_cb_dir   = r"D:\banknotes_4-15\registered_cubert"

print("Loading from:")
print("  Thorlabs:", tl_dir)
print("  Cubert original:", orig_cb_dir)
print("  Cubert registered:", reg_cb_dir, "\n")

# Get matching IDs across directories
def extract_idx(fname):
    m = re.search(r"(\d+)", fname)
    return m.group(1) if m else None

tl_files  = {extract_idx(f): f for f in os.listdir(tl_dir)  if f.endswith(".tif")}
cb_files  = {extract_idx(f): f for f in os.listdir(orig_cb_dir) if f.endswith(".tif")}
reg_files = {extract_idx(f): f for f in os.listdir(reg_cb_dir) if f.endswith(".tif")}

# Intersection of IDs
common = sorted(list(set(tl_files.keys()) & set(cb_files.keys()) & set(reg_files.keys())))
print(f"Found {len(common)} matching Cubert/Thorlabs IDs.\n")

# Use first 3
N = min(3, len(common))
selected = common[:N]
print("Inspecting image IDs:", selected, "\n")

# Bands to show
bands_to_show = [5, 30, 60, 90]

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
plt.figure(figsize=(16, 12))

plot_index = 1

for idx in selected:
    print(f"Loading ID {idx}...")

    # Load Thorlabs image and crop 2048×2048
    tl_img = tifffile.imread(os.path.join(tl_dir, tl_files[idx]))[0]
    H, W = tl_img.shape
    crop = min(H, W)
    x0 = (W - crop) // 2
    tl_crop = tl_img[:, x0:x0 + crop]
    tl_small = cv2.resize(tl_crop, (410, 410), cv2.INTER_AREA)

    # Load original + registered Cubert
    cb_orig = tifffile.imread(os.path.join(orig_cb_dir,  cb_files[idx]))   # (106,410,410)
    cb_reg  = tifffile.imread(os.path.join(reg_cb_dir, reg_files[idx]))   # (106,410,410)

    for b in bands_to_show:

        plt.subplot(N, len(bands_to_show)*3, plot_index)
        plt.imshow(tl_small, cmap='gray')
        plt.title(f"ID {idx}\nThorlabs (↓)")
        plt.axis('off')
        plot_index += 1

        plt.subplot(N, len(bands_to_show)*3, plot_index)
        plt.imshow(cb_orig[b], cmap='gray')
        plt.title(f"Orig Cubert\nBand {b}")
        plt.axis('off')
        plot_index += 1

        plt.subplot(N, len(bands_to_show)*3, plot_index)
        plt.imshow(cb_reg[b], cmap='gray')
        plt.title(f"Registered Cubert\nBand {b}")
        plt.axis('off')
        plot_index += 1

plt.tight_layout()
plt.show()
