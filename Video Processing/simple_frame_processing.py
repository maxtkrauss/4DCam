"""
Simple Thorlabs polarization video processing

- Input: multi-page TIFFs shaped (T, H, W)
- Output: per-frame TIFFs shaped (5, H, W)
    [pol0, pol135, pol90, pol45, raw]

No cropping
No warping
No translation
"""

import os
from pathlib import Path
import numpy as np
import tifffile as tiff
import cv2

# ----------------------------
# USER SETTINGS
# ----------------------------
TIFF_FILES = [
    r"D:\umbrella_video\umbrella_unprocessed_0.tif",
    r"D:\umbrella_video\umbrella_unprocessed_1.tif",
    r"D:\umbrella_video\umbrella_unprocessed_2.tif",
    r"D:\umbrella_video\umbrella_unprocessed_3.tif",
    r"D:\umbrella_video\umbrella_unprocessed_4.tif",
    r"D:\umbrella_video\umbrella_unprocessed_5.tif",
    r"D:\umbrella_video\umbrella_unprocessed_6.tif",
    r"D:\umbrella_video\umbrella_unprocessed_7.tif",
    r"D:\umbrella_video\umbrella_unprocessed_8.tif",
]

OUT_DIR = Path(r"D:\umbrella_video\umbrella_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_INDEX = 0  # global frame counter start

# ----------------------------
# POLARIZATION DEMOSAIC
# ----------------------------
def demosaic_polarization(raw):
    """
    raw: (H, W) polarization mosaic
    returns: (4, H/2, W/2) [0, 135, 90, 45]
    """
    pol = np.empty((4, raw.shape[0] // 2, raw.shape[1] // 2), dtype=np.float32)

    pol[0] = raw[0::2, 0::2]  # 0°
    pol[1] = raw[1::2, 0::2]  # 135°
    pol[2] = raw[1::2, 1::2]  # 90°
    pol[3] = raw[0::2, 1::2]  # 45°

    return pol

def upsample_pol_channels(pol_half, target_hw):
    """
    pol_half: (4, H/2, W/2)
    returns:  (4, H, W)
    """
    H, W = target_hw
    pol_full = np.empty((4, H, W), dtype=np.float32)

    for i in range(4):
        pol_full[i] = cv2.resize(
            pol_half[i],
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

    return pol_full

# ----------------------------
# MAIN PROCESSING
# ----------------------------
def process_video_tiffs(tiff_files, out_dir, start_index=0):
    frame_idx = start_index

    for tif_path in tiff_files:
        tif_path = Path(tif_path)
        frames = tiff.imread(str(tif_path))  # (T, H, W)

        if frames.ndim != 3:
            raise ValueError(f"{tif_path.name}: expected (T,H,W), got {frames.shape}")

        T, H, W = frames.shape
        print(f"Loaded {tif_path.name} | frames={T} | size={H}x{W}")

        for i in range(T):
            raw = frames[i].astype(np.float32)

            pol_half = demosaic_polarization(raw)          # (4, H/2, W/2)
            pol_full = upsample_pol_channels(pol_half, (H, W))  # (4, H, W)

            stack5 = np.zeros((5, H, W), dtype=np.float32)
            stack5[:4] = pol_full
            stack5[4] = raw

            out_path = out_dir / f"frame_{frame_idx:06d}.tif"
            tiff.imwrite(out_path, stack5)

            if frame_idx % 50 == 0:
                print(f"Saved {out_path.name}")

            frame_idx += 1

    print(f"Done. Total frames written: {frame_idx - start_index}")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    process_video_tiffs(TIFF_FILES, OUT_DIR, START_INDEX)
