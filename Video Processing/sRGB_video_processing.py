import os
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from scipy.signal import wiener

# ----------------------------
# Wavelength model (first 106)
# ----------------------------
WAVELENGTHS = np.linspace(450, 850, 106)

# RGB band ranges (your method)
R_IDX = np.where((WAVELENGTHS >= 600) & (WAVELENGTHS <= 700))[0]
G_IDX = np.where((WAVELENGTHS >= 500) & (WAVELENGTHS <= 600))[0]
B_IDX = np.where((WAVELENGTHS >= 450) & (WAVELENGTHS <= 500))[0]


def extract_number(filename, prefix):
    """Extract numeric suffix after prefix, e.g. tl_gen_12.tif -> 12"""
    m = re.search(rf"{re.escape(prefix)}(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else float("inf")


def list_hsi_tiffs(folder_path, file_prefix="tl_gen_"):
    """Return sorted TIFF list for prefix."""
    tiff_files = [
        f for f in glob(os.path.join(folder_path, f"{file_prefix}*.tif"))
        if os.path.basename(f).startswith(file_prefix)
    ]
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found with prefix '{file_prefix}' in {folder_path}")
    return sorted(tiff_files, key=lambda f: extract_number(f, file_prefix))


def crop_center(img_chw, crop_hw=(120, 120)):
    """
    Crop the center region of a (C, H, W) cube.
    """
    if img_chw.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {img_chw.shape}")

    C, H, W = img_chw.shape
    ch, cw = crop_hw
    if ch > H or cw > W:
        raise ValueError(f"Crop {crop_hw} is larger than image {(H, W)}")

    start_h = (H - ch) // 2
    start_w = (W - cw) // 2
    return img_chw[:, start_h:start_h + ch, start_w:start_w + cw]


def sharpen_channel(channel_2d, wiener_window=(3, 3)):
    """Apply Wiener deconvolution/denoise to a single 2D channel."""
    return wiener(channel_2d, wiener_window)


def enhance_rgb(rgb_img, gamma=0.6, global_min=None, global_max=None, eps=1e-12):
    """
    Apply gamma + global contrast stretch (per-channel).
    rgb_img is float in arbitrary range; we clip to [0,1] before gamma like your script.
    """
    img = np.clip(rgb_img, 0.0, 1.0) ** float(gamma)

    if global_min is None or global_max is None:
        return np.clip(img, 0.0, 1.0)

    out = img.copy()
    for i in range(3):
        cmin = float(global_min[i])
        cmax = float(global_max[i])
        if cmax > cmin + eps:
            out[..., i] = (out[..., i] - cmin) / (cmax - cmin)
        else:
            out[..., i] = 0.0

    return np.clip(out, 0.0, 1.0)


def compute_rgb_pan_from_cube106(cube106_chw):
    """
    cube106_chw: (106, H, W)
    Returns:
      rgb: (H, W, 3) float32
      pan: (H, W) float32
    """
    # RGB mapping via band-averaging
    r = np.mean(cube106_chw[R_IDX, :, :], axis=0)
    g = np.mean(cube106_chw[G_IDX, :, :], axis=0)
    b = np.mean(cube106_chw[B_IDX, :, :], axis=0)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # Panchromatic
    pan = np.mean(cube106_chw, axis=0).astype(np.float32)
    return rgb, pan


def rgb_to_bgr_u8(rgb01):
    """rgb01 in [0,1] -> uint8 BGR for OpenCV VideoWriter."""
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def pan_to_u8(pan2d, eps=1e-12):
    """
    Match your script’s PAN behavior:
      pan_norm = (pan - min) / (max - min)
    """
    pmin = float(np.min(pan2d))
    pmax = float(np.max(pan2d))
    pan_norm = (pan2d - pmin) / (pmax - pmin + eps)
    return (np.clip(pan_norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def create_srgb_pan_videos_sequential(
    tiff_files,
    out_srgb_path,
    out_pan_path=None,
    fps=35,
    expected_hw=(410, 410),
    crop_hw=(120, 120),
    gamma=0.6,
    wiener_window=(3, 3),
    interpolate=False,
    verbose=True,
):
    """
    Two-pass sequential video renderer using your implementation:

    Pass 1: compute GLOBAL min/max for RGB channels (after crop and RGB band-mean)
    Pass 2: write sRGB video:
        rgb_enhanced = enhance_rgb(rgb, gamma, global_min, global_max)
        rgb_sharp = wiener(rgb_enhanced[...,c]) per channel
    Optional PAN video:
        pan = mean(cube106) then per-frame min/max normalization (same as your script)
    """
    if not tiff_files:
        raise ValueError("Empty TIFF file list.")

    # ---- Determine output size from first frame ----
    first = tiff.imread(tiff_files[0])
    if first.ndim != 3 or first.shape[0] != 212:
        raise ValueError(f"{tiff_files[0]}: expected (212,H,W), got {first.shape}")

    H, W = first.shape[1], first.shape[2]
    if expected_hw is not None and (H, W) != tuple(expected_hw):
        raise ValueError(f"{tiff_files[0]}: unexpected spatial size {(H, W)}. Expected {expected_hw}.")

    ch, cw = crop_hw
    if ch > H or cw > W:
        raise ValueError(f"Crop {crop_hw} is larger than image {(H, W)}")

    # ---- PASS 1: compute global min/max (streaming) ----
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)

    if verbose:
        print(f"Found {len(tiff_files)} files. Pass 1/2: computing global RGB min/max...")

    for fp in tqdm(tiff_files, desc="Pass 1: global RGB min/max"):
        cube = tiff.imread(fp)  # (212,H,W)
        if cube.ndim != 3 or cube.shape[0] != 212:
            raise ValueError(f"{fp}: expected (212,H,W), got {cube.shape}")
        if expected_hw is not None and (cube.shape[1], cube.shape[2]) != tuple(expected_hw):
            raise ValueError(f"{fp}: unexpected spatial size {(cube.shape[1], cube.shape[2])}. Expected {expected_hw}.")

        cube106 = cube[:106].astype(np.float32, copy=False)
        cropped = crop_center(cube106, crop_hw=crop_hw)  # (106,ch,cw)
        rgb, _ = compute_rgb_pan_from_cube106(cropped)   # (ch,cw,3)

        # Update global min/max per channel
        for i in range(3):
            c = rgb[..., i]
            cmin = float(np.min(c))
            cmax = float(np.max(c))
            if cmin < global_min[i]:
                global_min[i] = cmin
            if cmax > global_max[i]:
                global_max[i] = cmax

    if verbose:
        print(f"Global min: {global_min}")
        print(f"Global max: {global_max}")

    # ---- PASS 2: write videos (streaming) ----
    os.makedirs(os.path.dirname(out_srgb_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    srgb_writer = cv2.VideoWriter(out_srgb_path, fourcc, fps, (cw, ch), isColor=True)
    if not srgb_writer.isOpened():
        raise RuntimeError(f"Could not open sRGB video writer for {out_srgb_path}")

    pan_writer = None
    if out_pan_path is not None:
        os.makedirs(os.path.dirname(out_pan_path), exist_ok=True)
        pan_writer = cv2.VideoWriter(out_pan_path, fourcc, fps, (cw, ch), isColor=False)
        if not pan_writer.isOpened():
            raise RuntimeError(f"Could not open PAN video writer for {out_pan_path}")

    prev_cube106_crop = None  # for interpolation (cropped 106,ch,cw)

    # Progress count
    total_out = (2 * len(tiff_files) - 1) if (interpolate and len(tiff_files) > 1) else len(tiff_files)
    pbar = tqdm(total=total_out, desc="Pass 2: writing frames")

    try:
        for fp in tiff_files:
            cube = tiff.imread(fp)  # (212,H,W)
            cube106 = cube[:106].astype(np.float32, copy=False)
            cropped = crop_center(cube106, crop_hw=crop_hw)  # (106,ch,cw)

            def write_from_crop(crop106):
                rgb, pan = compute_rgb_pan_from_cube106(crop106)

                # Your RGB pipeline: gamma + global stretch + wiener sharpen
                rgb_enh = enhance_rgb(rgb, gamma=gamma, global_min=global_min, global_max=global_max)
                rgb_sharp = np.stack(
                    [sharpen_channel(rgb_enh[..., i], wiener_window=wiener_window) for i in range(3)],
                    axis=-1
                ).astype(np.float32)

                # NOTE: wiener can slightly overshoot; clip before quantization
                srgb_writer.write(rgb_to_bgr_u8(np.clip(rgb_sharp, 0.0, 1.0)))

                if pan_writer is not None:
                    pan_u8 = pan_to_u8(pan)
                    pan_writer.write(pan_u8)

            # Interpolated frame (between prev and current)
            if interpolate and prev_cube106_crop is not None:
                interp = 0.5 * prev_cube106_crop + 0.5 * cropped
                write_from_crop(interp)
                pbar.update(1)

            # Current frame
            write_from_crop(cropped)
            pbar.update(1)

            prev_cube106_crop = cropped

    finally:
        pbar.close()
        srgb_writer.release()
        if pan_writer is not None:
            pan_writer.release()

    if verbose:
        print(f"Saved sRGB video: {out_srgb_path}")
        if out_pan_path is not None:
            print(f"Saved PAN video:  {out_pan_path}")


# ---------------------------------------------------------------------
# Example usage for your CHPC umbrella run (tl_gen_*.tif in images/)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    folder_path = r"/scratch/general/nfs1/u1528328/Umbrella_Video_Frames/Umbrella_Video_aligned_eval_pol0/validation_latest/images/"
    file_prefix = "tl_gen_"

    tiff_files = list_hsi_tiffs(folder_path, file_prefix=file_prefix)

    out_srgb = r"/uufs/chpc.utah.edu/common/home/u1528328/Documents/NASA_HSI/processed_videos/Umbrellas/Umbrella_35fps_pol0_sRGB_global_refined.mp4"
    out_pan  = r"/uufs/chpc.utah.edu/common/home/u1528328/Documents/NASA_HSI/processed_videos/Umbrellas/Umbrella_35fps_pol0_PAN_refined.mp4"

    create_srgb_pan_videos_sequential(
        tiff_files=tiff_files,
        out_srgb_path=out_srgb,
        out_pan_path=out_pan,     # set None if you don't want PAN
        fps=35,
        expected_hw=(410, 410),
        crop_hw=(370, 370),       # crop 20 px from each side
        gamma=0.6,                # matches your script
        wiener_window=(3, 3),     # matches your script
        interpolate=False,
        verbose=True,
    )
