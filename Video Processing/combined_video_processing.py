import os
import re
from glob import glob

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from scipy.signal import wiener

# ----------------------------
# Wavelength model (106 bands)
# ----------------------------
WAVELENGTHS = np.linspace(450, 850, 106)  # 106 bands, 450-850 nm

# RGB band ranges (your method)
R_IDX = np.where((WAVELENGTHS >= 600) & (WAVELENGTHS <= 700))[0]
G_IDX = np.where((WAVELENGTHS >= 500) & (WAVELENGTHS <= 600))[0]
B_IDX = np.where((WAVELENGTHS >= 450) & (WAVELENGTHS <= 500))[0]


def nearest_band_indices(desired_wavelengths, wavelengths=WAVELENGTHS):
    return [int(np.argmin(np.abs(wavelengths - wl))) for wl in desired_wavelengths]


def extract_number(filename, prefix):
    m = re.search(rf"{re.escape(prefix)}(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else float("inf")


def list_hsi_tiffs(folder_path, file_prefix="tl_gen_"):
    tiff_files = [
        f for f in glob(os.path.join(folder_path, f"{file_prefix}*.tif"))
        if os.path.basename(f).startswith(file_prefix)
    ]
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found with prefix '{file_prefix}' in {folder_path}")
    return sorted(tiff_files, key=lambda f: extract_number(f, file_prefix))


def read_cube212(fp, expected_hw=(410, 410)):
    cube = tiff.imread(fp)  # (212,H,W)
    if cube.ndim != 3 or cube.shape[0] != 212:
        raise ValueError(f"{fp}: expected (212,H,W), got {cube.shape}")
    H, W = cube.shape[1], cube.shape[2]
    if expected_hw is not None and (H, W) != tuple(expected_hw):
        raise ValueError(f"{fp}: unexpected spatial size {(H, W)}. Expected {expected_hw}.")
    return cube


def enhance_rgb(rgb_img, gamma=0.6, global_min=None, global_max=None, eps=1e-12):
    # gamma after clipping to [0,1] just like your script
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


def compute_rgb_from_cube106(cube106_chw):
    # band-averaging RGB mapping
    r = np.mean(cube106_chw[R_IDX, :, :], axis=0)
    g = np.mean(cube106_chw[G_IDX, :, :], axis=0)
    b = np.mean(cube106_chw[B_IDX, :, :], axis=0)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)  # (H,W,3)
    return rgb


def rgb01_to_bgr_u8(rgb01):
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def band_to_u8(frame_2d, eps=1e-12):
    # per-frame min/max normalize (matches your stitched script behavior)
    fmin = float(np.min(frame_2d))
    fmax = float(np.max(frame_2d))
    norm = (frame_2d - fmin) / (fmax - fmin + eps)
    return (np.clip(norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def put_label_bgr(img_bgr, text, x=8, y=24, font_scale=0.7):
    # shadow
    cv2.putText(img_bgr, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    # main
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def create_5wl_plus_srgb_video(
    tiff_files,
    output_path,
    desired_wavelengths=(450, 549, 648, 724, 816),
    fps=35,
    expected_hw=(410, 410),
    gamma=0.6,
    wiener_window=(3, 3),
    srgb_wiener=True,
    verbose=True,
):
    if not tiff_files:
        raise ValueError("Empty TIFF file list.")

    band_idx = nearest_band_indices(desired_wavelengths)

    # ---------- PASS 1: global min/max for sRGB (streaming) ----------
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)

    if verbose:
        print(f"Found {len(tiff_files)} frames.")
        print("Pass 1/2: computing global sRGB min/max...")

    for fp in tqdm(tiff_files, desc="Pass 1: global sRGB min/max"):
        cube = read_cube212(fp, expected_hw=expected_hw)
        cube106 = cube[:106].astype(np.float32, copy=False)  # (106,H,W)

        rgb = compute_rgb_from_cube106(cube106)  # (H,W,3)
        for c in range(3):
            cmin = float(np.min(rgb[..., c]))
            cmax = float(np.max(rgb[..., c]))
            if cmin < global_min[c]:
                global_min[c] = cmin
            if cmax > global_max[c]:
                global_max[c] = cmax

    if verbose:
        print(f"Global min (RGB): {global_min}")
        print(f"Global max (RGB): {global_max}")

    # ---------- PASS 2: write stitched 6-panel video ----------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # determine H, W from first frame
    first = read_cube212(tiff_files[0], expected_hw=expected_hw)
    H, W = first.shape[1], first.shape[2]
    out_w = 6 * W
    out_h = H

    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    if verbose:
        print("Pass 2/2: rendering frames...")

    try:
        for fp in tqdm(tiff_files, desc="Pass 2: write video"):
            cube = read_cube212(fp, expected_hw=expected_hw)
            cube106 = cube[:106].astype(np.float32, copy=False)  # (106,H,W)

            # ---- 5 wavelength panels (grayscale -> BGR) ----
            panels_bgr = []
            for wl, bi in zip(desired_wavelengths, band_idx):
                band = cube106[bi]  # (H,W)
                band_u8 = band_to_u8(band)
                band_bgr = cv2.cvtColor(band_u8, cv2.COLOR_GRAY2BGR)
                put_label_bgr(band_bgr, f"{int(round(wl))}nm")
                panels_bgr.append(band_bgr)

            # ---- sRGB panel (color) ----
            rgb = compute_rgb_from_cube106(cube106)  # float32 arbitrary
            rgb_enh = enhance_rgb(rgb, gamma=gamma, global_min=global_min, global_max=global_max)

            if srgb_wiener:
                rgb_enh = np.stack(
                    [wiener(rgb_enh[..., i], wiener_window) for i in range(3)],
                    axis=-1
                ).astype(np.float32)
                rgb_enh = np.clip(rgb_enh, 0.0, 1.0)

            srgb_bgr = rgb01_to_bgr_u8(rgb_enh)
            put_label_bgr(srgb_bgr, "sRGB")

            panels_bgr.append(srgb_bgr)

            # ---- stitch 6 panels horizontally ----
            stitched = np.hstack(panels_bgr)  # (H, 6W, 3)
            writer.write(stitched)

    finally:
        writer.release()

    print(f"Saved video: {output_path}")
    return output_path


if __name__ == "__main__":
    folder_path = r"/scratch/general/nfs1/u1528328/Umbrella_Video_Frames_results_135/Umbrella_Video_aligned_pol0/validation_latest/images/"
    file_prefix = "tl_gen_"

    tiff_files = list_hsi_tiffs(folder_path, file_prefix=file_prefix)

    out_path = r"/uufs/chpc.utah.edu/common/home/u1528328/Documents/NASA_HSI/processed_videos/Umbrellas/Umbrella_35fps_pol135__5wl_plus_sRGB.mp4"

    create_5wl_plus_srgb_video(
        tiff_files=tiff_files,
        output_path=out_path,
        desired_wavelengths=(450, 549, 648, 724, 816),
        fps=35,
        expected_hw=(410, 410),
        gamma=0.6,
        wiener_window=(3, 3),
        srgb_wiener=True,
        verbose=True,
    )
