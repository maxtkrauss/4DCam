import os
import re
from glob import glob

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# ----------------------------
# Wavelength model (106 bands)
# ----------------------------
WAVELENGTHS = np.linspace(450, 850, 106)  # 106 bands, 450-850 nm


def nearest_band_indices(desired_wavelengths, wavelengths=WAVELENGTHS):
    """Return closest band indices for a list of desired wavelengths."""
    return [int(np.argmin(np.abs(wavelengths - wl))) for wl in desired_wavelengths]


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

    tiff_files = sorted(tiff_files, key=lambda f: extract_number(f, file_prefix))
    return tiff_files


def read_selected_cube(fp, use_uncertainty, band_indices=None, expected_hw=(410, 410), crop_pixels=0):
    """
    Read one TIFF (212,H,W) -> return selected half (106,H,W) or selected bands (K,H,W).
    """
    cube = tiff.imread(fp)  # expected (212, H, W)

    if cube.ndim != 3 or cube.shape[0] != 212:
        raise ValueError(f"{fp}: unexpected shape {cube.shape}. Expected (212, H, W).")

    H, W = cube.shape[1], cube.shape[2]
    if expected_hw is not None and (H, W) != tuple(expected_hw):
        raise ValueError(f"{fp}: unexpected spatial size {(H, W)}. Expected {expected_hw}.")

    half = cube[106:212] if use_uncertainty else cube[0:106]  # (106,H,W)

    if band_indices is not None:
        half = half[band_indices]  # (K,H,W)

    if crop_pixels > 0:
        half = half[:, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

    return half


def enhance_frame_u8(frame_2d, clahe, sharpen_kernel):
    """
    Normalize -> CLAHE -> light sharpen -> light blur (returns uint8).
    clahe and sharpen_kernel are pre-created for speed.
    """
    normalized = cv2.normalize(frame_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = clahe.apply(normalized)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    denoised = cv2.GaussianBlur(sharpened, (3, 3), 0.5)
    return denoised


def add_labels(stitched_u8, labels):
    """
    Draw labels over each band panel.
    Labels can be floats/ints (wavelengths) OR strings.
    """
    frame = stitched_u8.copy()
    h, w = frame.shape[:2]
    n = len(labels)
    band_w = w // n if n > 0 else w

    for i, label in enumerate(labels):
        x = i * band_w + 5
        y = 20

        if isinstance(label, (int, float, np.integer, np.floating)):
            text = f"{int(round(label))}nm"
        else:
            text = str(label)

        # shadow
        cv2.putText(frame, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,), 2, cv2.LINE_AA)
        # main text
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)

    return frame


def cube_to_stitched_u8(cube, enhance, clahe, sharpen_kernel):
    """
    cube: (K,H,W) -> stitched uint8 (H, K*W)
    """
    K, H, W = cube.shape
    bands_u8 = []

    if enhance:
        for b in range(K):
            bands_u8.append(enhance_frame_u8(cube[b], clahe, sharpen_kernel))
    else:
        for b in range(K):
            bands_u8.append(cv2.normalize(cube[b], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    return np.hstack(bands_u8)


def create_stitched_video_sequential(
    tiff_files,
    output_path,
    fps=35,
    enhance=True,
    labels=None,
    use_uncertainty=False,
    band_indices=None,
    expected_hw=(410, 410),
    crop_pixels=0,
    interpolate=False,   # on-the-fly interpolation (prev + curr)
):
    """
    Sequential renderer:
      - reads ONE TIFF at a time
      - slices bands
      - enhances + stitches
      - writes to video
      - optional interpolation uses only prev+curr (no big RAM hit)

    interpolate=True will output:
      frame0,
      interp(frame0, frame1),
      frame1,
      interp(frame1, frame2),
      ...
      last frame
    (effective FPS visually increases; keep fps same unless you want to change timing)
    """
    if not tiff_files:
        raise ValueError("Empty TIFF file list.")

    # Pre-create enhancement objects once (big speed win)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sharpen_kernel = (np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]], dtype=np.float32) / 5.0)

    # Read first frame to determine output size
    first_cube = read_selected_cube(
        tiff_files[0],
        use_uncertainty=use_uncertainty,
        band_indices=band_indices,
        expected_hw=expected_hw,
        crop_pixels=crop_pixels,
    )
    K, H, W = first_cube.shape
    stitched_W = K * W

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (stitched_W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    prev_cube = None

    # tqdm total: if interpolating, approximate output count
    total_out = (2 * len(tiff_files) - 1) if (interpolate and len(tiff_files) > 1) else len(tiff_files)
    pbar = tqdm(total=total_out, desc=f"Rendering ({'sigma' if use_uncertainty else 'spectral'})")

    try:
        for i, fp in enumerate(tiff_files):
            cube = read_selected_cube(
                fp,
                use_uncertainty=use_uncertainty,
                band_indices=band_indices,
                expected_hw=expected_hw,
                crop_pixels=crop_pixels,
            )

            # If interpolation, write prev (if first), then interp(prev,curr), then curr later.
            if interpolate and prev_cube is not None:
                # Interpolated cube (float math); same shape (K,H,W)
                interp_cube = 0.5 * prev_cube + 0.5 * cube
                stitched = cube_to_stitched_u8(interp_cube, enhance, clahe, sharpen_kernel)
                if labels is not None:
                    stitched = add_labels(stitched, labels)
                writer.write(stitched)
                pbar.update(1)

            # Write current cube
            stitched = cube_to_stitched_u8(cube, enhance, clahe, sharpen_kernel)
            if labels is not None:
                stitched = add_labels(stitched, labels)
            writer.write(stitched)
            pbar.update(1)

            prev_cube = cube

    finally:
        pbar.close()
        writer.release()

    print(f"Saved video: {output_path}")
    return output_path


if __name__ == "__main__":
    folder_path = r"/scratch/general/nfs1/u1528328/Umbrella_Video_Frames/Umbrella_Video_aligned_eval_pol0/validation_latest/images"
    
    file_prefix = "tl_gen_"

    # List files once (cheap)
    tiff_files = list_hsi_tiffs(folder_path, file_prefix=file_prefix)

    # Chosen wavelengths stitched video (spectral)
    desired_wavelengths = [450, 550, 650, 750, 850]
    band_idx = nearest_band_indices(desired_wavelengths)
    desired_labels = [f"{int(w)}nm" for w in desired_wavelengths]

    out_sel_spec = r"/uufs/chpc.utah.edu/common/home/u1528328/Documents/NASA_HSI/processed_videos/Umbrellas_full/Umbrella_35fps_pol0_refined.mp4"
    create_stitched_video_sequential(
        tiff_files=tiff_files,
        output_path=out_sel_spec,
        fps=35,
        enhance=False,
        labels=desired_labels,
        use_uncertainty=False,
        band_indices=band_idx,
        expected_hw=(410, 410),
        crop_pixels=20,
        interpolate=False,        
    )
