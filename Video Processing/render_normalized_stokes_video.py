import argparse
import os
import re
from glob import glob
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm


WAVELENGTHS = np.linspace(450, 850, 106)
DEFAULT_WAVELENGTHS = (450, 500, 550, 650, 750)
COLORMAPS = {
    "gray": None,
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a cleaner stitched video from precomputed S0/S1/S2 cubes using S1/S0 and S2/S0."
    )
    parser.add_argument(
        "--precomputed_root",
        type=str,
        required=True,
        help="Root containing s0/, s1/, and s2/ TIFF folders from precompute_stokes_frames.py.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=35, help="Output frames per second.")
    parser.add_argument(
        "--wavelengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_WAVELENGTHS),
        help="Five wavelengths to render, e.g. 450 500 550 650 750",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="Optional frame cap for smoke tests. Use 0 for all frames.")
    parser.add_argument(
        "--s0_mask_frac",
        type=float,
        default=0.10,
        help="Mask S1/S0 and S2/S0 where S0 is below this fraction of the global S0 max for that wavelength.",
    )
    parser.add_argument(
        "--signed_limit",
        type=float,
        default=0.20,
        help="Fixed symmetric display range for S1/S0 and S2/S0 after clipping. Use 0 for global auto range.",
    )
    parser.add_argument(
        "--signed_threshold",
        type=float,
        default=0.05,
        help="Set |S1/S0| and |S2/S0| below this value to zero after smoothing.",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for S1/S0 and S2/S0. Use 0 to disable.",
    )
    parser.add_argument(
        "--signed_cmap",
        type=str,
        default="gray",
        choices=list(COLORMAPS.keys()),
        help="Colormap for S1/S0 and S2/S0.",
    )
    return parser.parse_args()


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_tiffs(folder_path: str, prefix: str) -> List[str]:
    tiff_files = [
        f for f in glob(os.path.join(folder_path, f"{prefix}*.tif"))
        if os.path.basename(f).startswith(prefix)
    ]
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found with prefix '{prefix}' in {folder_path}")
    return sorted(tiff_files, key=lambda f: natural_key(os.path.basename(f)))


def band_indices_for(wavelengths: Sequence[int]) -> List[int]:
    return [int(np.argmin(np.abs(WAVELENGTHS - wl))) for wl in wavelengths]


def load_file_lists(precomputed_root: str) -> Dict[str, List[str]]:
    file_lists = {
        "s0": list_tiffs(os.path.join(precomputed_root, "s0"), "s0_"),
        "s1": list_tiffs(os.path.join(precomputed_root, "s1"), "s1_"),
        "s2": list_tiffs(os.path.join(precomputed_root, "s2"), "s2_"),
    }
    lengths = [len(v) for v in file_lists.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Precomputed folders have different frame counts: {lengths}")
    return file_lists


def read_cube(fp: str) -> np.ndarray:
    cube = tiff.imread(fp).astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f"{fp}: expected (C,H,W), got {cube.shape}")
    return cube


def put_label(img_bgr: np.ndarray, text: str, pad: int = 10, font_scale: float = 0.68) -> None:
    h = img_bgr.shape[0]
    y = h - pad
    x = pad
    cv2.putText(img_bgr, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def scalar_to_gray_bgr(frame_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    norm = np.clip((frame_2d - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    u8 = (norm * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def symmetric_display_bgr(frame_2d: np.ndarray, limit: float, cmap_name: str) -> np.ndarray:
    norm = np.clip((frame_2d / max(limit, 1e-12) + 1.0) * 0.5, 0.0, 1.0)
    u8 = (norm * 255.0 + 0.5).astype(np.uint8)
    cmap = COLORMAPS[cmap_name]
    if cmap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(u8, cmap)


def compute_global_s0_stats(file_lists: Dict[str, List[str]], band_indices: Sequence[int], max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    total = len(file_lists["s0"]) if max_frames <= 0 else min(max_frames, len(file_lists["s0"]))
    global_min = np.full(len(band_indices), np.inf, dtype=np.float64)
    global_max = np.full(len(band_indices), -np.inf, dtype=np.float64)

    for idx in tqdm(range(total), desc="Pass 1: global S0 min/max"):
        s0 = read_cube(file_lists["s0"][idx])
        for j, bi in enumerate(band_indices):
            band = s0[bi]
            global_min[j] = min(global_min[j], float(np.min(band)))
            global_max[j] = max(global_max[j], float(np.max(band)))
    return global_min, global_max


def compute_global_signed_limits(
    file_lists: Dict[str, List[str]],
    band_indices: Sequence[int],
    max_frames: int,
    gaussian_sigma: float,
) -> np.ndarray:
    total = len(file_lists["s0"]) if max_frames <= 0 else min(max_frames, len(file_lists["s0"]))
    global_limits = np.full(len(band_indices), 1e-6, dtype=np.float64)

    for idx in tqdm(range(total), desc="Pass 1b: global signed limits"):
        s0 = read_cube(file_lists["s0"][idx])
        s1 = read_cube(file_lists["s1"][idx]) / np.maximum(s0, 1e-12)
        s2 = read_cube(file_lists["s2"][idx]) / np.maximum(s0, 1e-12)
        if gaussian_sigma > 0:
            s1 = np.stack([smooth_map(s1[b], gaussian_sigma) for b in range(s1.shape[0])], axis=0)
            s2 = np.stack([smooth_map(s2[b], gaussian_sigma) for b in range(s2.shape[0])], axis=0)

        for j, bi in enumerate(band_indices):
            global_limits[j] = max(
                global_limits[j],
                float(np.max(np.abs(s1[bi]))),
                float(np.max(np.abs(s2[bi]))),
            )
    return global_limits


def smooth_map(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def render_video(
    precomputed_root: str,
    output_path: str,
    fps: int,
    wavelengths: Sequence[int],
    max_frames: int,
    s0_mask_frac: float,
    signed_limit: float,
    signed_threshold: float,
    gaussian_sigma: float,
    signed_cmap: str,
) -> None:
    if len(wavelengths) != 5:
        raise ValueError("Please provide exactly 5 wavelengths.")

    file_lists = load_file_lists(precomputed_root)
    total = len(file_lists["s0"]) if max_frames <= 0 else min(max_frames, len(file_lists["s0"]))
    band_indices = band_indices_for(wavelengths)
    global_s0_min, global_s0_max = compute_global_s0_stats(file_lists, band_indices, max_frames)
    global_signed_limits = None
    if signed_limit <= 0:
        global_signed_limits = compute_global_signed_limits(file_lists, band_indices, max_frames, gaussian_sigma)

    first = read_cube(file_lists["s0"][0])
    tile_h, tile_w = first.shape[1], first.shape[2]
    out_h = 3 * tile_h
    out_w = 5 * tile_w

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for idx in tqdm(range(total), desc="Pass 2: render normalized Stokes video"):
            s0 = read_cube(file_lists["s0"][idx])
            s1 = read_cube(file_lists["s1"][idx])
            s2 = read_cube(file_lists["s2"][idx])

            rows = []
            s1n = s1 / np.maximum(s0, 1e-12)
            s2n = s2 / np.maximum(s0, 1e-12)

            for row_label, cube, mode in (
                ("S0", s0, "s0"),
                ("S1", s1n, "signed"),
                ("S2", s2n, "signed"),
            ):
                panels = []
                for j, (wl, bi) in enumerate(zip(wavelengths, band_indices)):
                    band = cube[bi]
                    if mode == "s0":
                        panel = scalar_to_gray_bgr(band, vmin=float(global_s0_min[j]), vmax=float(global_s0_max[j]))
                    else:
                        s0_band = s0[bi]
                        band = smooth_map(band, gaussian_sigma)
                        mask = s0_band < float(s0_mask_frac) * float(global_s0_max[j])
                        current_limit = float(global_signed_limits[j]) if global_signed_limits is not None else float(signed_limit)
                        band = np.clip(band, -current_limit, current_limit)
                        band[np.abs(band) < signed_threshold] = 0.0
                        band[mask] = 0.0
                        panel = symmetric_display_bgr(band, limit=current_limit, cmap_name=signed_cmap)
                    put_label(panel, f"{row_label}  {int(round(wl))}nm")
                    panels.append(panel)
                rows.append(np.hstack(panels))

            writer.write(np.vstack(rows))
    finally:
        writer.release()

    print(f"Saved video: {output_path}")


def main() -> None:
    args = parse_args()
    render_video(
        precomputed_root=args.precomputed_root,
        output_path=args.output_path,
        fps=args.fps,
        wavelengths=args.wavelengths,
        max_frames=args.max_frames,
        s0_mask_frac=args.s0_mask_frac,
        signed_limit=args.signed_limit,
        signed_threshold=args.signed_threshold,
        gaussian_sigma=args.gaussian_sigma,
        signed_cmap=args.signed_cmap,
    )


if __name__ == "__main__":
    main()
