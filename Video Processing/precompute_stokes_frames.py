import argparse
import os
import re
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import tifffile as tiff
from tqdm import tqdm


POLS = (0, 45, 90, 135)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute S0/S1/S2/DoLP/AoLP cubes from four polarization reconstruction folders."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root folder containing Umbrella_Video_aligned_eval_pol0/pol45/pol90/pol135 result folders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root where s0/s1/s2/dolp/aolp folders will be created.",
    )
    parser.add_argument("--crop", type=int, default=20, help="Pixels to crop from each border.")
    parser.add_argument("--expected_hw", type=int, nargs=2, default=[410, 410], help="Expected uncropped H W.")
    parser.add_argument("--max_frames", type=int, default=0, help="Optional frame cap for smoke tests. Use 0 for all frames.")
    return parser.parse_args()


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_tiffs(folder_path: str, file_prefix: str = "tl_gen_") -> List[str]:
    tiff_files = [
        f for f in glob(os.path.join(folder_path, f"{file_prefix}*.tif"))
        if os.path.basename(f).startswith(file_prefix)
    ]
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found with prefix '{file_prefix}' in {folder_path}")
    return sorted(tiff_files, key=lambda f: natural_key(os.path.basename(f)))


def crop_cube(cube: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return cube
    return cube[:, crop:-crop, crop:-crop]


def read_mu_cube(fp: str, expected_hw: Tuple[int, int], crop: int) -> np.ndarray:
    cube = tiff.imread(fp).astype(np.float32)
    if cube.ndim != 3 or cube.shape[0] != 212:
        raise ValueError(f"{fp}: expected (212,H,W), got {cube.shape}")
    if tuple(cube.shape[1:]) != tuple(expected_hw):
        raise ValueError(f"{fp}: expected spatial size {expected_hw}, got {tuple(cube.shape[1:])}")
    mu = cube[:106]
    return crop_cube(mu, crop)


def load_aligned_file_lists(results_root: str) -> Dict[int, List[str]]:
    file_lists: Dict[int, List[str]] = {}
    lengths = []
    for pol in POLS:
        folder = os.path.join(results_root, f"Umbrella_Video_aligned_eval_pol{pol}", "validation_latest", "images")
        files = list_tiffs(folder, file_prefix="tl_gen_")
        file_lists[pol] = files
        lengths.append(len(files))
    if len(set(lengths)) != 1:
        raise ValueError(f"Polarization folders have different frame counts: {lengths}")
    return file_lists


def compute_products(cubes: Dict[int, np.ndarray], eps: float = 1e-12):
    i0 = cubes[0]
    i45 = cubes[45]
    i90 = cubes[90]
    i135 = cubes[135]

    s0 = i0 + i90
    s1 = i0 - i90
    s2 = i45 - i135
    dolp = np.sqrt(np.maximum(s1 ** 2 + s2 ** 2, 0.0)) / np.maximum(s0, eps)
    dolp = np.clip(dolp, 0.0, 1.0).astype(np.float32)
    aolp = 0.5 * np.arctan2(s2, s1)
    aolp_deg = np.rad2deg(aolp).astype(np.float32)
    return s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32), dolp, aolp_deg


def ensure_out_dirs(output_root: str) -> Dict[str, str]:
    out_dirs = {}
    for key in ("s0", "s1", "s2", "dolp", "aolp"):
        path = os.path.join(output_root, key)
        os.makedirs(path, exist_ok=True)
        out_dirs[key] = path
    return out_dirs


def out_name(prefix: str, src_path: str) -> str:
    base = os.path.basename(src_path)
    if base.startswith("tl_gen_"):
        return prefix + base[len("tl_gen_"):]
    return f"{prefix}{base}"


def main() -> None:
    args = parse_args()
    file_lists = load_aligned_file_lists(args.results_root)
    total = len(file_lists[0]) if args.max_frames <= 0 else min(args.max_frames, len(file_lists[0]))
    out_dirs = ensure_out_dirs(args.output_root)

    print(f"Results root: {args.results_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Frames to process: {total}")

    for idx in tqdm(range(total), desc="Precomputing Stokes products"):
        cubes = {
            pol: read_mu_cube(file_lists[pol][idx], expected_hw=tuple(args.expected_hw), crop=args.crop)
            for pol in POLS
        }
        s0, s1, s2, dolp, aolp = compute_products(cubes)

        ref_path = file_lists[0][idx]
        tiff.imwrite(os.path.join(out_dirs["s0"], out_name("s0_", ref_path)), s0)
        tiff.imwrite(os.path.join(out_dirs["s1"], out_name("s1_", ref_path)), s1)
        tiff.imwrite(os.path.join(out_dirs["s2"], out_name("s2_", ref_path)), s2)
        tiff.imwrite(os.path.join(out_dirs["dolp"], out_name("dolp_", ref_path)), dolp)
        tiff.imwrite(os.path.join(out_dirs["aolp"], out_name("aolp_", ref_path)), aolp)

    print("Done.")
    for key, path in out_dirs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
