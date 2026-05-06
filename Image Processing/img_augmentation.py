import os
import re
from pathlib import Path
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# === Base directories ===
# base_dir = r"D:\banknotes_4-15\processed\training"  # for local testing
base_dir = r"/scratch/general/nfs1/u1528328/Trees_2.0/split/training"
cubert_dir = Path(base_dir) / "cubert"
thorlabs_dir = Path(base_dir) / "thorlabs"
output_dir = Path(base_dir) / "augmented"

# Output folders
aug_cubert_dir = output_dir / "cubert"
aug_thorlabs_dir = output_dir / "thorlabs"
aug_cubert_dir.mkdir(parents=True, exist_ok=True)
aug_thorlabs_dir.mkdir(parents=True, exist_ok=True)

# === Helper functions for filename matching ===
def get_prefix(filename, suffix):
    """Extract shared prefix by removing suffix."""
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return None

def get_idx(filename):
    """
    Extract ID from filename. Supports:
    1. Timestamp format: YYYYMMDD_HHMMSS (e.g., cubert_20260201_165214.tif)
    2. Simple numeric ID (e.g., image_001.tif)
    """
    # Try timestamp format first
    m = re.search(r"(\d{8}_\d{6})", filename)
    if m:
        return m.group(1)
    # Fall back to simple numeric ID
    m = re.search(r"(\d+)", filename)
    return m.group(1) if m else None

def match_file_pairs(cubert_dir: Path, thorlabs_dir: Path):
    """
    Match file pairs using both prefix-suffix and ID-based methods.
    Returns: (cubert_matched, thorlabs_matched, common_ids, matching_method)
    """
    # Get all .tif files
    cubert_files_all = sorted([f.name for f in cubert_dir.glob("*.tif")])
    thorlabs_files_all = sorted([f.name for f in thorlabs_dir.glob("*.tif")])
    
    # METHOD 1: Try prefix-suffix matching
    cubert_suffix = "_cubert.tif"
    thorlabs_suffix = "_thorlabs.tif"
    
    cubert_prefixes = {get_prefix(f, cubert_suffix): f for f in cubert_files_all 
                       if get_prefix(f, cubert_suffix) is not None}
    thorlabs_prefixes = {get_prefix(f, thorlabs_suffix): f for f in thorlabs_files_all 
                         if get_prefix(f, thorlabs_suffix) is not None}
    
    common_prefixes_method1 = set(cubert_prefixes.keys()) & set(thorlabs_prefixes.keys())

    # METHOD 2: Try ID-based matching (timestamp or numeric)
    cubert_ids = {get_idx(f): f for f in cubert_files_all if get_idx(f) is not None}
    thorlabs_ids = {get_idx(f): f for f in thorlabs_files_all if get_idx(f) is not None}
    
    common_ids_method2 = set(cubert_ids.keys()) & set(thorlabs_ids.keys())

    # Choose the method that found more matches
    if len(common_prefixes_method1) >= len(common_ids_method2):
        # Use prefix-suffix method
        common_ids = sorted(list(common_prefixes_method1))
        cubert_matched = cubert_prefixes
        thorlabs_matched = thorlabs_prefixes
        matching_method = "prefix-suffix"
    else:
        # Use ID-based method
        common_ids = sorted(list(common_ids_method2))
        cubert_matched = cubert_ids
        thorlabs_matched = thorlabs_ids
        matching_method = "ID-based (timestamp/numeric)"
    
    return cubert_matched, thorlabs_matched, common_ids, matching_method

# === 16 deterministic transforms: 4 rotations × H-flip × V-flip ===
def generate_transforms_and_names():
    transforms, names = [], []
    idx = 0
    for angle in (0, 90, 180, 270):
        for flip_h in (False, True):
            for flip_v in (False, True):
                transforms.append((angle, flip_h, flip_v))
                names.append(f"aug{idx:02d}_r{angle}_h{int(flip_h)}_v{int(flip_v)}")
                idx += 1
    return transforms, names

TRANSFORMS, AUG_NAMES = generate_transforms_and_names()

def get_spatial_axes(arr_shape):
    """
    Determine (h_axis, w_axis, c_axis or None) for 2D or 3D arrays.
    Assumes multi-channel TIFFs are 3D with the smallest dimension as channels.
    Handles (C,H,W) and (H,W,C) robustly.
    """
    if len(arr_shape) == 2:
        return 0, 1, None  # (H, W)
    if len(arr_shape) != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr_shape}")

    c_axis = int(np.argmin(arr_shape))  # channel dimension is typically smallest
    spatial_axes = [ax for ax in range(3) if ax != c_axis]
    # Assign h_axis/w_axis deterministically
    h_axis, w_axis = spatial_axes[0], spatial_axes[1]
    return h_axis, w_axis, c_axis

def apply_transform(arr, angle, flip_h, flip_v):
    """
    Apply rotation and flips to the full multi-channel array using NumPy.
    - Rotation with np.rot90 across spatial axes (no per-frame loop).
    - Flips along spatial axes.
    Ensures the result is contiguous (tifffile-friendly).
    """
    h_axis, w_axis, _ = get_spatial_axes(arr.shape)
    k = (angle // 90) % 4
    out = np.rot90(arr, k=k, axes=(h_axis, w_axis))
    if flip_h:
        out = np.flip(out, axis=w_axis)
    if flip_v:
        out = np.flip(out, axis=h_axis)
    return np.ascontiguousarray(out)

def augment_paired_files(cubert_matched, thorlabs_matched, common_ids, 
                         cubert_src_dir: Path, thorlabs_src_dir: Path,
                         cubert_dst_dir: Path, thorlabs_dst_dir: Path,
                         overwrite: bool = False):
    """
    Augment paired files to ensure corresponding augmentations.
    """
    for pair_id in tqdm(common_ids, desc="Augmenting paired images", total=len(common_ids)):
        cubert_filename = cubert_matched[pair_id]
        thorlabs_filename = thorlabs_matched[pair_id]
        
        cubert_path = cubert_src_dir / cubert_filename
        thorlabs_path = thorlabs_src_dir / thorlabs_filename
        
        # Read entire stacks
        cubert_arr = tiff.imread(cubert_path)
        thorlabs_arr = tiff.imread(thorlabs_path)
        
        cubert_stem = Path(cubert_filename).stem
        thorlabs_stem = Path(thorlabs_filename).stem
        
        # Apply same transforms to both
        for (angle, fh, fv), suffix in zip(TRANSFORMS, AUG_NAMES):
            # Cubert augmentation
            cubert_out_name = f"{cubert_stem}_{suffix}.tif"
            cubert_out_path = cubert_dst_dir / cubert_out_name
            if not cubert_out_path.exists() or overwrite:
                cubert_aug = apply_transform(cubert_arr, angle, fh, fv)
                tiff.imwrite(cubert_out_path, cubert_aug)
            
            # Thorlabs augmentation
            thorlabs_out_name = f"{thorlabs_stem}_{suffix}.tif"
            thorlabs_out_path = thorlabs_dst_dir / thorlabs_out_name
            if not thorlabs_out_path.exists() or overwrite:
                thorlabs_aug = apply_transform(thorlabs_arr, angle, fh, fv)
                tiff.imwrite(thorlabs_out_path, thorlabs_aug)

# === Match files and run paired augmentation ===
print("[INFO] Matching file pairs...")
cubert_matched, thorlabs_matched, common_ids, matching_method = match_file_pairs(cubert_dir, thorlabs_dir)

print(f"[INFO] Matching method: {matching_method}")
print(f"[INFO] Found {len(common_ids)} paired images.")

if len(common_ids) > 0:
    print(f"[INFO] Example pairs:")
    for i, pair_id in enumerate(common_ids[:3]):
        print(f"  {thorlabs_matched[pair_id]} <-> {cubert_matched[pair_id]}")
    if len(common_ids) > 3:
        print(f"  ... and {len(common_ids) - 3} more")
else:
    print("[ERROR] No matching paired filenames found. Check directory paths and naming conventions.")
    exit(1)

print("[INFO] Starting paired augmentation...")
augment_paired_files(cubert_matched, thorlabs_matched, common_ids,
                    cubert_dir, thorlabs_dir,
                    aug_cubert_dir, aug_thorlabs_dir)

print(f"[DONE] 16× augmented images written under: {output_dir}")
print(f"  Cubert: {len(list(aug_cubert_dir.glob('*.tif')))} files")
print(f"  Thorlabs: {len(list(aug_thorlabs_dir.glob('*.tif')))} files")