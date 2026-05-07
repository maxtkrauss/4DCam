import os
import random
import shutil
import time
import re

def split_paired_dataset(cubert_dir, thorlabs_dir, output_base, train_ratio=0.9, seed=42, progress_interval=50):
    start_time = time.time()
    random.seed(seed)

    # Get sorted list of files that exist in both directories
    cubert_files = sorted([f for f in os.listdir(cubert_dir) if f.lower().endswith('.tif')])
    thorlabs_files = sorted([f for f in os.listdir(thorlabs_dir) if f.lower().endswith('.tif')])

    # Only keep files present in BOTH
    common_files = sorted(list(set(cubert_files) & set(thorlabs_files)))

    print(f"[INFO] Found {len(common_files)} paired images.")
    if not common_files:
        print("[ERROR] No matching filenames found. Check directory paths.")
        return

    # Shuffle for random split
    random.shuffle(common_files)

    # Split into train/val
    split_idx = int(len(common_files) * train_ratio)
    train_files = common_files[:split_idx]
    val_files = common_files[split_idx:]

    print(f"[INFO] Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_base, split, "cubert"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "thorlabs"), exist_ok=True)

    def move_pairs(file_list, split):
        split_start = time.time()
        for i, fname in enumerate(file_list, start=1):
            src_cubert = os.path.join(cubert_dir, fname)
            src_thorlabs = os.path.join(thorlabs_dir, fname)
            dst_cubert = os.path.join(output_base, split, "cubert", fname)
            dst_thorlabs = os.path.join(output_base, split, "thorlabs", fname)

            shutil.move(src_cubert, dst_cubert)
            shutil.move(src_thorlabs, dst_thorlabs)

            if i <= 3:  # show first few for debugging
                print(f"[DEBUG] Example {split} pair moved: {fname}")
            if i % progress_interval == 0:
                elapsed = time.time() - split_start
                print(f"[INFO] Moved {i}/{len(file_list)} {split} pairs in {elapsed:.1f} sec")

    print("[INFO] Moving train set...")
    move_pairs(train_files, "training")
    print("[INFO] Moving val set...")
    move_pairs(val_files, "validation")

    total_elapsed = time.time() - start_time
    print(f"[DONE] Dataset split complete in {total_elapsed:.1f} sec")

if __name__ == "__main__":
    cubert_dir = r"/scratch/general/nfs1/u1528328/res_chart_run_2026_02_04/res_chart_run_2026_02_04/cubert"
    thorlabs_dir = r"/scratch/general/nfs1/u1528328/res_chart_run_2026_02_04/res_chart_run_2026_02_04/thorlabs"
    output_base = r"/scratch/general/nfs1/u1528328/res_chart_run_2026_02_04/res_chart_run_2026_02_04/split"

    # Helper to extract the shared prefix for pairing (prefix-suffix method)
    def get_prefix(filename, suffix):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
        return None
    
    # Helper to extract ID from filename (timestamp or numeric method)
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

    # Get all .tif files
    cubert_files_all = sorted([f for f in os.listdir(cubert_dir) if f.lower().endswith(".tif")])
    thorlabs_files_all = sorted([f for f in os.listdir(thorlabs_dir) if f.lower().endswith(".tif")])

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
        common_prefixes = sorted(list(common_prefixes_method1))
        cubert_matched = cubert_prefixes
        thorlabs_matched = thorlabs_prefixes
        matching_method = "prefix-suffix"
    else:
        # Use ID-based method
        common_prefixes = sorted(list(common_ids_method2))
        cubert_matched = cubert_ids
        thorlabs_matched = thorlabs_ids
        matching_method = "ID-based (timestamp/numeric)"

    print(f"[INFO] Matching method: {matching_method}")
    print(f"[INFO] Found {len(common_prefixes)} paired images.")
    
    if len(common_prefixes) > 0:
        print(f"[INFO] Example pairs:")
        for i, pair_id in enumerate(common_prefixes[:3]):
            print(f"  {thorlabs_matched[pair_id]} <-> {cubert_matched[pair_id]}")
        if len(common_prefixes) > 3:
            print(f"  ... and {len(common_prefixes) - 3} more")
    
    if not common_prefixes:
        print("[ERROR] No matching paired filenames found. Check directory paths and naming conventions.")
        exit(1)

    # Shuffle for random split
    random.seed(42)
    random.shuffle(common_prefixes)

    # Split into train/val
    train_ratio = 0.8
    split_idx = int(len(common_prefixes) * train_ratio)
    train_prefixes = common_prefixes[:split_idx]
    val_prefixes = common_prefixes[split_idx:]

    print(f"[INFO] Train pairs: {len(train_prefixes)}, Val pairs: {len(val_prefixes)}")

    # Create output directories
    for split in ["training", "validation"]:
        os.makedirs(os.path.join(output_base, split, "cubert"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "thorlabs"), exist_ok=True)

    def move_pairs(prefix_list, split):
        for i, prefix in enumerate(prefix_list, start=1):
            # Use cubert_matched and thorlabs_matched (works for both methods)
            cubert_file = cubert_matched[prefix]
            thorlabs_file = thorlabs_matched[prefix]
            src_cubert = os.path.join(cubert_dir, cubert_file)
            src_thorlabs = os.path.join(thorlabs_dir, thorlabs_file)
            dst_cubert = os.path.join(output_base, split, "cubert", cubert_file)
            dst_thorlabs = os.path.join(output_base, split, "thorlabs", thorlabs_file)

            shutil.move(src_cubert, dst_cubert)
            shutil.move(src_thorlabs, dst_thorlabs)

            if i <= 3:
                print(f"[DEBUG] Example {split} pair moved: {cubert_file}, {thorlabs_file}")
            if i % 50 == 0:
                print(f"[INFO] Moved {i}/{len(prefix_list)} {split} pairs")

    print("[INFO] Moving train set...")
    move_pairs(train_prefixes, "training")
    print("[INFO] Moving val set...")
    move_pairs(val_prefixes, "validation")

    print("[DONE] Dataset split complete.")


    # split_paired_dataset(cubert_dir, thorlabs_dir, output_base, train_ratio=0.8, seed=42)