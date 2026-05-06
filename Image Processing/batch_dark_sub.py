#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import tifffile as tiff


# ============================
# INPUT DIRECTORIES
# ============================

BASE = Path("/scratch/general/nfs1/u1528328/res_chart_run_2026_02_04/res_chart_run_2026_02_04/raw")

CUBERT_DIR = BASE / "charts" / "cubert"
THORLABS_DIR = BASE / "charts" / "thorlabs"

MASTER_DARK_CUBERT = BASE / "dark" / "master_dark_cubert.tif"
MASTER_DARK_THORLABS = BASE / "dark" / "master_dark_thorlabs.tif"

# ============================
# OUTPUT DIRECTORIES
# ============================

CUBERT_OUT = CUBERT_DIR.parent / "cubert_darksub"
THORLABS_OUT = THORLABS_DIR.parent / "thorlabs_darksub"

CUBERT_OUT.mkdir(exist_ok=True)
THORLABS_OUT.mkdir(exist_ok=True)


def subtract_dark(image_path, dark, output_dir):
    img = tiff.imread(image_path)

    if img.shape != dark.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"{image_path.name}: {img.shape}\n"
            f"Dark: {dark.shape}"
        )

    # convert to float for safe subtraction
    img = img.astype(np.float32)
    dark = dark.astype(np.float32)

    corrected = img - dark

    # clip negatives
    corrected = np.clip(corrected, 0, None)

    # convert back to uint16
    corrected = corrected.astype(np.uint16)

    out_path = output_dir / image_path.name
    tiff.imwrite(out_path, corrected)


def process_folder(folder, dark_path, output_dir, label):
    print(f"\nProcessing {label}...")
    print(f"Input folder:  {folder}")
    print(f"Output folder: {output_dir}")

    dark = tiff.imread(dark_path)

    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    print(f"Total files: {len(files)}")

    for i, f in enumerate(files, 1):
        subtract_dark(f, dark, output_dir)

        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] done")

    print("Finished.")


def main():
    process_folder(CUBERT_DIR, MASTER_DARK_CUBERT, CUBERT_OUT, "CUBERT")
    process_folder(THORLABS_DIR, MASTER_DARK_THORLABS, THORLABS_OUT, "THORLABS")


if __name__ == "__main__":
    main()