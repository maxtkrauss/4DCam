#!/usr/bin/env python3
"""
Flatten 4D hyperspectral images to 3D for neural network training.
Transforms (4, 212, 128, 128) -> (424, 128, 128) by:
1. Taking first 106 channels from dimension 1
2. Flattening dimensions 0 and 1: (4, 106) -> 424 channels
"""

import os
import sys
from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
from tqdm import tqdm


def flatten_4d_image(img):
    """
    Flatten 4D image (4, N, H, W) to 3D (4*106, H, W).
    Takes first 106 channels from dimension 1.
    
    Args:
        img: 4D numpy array of shape (4, N, H, W) where N >= 106
    
    Returns:
        3D numpy array of shape (424, H, W)
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {img.shape}")
    
    # Take first 106 channels from dimension 1
    img_subset = img[:, :106, :, :]  # (4, 106, H, W)
    
    # Flatten first two dimensions: (4, 106, H, W) -> (424, H, W)
    flattened = img_subset.reshape(-1, img_subset.shape[2], img_subset.shape[3])
    
    print(f"  Transformed {img.shape} -> {flattened.shape}")
    return flattened


def process_directory(input_dir, output_dir):
    """Process all .tif files in directory structure."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all .tif files
    tif_files = list(input_path.rglob("*.tif"))
    
    if len(tif_files) == 0:
        print(f"No .tif files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} .tif files")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    processed = 0
    skipped = 0
    errors = 0
    
    for tif_file in tqdm(tif_files, desc="Processing images"):
        try:
            # Read image
            img = imread(str(tif_file))
            
            # Get relative path from input directory
            rel_path = tif_file.relative_to(input_path)
            output_file = output_path / rel_path
            
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process based on dimensionality
            if img.ndim == 4:
                # Flatten 4D image
                flattened = flatten_4d_image(img)
                imwrite(str(output_file), flattened, compression='deflate',
                       photometric='minisblack', metadata={'axes': 'CYX'})
                processed += 1
            elif img.ndim == 3:
                # Already 3D, just copy
                print(f"  Skipping {rel_path} (already 3D: {img.shape})")
                imwrite(str(output_file), img, compression='deflate',
                       photometric='minisblack', metadata={'axes': 'CYX'})
                skipped += 1
            else:
                print(f"  WARNING: Unexpected shape {img.shape} for {rel_path}")
                skipped += 1
                
        except Exception as e:
            print(f"  ERROR processing {tif_file}: {e}")
            errors += 1
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed (4D->3D): {processed}")
    print(f"Skipped (already 3D): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total: {len(tif_files)}")
    print("="*80)


def main():
    input_dir = "/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/spectral_datasets/4D_camo_mix_plant"
    output_dir = "/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/spectral_datasets/flat_camo_mix_plant"
    
    print("="*80)
    print("FLATTEN 4D HYPERSPECTRAL IMAGES")
    print("="*80)
    print(f"Transform: (4, 212, H, W) -> (424, H, W)")
    print(f"  - Take first 106 channels from dim 1")
    print(f"  - Flatten dims 0,1: (4, 106) -> 424 channels")
    print("="*80)
    print()
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    process_directory(input_dir, output_dir)
    
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
