#!/usr/bin/env python3
"""
Generate augmented dataset from multi-channel TIFF images (any dimension).
Creates 10x dataset with various augmentations and saves to disk.
Handles TIFF images with any number of channels (grayscale, RGB, hyperspectral, etc.)
"""

import os
import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import random
from tqdm import tqdm



def rotate_image(img, angle):
    """Rotate image by angle degrees. Works with any number of channels."""
    from scipy.ndimage import rotate
    # img shape: (C, H, W)
    rotated = np.zeros_like(img)
    for c in range(img.shape[0]):
        rotated[c] = rotate(img[c], angle, reshape=False, order=1)
    return rotated

def flip_horizontal(img):
    """Flip image horizontally. Works with any number of channels."""
    return np.flip(img, axis=2).copy()

def flip_vertical(img):
    """Flip image vertically. Works with any number of channels."""
    return np.flip(img, axis=1).copy()

def add_noise(img, noise_level=0.02):
    """Add Gaussian noise"""
    noise = np.random.randn(*img.shape) * noise_level * img.std()
    return np.clip(img + noise, img.min(), img.max())

def adjust_brightness(img, factor):
    """Adjust brightness"""
    return np.clip(img * factor, img.min(), img.max())

def adjust_contrast(img, factor):
    """Adjust contrast"""
    mean = img.mean(axis=(1, 2), keepdims=True)
    return np.clip((img - mean) * factor + mean, img.min(), img.max())

def random_crop_and_resize(img, crop_factor=0.9):
    """Randomly crop and resize back to original size"""
    from scipy.ndimage import zoom
    c, h, w = img.shape
    new_h, new_w = int(h * crop_factor), int(w * crop_factor)
    
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    
    cropped = img[:, top:top+new_h, left:left+new_w]
    
    # Resize back
    zoom_factors = (1, h/new_h, w/new_w)
    resized = zoom(cropped, zoom_factors, order=1)
    return resized

def generate_augmentations(img):
    """Generate multiple augmented versions of an image"""
    augmented = []
    
    # 1. Original
    augmented.append(('original', img.copy()))
    
    # 2. Horizontal flip
    augmented.append(('hflip', flip_horizontal(img)))
    
    # 3. Vertical flip
    augmented.append(('vflip', flip_vertical(img)))
    
    # 4. Both flips
    augmented.append(('hvflip', flip_vertical(flip_horizontal(img))))
    
    # 5. Rotate 90
    augmented.append(('rot90', rotate_image(img, 90)))
    
    # 6. Rotate 180
    augmented.append(('rot180', rotate_image(img, 180)))
    
    # 7. Rotate 270
    augmented.append(('rot270', rotate_image(img, 270)))
    
    # 8. Slight rotation + crop
    augmented.append(('rot15_crop', random_crop_and_resize(rotate_image(img, 15), 0.95)))
    
    # 9. Brightness up + noise
    augmented.append(('bright_noise', add_noise(adjust_brightness(img, 1.15), 0.01)))
    
    # 10. Contrast + brightness down
    augmented.append(('contrast_dark', adjust_brightness(adjust_contrast(img, 1.2), 0.9)))
    
    return augmented

def augment_dataset(input_dir, output_dir, augmentation_factor=10, max_channels=None):
    """
    Augment all TIFF images in input_dir and save to output_dir.
    Works with TIFF images of any dimension.
    
    Args:
        input_dir: Path to original dataset (e.g., 'dataset/foleage/train')
        output_dir: Path to save augmented dataset
        augmentation_factor: Number of augmented versions per image
        max_channels: Maximum number of channels to keep (None = keep all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    total_images = 0
    total_augmented = 0
    channel_info = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all TIFF files
        tiff_files = list(class_dir.glob('*.tif')) + list(class_dir.glob('*.tiff'))
        
        print(f"\nProcessing class: {class_name} ({len(tiff_files)} images)")
        
        for tiff_file in tqdm(tiff_files, desc=f"Augmenting {class_name}"):
            # Load image with tifffile (assumes already in C, H, W format)
            img = imread(str(tiff_file))
            
            # Verify it's 3D and handle 2D grayscale edge case
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim != 3:
                raise ValueError(f"Expected 3D image, got shape {img.shape} for {tiff_file}")
            
            # Track channel info
            if class_name not in channel_info:
                channel_info[class_name] = {'shapes': set(), 'channels': set()}
            channel_info[class_name]['shapes'].add(img.shape)
            channel_info[class_name]['channels'].add(img.shape[0])
            
            # Limit channels if specified
            if max_channels is not None and img.shape[0] > max_channels:
                print(f"  Note: {tiff_file.name} has {img.shape[0]} channels, using first {max_channels}")
                img = img[:max_channels]
            
            total_images += 1
            
            # Generate augmentations
            augmentations = generate_augmentations(img)
            
            # Save augmented images (up to augmentation_factor)
            for idx, (aug_name, aug_img) in enumerate(augmentations[:augmentation_factor]):
                # Create filename
                base_name = tiff_file.stem
                output_filename = f"{base_name}_aug{idx}_{aug_name}.tif"
                output_filepath = output_class_dir / output_filename
                
                # Save as TIFF (preserve dtype)
                # For multi-channel images (C, H, W), save as multi-page TIFF with proper metadata
                imwrite(str(output_filepath), aug_img.astype(img.dtype), 
                       photometric='minisblack', metadata={'axes': 'CYX'})
                total_augmented += 1
    
    print(f"\n{'='*60}")
    print(f"Augmentation Complete!")
    print(f"{'='*60}")
    print(f"Original images: {total_images}")
    print(f"Augmented images: {total_augmented}")
    print(f"Expansion factor: {total_augmented/total_images:.1f}x")
    print(f"\nChannel information by class:")
    for class_name, info in channel_info.items():
        print(f"  {class_name}: {sorted(info['channels'])} channels")
    print(f"\nOutput directory: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment TIFF dataset (any number of channels)")
    parser.add_argument("--input-dir", default="dataset/foleage/train", 
                       help="Input dataset directory")
    parser.add_argument("--output-dir", default="dataset/foleage_augmented/train", 
                       help="Output directory for augmented dataset")
    parser.add_argument("--augmentation-factor", type=int, default=10, 
                       help="Number of augmented versions per image")
    parser.add_argument("--max-channels", type=int, default=None,
                       help="Maximum number of channels to keep (default: keep all)")
    parser.add_argument("--test-input", default="dataset/foleage/test",
                       help="Input test dataset directory")
    parser.add_argument("--test-output", default="dataset/foleage_augmented/test",
                       help="Output test directory (copies without augmentation)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Channel TIFF Dataset Augmentation")
    print("="*60)
    if args.max_channels:
        print(f"Max channels: {args.max_channels}")
    else:
        print("Max channels: ALL (no limit)")
    print("="*60)
    
    # Augment training set
    print("\n[1/2] Augmenting TRAINING set...")
    augment_dataset(args.input_dir, args.output_dir, args.augmentation_factor, args.max_channels)
    
    # Copy test set without augmentation (just original images)
    print("\n[2/2] Copying TEST set (no augmentation)...")
    from shutil import copy2
    
    test_input = Path(args.test_input)
    test_output = Path(args.test_output)
    
    if test_input.exists():
        for class_dir in test_input.iterdir():
            if class_dir.is_dir():
                output_class_dir = test_output / class_dir.name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                tiff_files = list(class_dir.glob('*.tif')) + list(class_dir.glob('*.tiff'))
                
                # Apply max_channels limit if specified
                for tiff_file in tiff_files:
                    if args.max_channels:
                        # Load, limit channels, save (assumes already in C, H, W format)
                        img = imread(str(tiff_file))
                        if img.ndim == 2:
                            img = img[np.newaxis, :, :]
                        if img.shape[0] > args.max_channels:
                            img = img[:args.max_channels]
                        imwrite(str(output_class_dir / tiff_file.name), img.astype(img.dtype),
                               photometric='minisblack', metadata={'axes': 'CYX'})
                    else:
                        # Direct copy
                        copy2(str(tiff_file), str(output_class_dir / tiff_file.name))
                
                print(f"  Copied {len(tiff_files)} images from {class_dir.name}")
    else:
        print(f"  Test input directory not found: {test_input}")
        print(f"  Skipping test set copy")
    
    print("\n" + "="*60)
    print("✅ Dataset augmentation complete!")
    print("="*60)
    print(f"\nTo train with augmented dataset, use:")
    print(f"python main.py --data-path {args.output_dir.rsplit('/', 1)[0]} --batch-size 64 --epochs 50")
