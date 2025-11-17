#!/usr/bin/env python3
"""
5-fold cross-validation - reuses main.py code for 100% consistency.
Train on 10x augmented data, test on non-augmented data.
"""

import os
import sys
import time
import shutil
import argparse
import csv
import random
from copy import deepcopy

import torch
import torch.utils.data
from torchvision.transforms import transforms

# Import everything from main.py to ensure identical behavior
from models.mobilenetv3 import mobilenet_v3_large
from utils.dataset import ImageFolder
from utils.transforms import (MultiChannelResize, MultiChannelRandomHorizontalFlip, 
                               MultiChannelRandomVerticalFlip, MultiChannelNormalize, 
                               ToTensorIfNeeded, ConvertToGrayscale)
import utils


def get_images_by_class(data_dir):
    """Get all image paths organized by class."""
    class_images = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = sorted([f for f in os.listdir(class_path) if f.endswith('.tif')])
            class_images[class_name] = images
    return class_images


def create_fold_splits(class_images, n_folds=5, random_order=False, random_seed=42):
    """Create n_folds splits for cross-validation.
    
    Args:
        class_images: Dict mapping class names to lists of image filenames
        n_folds: Number of folds for cross-validation
        random_order: If True, shuffle images randomly before splitting
        random_seed: Random seed for reproducibility (only used if random_order=True)
    """
    folds = []
    class_fold_indices = {}
    
    for class_name, images in class_images.items():
        n_images = len(images)
        fold_size = n_images // n_folds
        
        # Create shuffled indices if random_order is True
        if random_order:
            indices_list = list(range(n_images))
            rng = random.Random(random_seed)
            rng.shuffle(indices_list)
        else:
            indices_list = list(range(n_images))
        
        # Split shuffled/sequential indices into folds
        indices = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = n_images if i == n_folds - 1 else (i + 1) * fold_size
            indices.append(indices_list[start_idx:end_idx])
        
        class_fold_indices[class_name] = indices
    
    for fold_idx in range(n_folds):
        test_files = {}
        train_files = {}
        
        for class_name, images in class_images.items():
            test_indices = class_fold_indices[class_name][fold_idx]
            test_files[class_name] = [images[i] for i in test_indices]
            
            train_indices = []
            for i in range(n_folds):
                if i != fold_idx:
                    train_indices.extend(class_fold_indices[class_name][i])
            train_files[class_name] = [images[i] for i in train_indices]
        
        folds.append((train_files, test_files))
    
    return folds


def create_fold_directories(fold_idx, train_files, test_files, 
                           original_dir, augmented_dir, output_base):
    """Create train/test directories for a specific fold using symlinks."""
    fold_dir = os.path.join(output_base, f'fold_{fold_idx}')
    train_dir = os.path.join(fold_dir, 'train')
    test_dir = os.path.join(fold_dir, 'test')
    
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    original_dir = os.path.abspath(original_dir)
    augmented_dir = os.path.abspath(augmented_dir)
    
    # Create train set from augmented data
    for class_name, filenames in train_files.items():
        class_train_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        
        for filename in filenames:
            base_name = os.path.splitext(filename)[0]
            aug_class_dir = os.path.join(augmented_dir, class_name)
            if os.path.exists(aug_class_dir):
                for aug_file in os.listdir(aug_class_dir):
                    if aug_file.startswith(base_name + '_aug') and aug_file.endswith('.tif'):
                        src = os.path.join(aug_class_dir, aug_file)
                        dst = os.path.join(class_train_dir, aug_file)
                        os.symlink(src, dst)
    
    # Create test set from original data
    for class_name, filenames in test_files.items():
        class_test_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_test_dir, exist_ok=True)
        
        for filename in filenames:
            src = os.path.join(original_dir, class_name, filename)
            dst = os.path.join(class_test_dir, filename)
            os.symlink(src, dst)
    
    return train_dir, test_dir


def train_one_fold(train_dir, test_dir, args, fold_idx):
    """Train and evaluate one fold - uses EXACT same setup as main.py."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Create transforms WITHOUT normalization first (need to know num_channels)
    # NOTE: If grayscale mode, we need to load the full multi-channel data first,
    # then convert to grayscale, then normalize as 1-channel
    train_transform_list = [
        ToTensorIfNeeded(),
    ]
    
    # Add grayscale conversion BEFORE resizing (more efficient - resize 1 channel instead of 424)
    if args.grayscale:
        train_transform_list.append(ConvertToGrayscale())
    
    # Now add resize and augmentations (operating on correct number of channels)
    train_transform_list.extend([
        MultiChannelResize((224, 224)),
        MultiChannelRandomHorizontalFlip(p=0.5),
        MultiChannelRandomVerticalFlip(p=0.5),
    ])
    
    # Temporary transform without normalization to load first image
    temp_transform = transforms.Compose(train_transform_list)
    
    # Create temporary dataset to detect channels
    normalize_per_channel = not args.global_normalize
    temp_dataset = ImageFolder(train_dir, transform=temp_transform, num_channels=args.num_channels, 
                               normalize_per_channel=normalize_per_channel)
    
    # Auto-detect channels from the dataset if not specified
    if args.num_channels is None:
        args.num_channels = temp_dataset.num_channels
        if fold_idx == 0:
            print(f'Auto-detected {args.num_channels} channels from training data')
    
    # CRITICAL FIX: If grayscale mode, the transform reduces channels to 1
    # We need to normalize with 1 channel, not the original number
    normalization_channels = 1 if args.grayscale else args.num_channels
    
    if fold_idx == 0:
        print(f'Using {"global" if args.global_normalize else "per-channel"} normalization')
        print(f'Normalization: {normalization_channels} channel(s) (original: {args.num_channels}, grayscale: {args.grayscale})')
    
    # Now add normalization with correct number of channels
    train_transform_list.append(
        MultiChannelNormalize(
            mean=[0.5] * normalization_channels,
            std=[0.5] * normalization_channels
        )
    )
    
    train_transform = transforms.Compose(train_transform_list)
    
    test_transform_list = [
        ToTensorIfNeeded(),
    ]
    
    # Add grayscale conversion BEFORE resizing (more efficient)
    if args.grayscale:
        test_transform_list.append(ConvertToGrayscale())
    
    # Now add resize
    test_transform_list.append(MultiChannelResize((224, 224)))
    
    # Add normalization with correct number of channels (1 if grayscale, otherwise original)
    test_transform_list.append(
        MultiChannelNormalize(
            mean=[0.5] * normalization_channels,
            std=[0.5] * normalization_channels
        )
    )
    
    test_transform = transforms.Compose(test_transform_list)
    
    # EXACT same dataset setup as main.py (channels already auto-detected above)
    train_dataset = ImageFolder(train_dir, transform=train_transform, num_channels=args.num_channels,
                                normalize_per_channel=normalize_per_channel)
    test_dataset = ImageFolder(test_dir, transform=test_transform, num_channels=args.num_channels,
                               normalize_per_channel=normalize_per_channel)
    
    print(f'Fold {fold_idx}: Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')
    print(f'  Classes: {train_dataset.classes}')
    print(f'  Input channels (from data): {args.num_channels}')
    print(f'  Model input channels (after transforms): {normalization_channels}')
    
    # EXACT same data loaders as main.py
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=4 if args.workers > 0 else None,  # Prefetch 4 batches per worker
        persistent_workers=True if args.workers > 0 else False  # Keep workers alive between epochs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=True if args.workers > 0 else False
    )
    
    # EXACT same model setup as main.py
    # CRITICAL: Model must use the number of channels AFTER transform (1 if grayscale, otherwise original)
    model = mobilenet_v3_large(
        num_classes=len(train_dataset.classes), 
        in_channels=normalization_channels,
        dropout=args.dropout
    ).to(device)
    
    # EXACT same optimizer/scheduler/criterion as main.py
    parameters = utils.add_weight_decay(model, weight_decay=args.weight_decay)
    criterion = utils.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = utils.RMSprop(parameters, lr=args.lr, alpha=0.9, eps=1e-3, weight_decay=0, momentum=args.momentum)
    scheduler = utils.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
        warmup_epochs=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr_init
    )
    
    # EXACT same DataParallel wrapping as main.py
    model = torch.nn.DataParallel(model)
    
    # Training loop - EXACT same structure as main.py
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train using main.py's train_one_epoch (but inline for simplicity)
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        # CRITICAL: scheduler.step with epoch number (same as main.py)
        scheduler.step(epoch + 1)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f'  Epoch [{epoch+1}/{args.epochs}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%')
    
    return best_acc, train_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='5-fold CV using main.py code')
    parser.add_argument('--original-dir', type=str, default='dataset_nosplit/textiles')
    parser.add_argument('--augmented-dir', type=str, default='dataset_nosplit/textiles_augmented')
    parser.add_argument('--output-dir', type=str, default='cv_folds_textiles')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--random-order', action='store_true', help='shuffle images randomly before splitting into folds (creates non-contiguous test groups)')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed for reproducibility (only used with --random-order)')
    
    # EXACT same hyperparameters as main.py
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--lr-step-size', type=int, default=20)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--warmup-lr-init', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--num-channels', type=int, default=None, help='number of input channels (default: auto-detect from dataset)')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--grayscale', action='store_true', help='convert multi-channel input to grayscale by averaging channels (sets num-channels to 1)')
    parser.add_argument('--global-normalize', action='store_true', help='use global normalization instead of per-channel (essential for hyperspectral data)')
    parser.add_argument("--disable-ema", action='store_true', help='disable exponential moving average (recommended for small datasets)')

    
    args = parser.parse_args()
    
    print('='*80)
    print('5-FOLD CROSS-VALIDATION (using main.py code)')
    print('='*80)
    print(f'Original: {args.original_dir}, Augmented: {args.augmented_dir}')
    print(f'Config: LR={args.lr}, BS={args.batch_size}, Epochs={args.epochs}, Dropout={args.dropout}')
    if args.grayscale:
        print(f'Grayscale mode: converting to 1 channel by averaging')
    else:
        print(f'Channels: {args.num_channels}')
    if args.random_order:
        print(f'Random fold splitting: ENABLED (seed={args.random_seed})')
    else:
        print(f'Sequential fold splitting: continuous groups')
    print('='*80 + '\n')
    
    # Get images and create folds
    class_images = get_images_by_class(args.original_dir)
    print('Images per class:')
    for class_name, images in class_images.items():
        print(f'  {class_name}: {len(images)} images')
    print()
    
    folds = create_fold_splits(class_images, args.n_folds, args.random_order, args.random_seed)
    results = []
    
    # Train each fold
    for fold_idx in range(args.n_folds):
        print(f'\n{"="*80}')
        print(f'FOLD {fold_idx + 1}/{args.n_folds}')
        print(f'{"="*80}')
        
        train_files, test_files = folds[fold_idx]
        
        print('Test set:')
        for class_name, filenames in test_files.items():
            print(f'  {class_name}: {len(filenames)} images')
            if args.random_order and len(filenames) <= 15:
                # Show first few test files to verify randomization
                print(f'    Sample files: {", ".join(filenames[:5])}{"..." if len(filenames) > 5 else ""}')
        print('Train set:')
        for class_name, filenames in train_files.items():
            print(f'  {class_name}: {len(filenames)} images (augmented versions)')
        print()
        
        train_dir, test_dir = create_fold_directories(
            fold_idx, train_files, test_files,
            args.original_dir, args.augmented_dir, args.output_dir
        )
        
        start_time = time.time()
        best_acc, final_train_acc, final_test_acc = train_one_fold(train_dir, test_dir, args, fold_idx)
        elapsed_time = time.time() - start_time
        
        results.append({
            'fold': fold_idx + 1,
            'best_test_acc': best_acc,
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
            'time_minutes': elapsed_time / 60.0
        })
        
        print(f'\nFold {fold_idx + 1} Results:')
        print(f'  Best Test: {best_acc:.2f}%')
        print(f'  Final Test: {final_test_acc:.2f}%')
        print(f'  Final Train: {final_train_acc:.2f}%')
        print(f'  Time: {elapsed_time/60.0:.1f} min')
        
        # Save intermediate results after each fold
        csv_path = os.path.join(args.output_dir, 'cv_results.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['fold', 'best_test_acc', 'final_test_acc', 'final_train_acc', 'time_minutes',
                         'lr', 'batch_size', 'epochs', 'dropout', 'weight_decay', 'label_smoothing']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                row = res.copy()
                row.update({
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'dropout': args.dropout,
                    'weight_decay': args.weight_decay,
                    'label_smoothing': args.label_smoothing
                })
                writer.writerow(row)
        print(f'  Results saved to: {csv_path}')
        
        # Delete fold directory to save disk space
        fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
        if os.path.exists(fold_dir):
            print(f'  Cleaning up fold directory to save disk space...')
            shutil.rmtree(fold_dir)
    
    # Summary
    print('\n' + '='*80)
    print('CROSS-VALIDATION SUMMARY')
    print('='*80)
    print(f'{"Fold":<8} {"Best Test":<12} {"Final Test":<12} {"Final Train":<12} {"Time(min)":<10}')
    print('-'*80)
    
    best_accs = [r['best_test_acc'] for r in results]
    final_test_accs = [r['final_test_acc'] for r in results]
    final_train_accs = [r['final_train_acc'] for r in results]
    
    for res in results:
        print(f'{res["fold"]:<8} {res["best_test_acc"]:>10.2f}%  {res["final_test_acc"]:>10.2f}%  {res["final_train_acc"]:>10.2f}%  {res["time_minutes"]:>8.1f}')
    
    print('-'*80)
    mean_best = sum(best_accs) / len(best_accs)
    mean_final_test = sum(final_test_accs) / len(final_test_accs)
    mean_final_train = sum(final_train_accs) / len(final_train_accs)
    std_best = (sum([(x - mean_best)**2 for x in best_accs]) / len(best_accs))**0.5
    std_final_test = (sum([(x - mean_final_test)**2 for x in final_test_accs]) / len(final_test_accs))**0.5
    
    print(f'{"MEAN":<8} {mean_best:>10.2f}%  {mean_final_test:>10.2f}%  {mean_final_train:>10.2f}%')
    print(f'{"STD":<8} {std_best:>10.2f}%  {std_final_test:>10.2f}%')
    print('='*80)
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, 'cv_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['fold', 'best_test_acc', 'final_test_acc', 'final_train_acc', 'time_minutes',
                     'lr', 'batch_size', 'epochs', 'dropout', 'weight_decay', 'label_smoothing']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for res in results:
            row = res.copy()
            row.update({
                'lr': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'dropout': args.dropout,
                'weight_decay': args.weight_decay,
                'label_smoothing': args.label_smoothing
            })
            writer.writerow(row)
    
    print(f'\nResults saved to: {csv_path}')
    print(f'NOTE: Fold directories were deleted after completion to save disk space.')


if __name__ == '__main__':
    main()
