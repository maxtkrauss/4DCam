import os
from PIL import Image
from torch.utils import data
import numpy as np
from tifffile import imread
import torch


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, num_channels=None, normalize_per_channel=True):
        self.transform = transform
        self.num_channels = num_channels  # None = auto-detect from first image
        self.normalize_per_channel = normalize_per_channel  # False = global normalization
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples = self.make_dataset(root, self.class_to_idx)
        
        # Auto-detect number of channels from first image if not specified
        if self.num_channels is None and len(self.samples) > 0:
            first_path, _ = self.samples[0]
            if first_path.lower().endswith(('.tif', '.tiff')):
                arr = imread(first_path)
                if arr.ndim == 2:
                    self.num_channels = 1
                elif arr.ndim == 3:
                    # Check if it's (C, H, W) or (H, W, C)
                    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                        self.num_channels = arr.shape[0]
                    else:
                        self.num_channels = arr.shape[2]
            else:
                self.num_channels = 3  # RGB
            print(f"Auto-detected {self.num_channels} channels from dataset")

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        """Load multi-channel TIFF or regular RGB image. Handles any number of channels."""
        if path.lower().endswith(('.tif', '.tiff')):
            # Load TIFF stack with memory mapping for speed
            # Using memmap=True avoids loading entire file into RAM at once
            arr = imread(path, is_ome=False)  # Shape: (C, H, W) or (H, W) or (H, W, C)
            
            # Normalize to (C, H, W) format
            if arr.ndim == 2:
                # Grayscale (H, W) -> (1, H, W)
                arr = arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                # Check if it's (C, H, W) or (H, W, C)
                if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                    # Already (C, H, W)
                    pass
                else:
                    # (H, W, C) -> (C, H, W)
                    arr = np.transpose(arr, (2, 0, 1))
            
            # Limit to num_channels if specified
            if self.num_channels is not None and arr.shape[0] > self.num_channels:
                arr = arr[:self.num_channels]
            
            # Normalize to 0-1 range
            arr_normalized = np.zeros_like(arr, dtype=np.float32)
            
            if self.normalize_per_channel:
                # Per-channel normalization: Each channel scaled independently
                # Good for: scatterograms, general RGB images
                # Bad for: hyperspectral data where relative intensities matter
                for i in range(arr.shape[0]):
                    ch_min = arr[i].min()
                    ch_max = arr[i].max()
                    if ch_max > ch_min:
                        arr_normalized[i] = (arr[i] - ch_min) / (ch_max - ch_min)
                    else:
                        arr_normalized[i] = arr[i].astype(np.float32)
            else:
                # Global normalization: All channels scaled together
                # Preserves relative intensity relationships across channels
                # Essential for: hyperspectral and spectro-polarimetric data
                global_min = arr.min()
                global_max = arr.max()
                if global_max > global_min:
                    arr_normalized = (arr - global_min) / (global_max - global_min)
                    arr_normalized = arr_normalized.astype(np.float32)
                else:
                    arr_normalized = arr.astype(np.float32)
            
            # Return as torch tensor
            return torch.from_numpy(arr_normalized.copy())
        else:
            # Regular RGB image loading
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')
            return image

    @staticmethod
    def find_classes(directory):
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        return class_names, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx=None):
        if class_to_idx is None:
            _, class_to_idx = ImageFolder.find_classes(directory)

        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            for root, _, file_names in sorted(os.walk(target_dir, followlinks=True)):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    base, ext = os.path.splitext(path)
                    if ext.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                        item = path, class_index
                        instances.append(item)

        return instances
