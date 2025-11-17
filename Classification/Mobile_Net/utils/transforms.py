"""Custom transforms for multi-channel images"""
import torch
import torch.nn.functional as F
import random


class MultiChannelResize:
    """Resize multi-channel tensors"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # img shape: (C, H, W)
            return F.interpolate(
                img.unsqueeze(0), 
                size=self.size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            # PIL Image
            from torchvision import transforms
            return transforms.Resize(self.size)(img)


class MultiChannelRandomHorizontalFlip:
    """Randomly flip multi-channel tensors"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if random.random() < self.p:
                return torch.flip(img, dims=[-1])
            return img
        else:
            from torchvision import transforms
            return transforms.RandomHorizontalFlip(self.p)(img)


class MultiChannelRandomVerticalFlip:
    """Randomly flip multi-channel tensors vertically"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if random.random() < self.p:
                return torch.flip(img, dims=[-2])
            return img
        else:
            from torchvision import transforms
            return transforms.RandomVerticalFlip(self.p)(img)


class MultiChannelNormalize:
    """Normalize multi-channel tensors"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor) and self.mean is not None:
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            return (img - mean) / std
        return img


class ToTensorIfNeeded:
    """Convert to tensor only if not already a tensor"""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img
        else:
            from torchvision import transforms
            return transforms.ToTensor()(img)


class ConvertToGrayscale:
    """Convert multi-channel image to grayscale by averaging all channels"""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Average across all channels: (C, H, W) -> (1, H, W)
            return img.mean(dim=0, keepdim=True)
        else:
            # If numpy array (C, H, W)
            import numpy as np
            if len(img.shape) == 3:
                return np.mean(img, axis=0, keepdims=True)
            return img
