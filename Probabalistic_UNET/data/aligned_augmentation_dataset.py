import os
import re
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from skimage import io
import torch
import numpy as np
import torch.nn.functional as F
import cv2

"""
AlignedDataset for paired image-to-image translation tasks.

This dataset class loads paired images from two directories (A and B), applies optional 16x geometric augmentations
(4 rotations × 2 horizontal flips × 2 vertical flips), and returns each augmented pair as a separate sample.
Augmentation is controlled by the `use_aug16` flag in the options. Each sample includes the augmentation parameters
used, allowing for reproducible and diverse training data.

- Supports natural sorting of file names for consistent pairing.
- Handles normalization and resizing to match the generator's input size.
- Designed for efficient use with PyTorch DataLoader and compatible with the HAMscope pix2pix pipeline.
"""

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def generate_transforms():
    transforms = []
    for angle in [0, 90, 180, 270]:
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                transforms.append((angle, flip_h, flip_v))
    return transforms

def upsample_bicubic(hsi, target_size=(660, 660)):
    upsampled = np.zeros((hsi.shape[0], *target_size), dtype=np.float32)
    for i in range(hsi.shape[0]):
        upsampled[i] = cv2.resize(hsi[i], target_size, interpolation=cv2.INTER_CUBIC)
    return upsampled

class AlignedAugmentationDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.polarization = opt.polarization
        self.video_mode = opt.video_mode
        self.GT_upsample = opt.GT_upsample
        self.A_paths, self.B_paths = [], []
        self.use_aug16 = bool(getattr(opt, 'use_aug16', False) and getattr(opt, 'isTrain', False))
        self.aug_transforms = generate_transforms() if self.use_aug16 else [(0, False, False)]

        dir_A = os.path.join(opt.dataroot, opt.phase, 'thorlabs')
        dir_B = os.path.join(opt.dataroot, opt.phase, 'cubert')

        if os.path.exists(dir_A):
            self.A_paths.extend(make_dataset(dir_A, opt.max_dataset_size))
            if os.path.exists(dir_B):
                self.B_paths.extend(make_dataset(dir_B, opt.max_dataset_size))
            else:
                a_paths_for_folder = make_dataset(dir_A, opt.max_dataset_size)
                self.B_paths.extend(a_paths_for_folder)

        self.A_paths = sorted(self.A_paths, key=natural_sort_key)
        self.B_paths = sorted(self.B_paths, key=natural_sort_key)
        print(f"A_paths: {self.A_paths}")
        print(f"B_paths: {self.B_paths}")   
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __len__(self):
        return min(self.A_size, self.B_size) * len(self.aug_transforms)

    def __getitem__(self, index):
        base_len = min(self.A_size, self.B_size)
        aug_idx = index // base_len
        img_idx = index % base_len

        A_path = self.A_paths[img_idx]
        B_path = self.B_paths[img_idx]

        A = io.imread(A_path).astype(np.float32)  # Shape: (5, 660, 660)
        if self.video_mode == False:
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B = io.imread(B_path).astype(np.float32)  # Shape: (106, 120, 120)

        if self.video_mode == True:
            # Set dummy tensor
            B = torch.zeros((106, 120, 120), dtype=torch.float32)  # Dummy tensor with expected shape

        if self.norm_bitwise:
            A = A / 4095
            B = B / 4095
        else:
            A = (A - A.min()) / (A.max() - A.min() + 1e-8)
            B = (B - B.min()) / (B.max() - B.min() + 1e-8)

        # Select desired polarization channel
        if self.polarization == 0:
            A = A[:1, :, :] # Shape: (1, 660, 660) (0 degree pol)

        if self.polarization == 45:
            A = A[1:2,:,:] # Shape: (1, 660, 660) (45 degree pol)

        if self.polarization == 90:
            A = A[2:3, :, :] # Shape: (1, 660, 660) (90 degree pol)

        if self.polarization == 135:
            A = A[3:4, :, :] # Shape: (1, 660, 660) (135 degree pol)

        if self.GT_upsample:
            # Modify ground-truth size (bicubic interpolation)     
            B = upsample_bicubic(B, (660,660)) # Shape: (106, 660, 660)
        else:
            # Modify ground-truth size (padding)
            B = np.pad(B, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0) # Shape: 106x128x128

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        while A.dim() < 4:
            A = A.unsqueeze(0)
        while B.dim() < 4:
            B = B.unsqueeze(0)

        netG = self.opt.netG
        if netG == 'unet_128':
            A = F.interpolate(A, size=(128, 128), mode='bilinear', align_corners=False)
        elif netG == 'unet_256':
            A = F.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)
        elif netG == 'unet_512':
            # A = F.interpolate(A, size=(512, 512), mode='bilinear', align_corners=False)
            print("foo")
        elif netG == 'unet_1024' or netG == 'unet_1024_mod':
                A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False)
                # pass
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized in aligned_dataset.py' % netG)
        #A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False) # Resize to 120x120

        # Remove batch dimension but keep channel dimension
        A = A.squeeze(0)
        B = B.squeeze(0)

        # print(f"Final A shape: {A.shape}, Final B shape: {B.shape}")

        # Apply augmentation
        angle, flip_h, flip_v = self.aug_transforms[aug_idx]
        # Rotate
        if angle != 0:
            k = angle // 90
            A = torch.rot90(A, k=k, dims=[1, 2])
            B = torch.rot90(B, k=k, dims=[1, 2])
        # Horizontal flip
        if flip_h:
            A = torch.flip(A, dims=[2])
            B = torch.flip(B, dims=[2])
        # Vertical flip
        if flip_v:
            A = torch.flip(A, dims=[1])
            B = torch.flip(B, dims=[1])

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
            'aug_flag': (angle, flip_h, flip_v)
        }