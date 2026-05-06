import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F
from data.input_perturbations import apply_optional_input_shift, build_aligned_samples

# Load the cropped master dark frames (only once, outside __getitem__)
#thorlabs_dark_cropped = np.load("/scratch/general/nfs1/u1528328/img_dir/dark_frames/thorlabs_display_masterdark_cropped.npy")  # Shape: (5, 660, 660)
#cubert_dark_cropped = np.load("/scratch/general/nfs1/u1528328/img_dir/dark_frames/cubert_display_masterdark_cropped.npy")  # Shape: (106, 120, 120)

import cv2

def upsample_bicubic(hsi, target_size=(660, 660)):
    upsampled = np.zeros((hsi.shape[0], *target_size), dtype=np.float32)
    for i in range(hsi.shape[0]):
        upsampled[i] = cv2.resize(hsi[i], target_size, interpolation=cv2.INTER_CUBIC)
    return upsampled


MODEL_SHAPE_SPECS = {
    'unet_128': {'input_hw': (128, 128), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'unet_256': {'input_hw': (256, 256), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'unet_512': {'input_hw': (512, 512), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'unet_1024': {'input_hw': (1024, 1024), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'unet_1024_mod': {'input_hw': (1024, 1024), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'unet_1024_to_256': {'input_hw': (1024, 1024), 'target_hw': (256, 256), 'target_mode': 'resize'},
    'unet_2048': {'input_hw': (2048, 2048), 'target_hw': (410, 410), 'target_mode': 'strict'},
    'nafnet_128': {'input_hw': (128, 128), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'nafnet_256': {'input_hw': (256, 256), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'nafnet_512': {'input_hw': (512, 512), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
    'nafnet_1024': {'input_hw': (1024, 1024), 'target_hw': (128, 128), 'target_mode': 'pad_120_to_128'},
}


def _shape3(x):
    return tuple(int(v) for v in x.shape[-3:])


def _shape_error(netG, actual_A, actual_B, expected_A, expected_B, A_path, B_path, detail):
    return (
        f"Shape contract failed for netG='{netG}': {detail}\n"
        f"  required input A:  ({expected_A[0]}, {expected_A[1]}, {expected_A[2]})\n"
        f"  actual input A:    {actual_A}\n"
        f"  required target B: ({expected_B[0]}, {expected_B[1]}, {expected_B[2]})\n"
        f"  actual target B:   {actual_B}\n"
        f"  A_path: {A_path}\n"
        f"  B_path: {B_path}"
    )


def _get_shape_spec(opt):
    if opt.netG not in MODEL_SHAPE_SPECS:
        choices = ', '.join(sorted(MODEL_SHAPE_SPECS))
        raise NotImplementedError(
            f"Generator model name '{opt.netG}' is not recognized in aligned_dataset.py. "
            f"Known model shape specs: {choices}"
        )
    spec = dict(MODEL_SHAPE_SPECS[opt.netG])
    if opt.GT_upsample:
        spec['target_hw'] = (660, 660)
        spec['target_mode'] = 'resize'
    return spec


def _prepare_target(B, spec, netG, expected_channels, B_path):
    target_hw = spec['target_hw']
    target_mode = spec['target_mode']
    actual = _shape3(B)

    if actual[0] != expected_channels:
        raise ValueError(
            f"Shape contract failed for netG='{netG}': target B must have "
            f"{expected_channels} channels before model-specific resizing, got {actual[0]}.\n"
            f"  B_path: {B_path}"
        )

    if target_mode == 'pad_120_to_128':
        if actual[-2:] == target_hw:
            return B
        if actual[-2:] != (120, 120):
            raise ValueError(
                f"Shape contract failed for netG='{netG}': target B must be "
                f"({expected_channels}, 120, 120) or ({expected_channels}, 128, 128), got {actual}.\n"
                f"  B_path: {B_path}"
            )
        return np.pad(B, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0)

    if target_mode == 'strict':
        if actual[-2:] != target_hw:
            raise ValueError(
                f"Shape contract failed for netG='{netG}': this full-resolution architecture "
                f"requires target B shape ({expected_channels}, {target_hw[0]}, {target_hw[1]}), got {actual}.\n"
                f"  B_path: {B_path}"
            )
        return B

    if target_mode == 'resize':
        if actual[-2:] == target_hw:
            return B
        return upsample_bicubic(B, target_hw)

    raise ValueError(f"Unknown target preparation mode '{target_mode}' for netG='{netG}'.")


def _resize_input(A, target_hw):
    A = A.unsqueeze(0)
    if tuple(A.shape[-2:]) != target_hw:
        A = F.interpolate(A, size=target_hw, mode='bilinear', align_corners=False)
    return A.squeeze(0)


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.samples = build_aligned_samples(opt)
        if not self.samples:
            diagnostics = getattr(opt, 'aligned_sample_diagnostics', [])
            detail_lines = [
                f"AlignedDataset found zero samples for dataroot='{opt.dataroot}', phase='{opt.phase}', video_mode={opt.video_mode}.",
                "Expected folders are '<dataroot>/<phase>/thorlabs' and, unless video_mode=True, '<dataroot>/<phase>/cubert'.",
            ]
            for diag in diagnostics:
                detail_lines.extend([
                    f"  dataroot: {diag['dataroot']}",
                    f"    A/thorlabs: {diag['dir_A']} exists={diag['dir_A_exists']} files={diag['a_count']}",
                    f"    B/cubert:   {diag['dir_B']} exists={diag['dir_B_exists']} files={diag['b_count']}",
                    f"    usable pairs: {diag['pair_count']}",
                ])
            if not diagnostics:
                detail_lines.append("  No dataroots were inspected.")
            detail_lines.append("For full-field video-only inference, set the DATASETS entry to video_mode=True. For alternate folder names, set phase/train_phase/test_phase in DATASETS.")
            raise ValueError("\n".join(detail_lines))
        self.A_size = len(self.samples)
        self.B_size = len(self.samples)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.opt = opt
        self.polarization = opt.polarization
        self.video_mode = opt.video_mode
        self.GT_upsample = opt.GT_upsample
        self.norm_bitwise = opt.norm_bitwise
        self.shape_spec = _get_shape_spec(opt)
        self.expected_target_nc = opt.output_nc // 2 if getattr(opt, 'use_nll', False) else opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
        index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor)       -- an image in the input domain
        B (tensor)       -- its corresponding image in the target domain
        A_paths (str)    -- image paths
        B_paths (str)    -- image paths
        """
        '''
        AB_path = self.AB_paths[index]
        base_name = os.path.basename(AB_path)
        AB_path = os.path.join(AB_path, base_name)
        
        base_name_2 = base_name.replace('_ms', '')
        A_path = os.path.join(AB_path, base_name_2 + '_RGB.bmp')
        A = Image.open(A_path).convert('L')
        
        # Load hyperspectral images
        B_images = []
        #B_paths = sorted([os.path.join(AB_path, fname) for fname in os.listdir(AB_path) if self.is_image_file(fname)])

        for i in range(1, 60):
            filename = f'{base_name}_{i:02d}.png'
            B_path = os.path.join(AB_path, filename)
            #print(f'b path: {B_path}')
            B_image = Image.open(B_path)  # Convert to grayscale

            # Check min, max, and bit depth
            B_array = np.array(B_image)
            #print(f'B_image {i} - min: {B_array.min()}, max: {B_array.max()}, dtype: {B_array.dtype}, shape: {B_array.shape}')

            B_images.append(B_image)
'''
        
        sample = self.samples[index % self.A_size]
        A_path = sample['A_path']
        A = io.imread(A_path).astype(np.float32)  # Shape: (5, 660, 660)
        
        if self.video_mode == False:
            B_path = sample['B_path']
            B = io.imread(B_path).astype(np.float32)  # Shape: (106, 120, 120)
        else:
            B_path = 'dummy'
            target_h, target_w = self.shape_spec['target_hw']
            B = np.zeros((self.expected_target_nc, target_h, target_w), dtype=np.float32)

        #print(f'A path: {A_path}')
        #print(f'B path: {B_path}')

        # **Apply dark subtraction**
        #A = np.clip(A - thorlabs_dark_cropped, 0, None)  # Subtract & threshold negative values to 0
        #B = np.clip(B - cubert_dark_cropped, 0, None)    # Subtract & threshold

        # Normalize to [0, 1]
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

        A, shift_meta = apply_optional_input_shift(A, self.opt)

        B = _prepare_target(B, self.shape_spec, self.opt.netG, self.expected_target_nc, B_path)

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        A = _resize_input(A, self.shape_spec['input_hw'])
        expected_A = (self.input_nc, *self.shape_spec['input_hw'])
        expected_B = (self.expected_target_nc, *self.shape_spec['target_hw'])
        if _shape3(A) != expected_A or _shape3(B) != expected_B:
            raise ValueError(
                _shape_error(
                    self.opt.netG,
                    _shape3(A),
                    _shape3(B),
                    expected_A,
                    expected_B,
                    A_path,
                    B_path,
                    'dataloader preprocessing produced incompatible tensors',
                )
            )

        #print(f'A_paths: {A_path} B_paths: {B_path}')
        # print(A.shape)
        # print(B.shape)

        if self.video_mode == True:
            return {
                'A': A,
                'B': B,
                'A_paths': A_path,
                'B_paths': 'dummy',
                'is_shifted': torch.tensor(shift_meta['is_shifted'], dtype=torch.float32),
                'shift_strength': torch.tensor(shift_meta['shift_strength'], dtype=torch.float32),
                'source_domain': torch.tensor(sample['source_domain'], dtype=torch.long),
                'is_ood': torch.tensor(max(shift_meta['is_shifted'], sample['is_domain_ood']), dtype=torch.float32),
            }
        else:
            return {
                'A': A,
                'B': B,
                'A_paths': A_path,
                'B_paths': B_path,
                'is_shifted': torch.tensor(shift_meta['is_shifted'], dtype=torch.float32),
                'shift_strength': torch.tensor(shift_meta['shift_strength'], dtype=torch.float32),
                'source_domain': torch.tensor(sample['source_domain'], dtype=torch.long),
                'is_ood': torch.tensor(max(shift_meta['is_shifted'], sample['is_domain_ood']), dtype=torch.float32),
            }
        


    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
