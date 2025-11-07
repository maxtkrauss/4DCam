import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F

# Load the cropped master dark frames (only once, outside __getitem__)
thorlabs_dark_cropped = np.load("/scratch/general/nfs1/u1528328/img_dir/dark_frames/thorlabs_display_masterdark_cropped.npy")  # Shape: (5, 660, 660)
cubert_dark_cropped = np.load("/scratch/general/nfs1/u1528328/img_dir/dark_frames/cubert_display_masterdark_cropped.npy")  # Shape: (106, 120, 120)

import cv2

def upsample_bicubic(hsi, target_size=(660, 660)):
    upsampled = np.zeros((hsi.shape[0], *target_size), dtype=np.float32)
    for i in range(hsi.shape[0]):
        upsampled[i] = cv2.resize(hsi[i], target_size, interpolation=cv2.INTER_CUBIC)
    return upsampled

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
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'thorlabs')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'cubert')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.opt = opt
        self.polarization = opt.polarization
        self.video_mode = opt.video_mode
        self.GT_upsample = opt.GT_upsample
        self.norm_bitwise = opt.norm_bitwise

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
        
        A_path = self.A_paths[index % self.A_size]
        A = io.imread(A_path).astype(np.float32)  # Shape: (5, 660, 660)
        
        if self.video_mode == False:
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B = io.imread(B_path).astype(np.float32)  # Shape: (106, 120, 120)

        if self.video_mode == True:
            # Set dummy tensor
            B = torch.zeros((106, 120, 120), dtype=torch.float32)  # Dummy tensor with expected shape

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

        if self.GT_upsample:
            # Modify ground-truth size (bicubic interpolation)     
            B = upsample_bicubic(B, (660,660)) # Shape: (106, 660, 660)
        else:
            # Modify ground-truth size (padding)
            B = np.pad(B, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0) # Shape: 106x128x128


        
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()
        # interpolated B instead of pad
        # B = B.unsqueeze(0)
        # B = F.interpolate(B, size=(128, 128), mode='bilinear', align_corners=False)
        # B = B.squeeze(0)

        #A = torch.from_numpy(A).float()
        #B = torch.from_numpy(B).float()

        netG = self.opt.netG

        A = A.unsqueeze(0)  # Add batch dimension
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
        A = A.squeeze(0)  # Remove batch dimension

        # print(f"Final A shape: {A.shape}, Final B shape: {B.shape}")

        #print(f'A_paths: {A_path} B_paths: {B_path}')
        # print(A.shape)
        # print(B.shape)

        if self.video_mode == True:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': 'dummy'}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        


    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
