import os
import torch
import numpy as np
from skimage import io
import torch.nn.functional as F

# Mock imports for standalone testing (remove in real use)
class BaseDataset:
    def __init__(self, opt): pass
def make_dataset(dir_path, max_size=float("inf")):
    # Return all .npy files in the directory
    return [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith(".npy")]


# -------------------------------------------------------------------------
# Simplified Dataset
# -------------------------------------------------------------------------
class AlignedDataset(BaseDataset):
    """
    Paired Thorlabs–Cubert dataset loader for hyperspectral reconstruction.

    - Thorlabs input: (1, 1008, 1008) or (1008, 1008)
        -> Upsampled to (1, 1024, 1024)
    - Cubert ground truth: (106, 187, 187)
        -> Kept unchanged
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'thorlabs')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'cubert')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.opt = opt
        self.polarization = opt.polarization
        self.video_mode = opt.video_mode
        self.norm_bitwise = opt.norm_bitwise

    def __getitem__(self, index):
        """Load and preprocess a paired Thorlabs (A) and Cubert (B) image."""
        # --- Load Thorlabs input (A) ---
        A_path = self.A_paths[index % self.A_size]
        A = np.load(A_path).astype(np.float32)  # Can be (H, W) or (C, H, W)

        # If grayscale (H, W), add channel dimension
        if A.ndim == 2:
            A = A[np.newaxis, :, :]  # → (1, H, W)

        # --- Load Cubert ground truth (B) ---
        if not self.video_mode:
            B_path = self.B_paths[index % self.B_size]
            B = np.load(B_path).astype(np.float32)  # Shape: (106, 187, 187)
        else:
            B_path = 'dummy'
            B = np.zeros((106, 187, 187), dtype=np.float32)

        # --- Normalize both to [0, 1] ---
        if self.norm_bitwise:
            A /= 4095.0
            B /= 4095.0
        else:
            A = (A - A.min()) / (A.max() - A.min() + 1e-8)
            B = (B - B.min()) / (B.max() - B.min() + 1e-8)

        # --- Handle polarization (keep all or single channel) ---
        if self.polarization is None:
            # Unpolarized input — keep as is (single channel)
            pass
        else:
            raise ValueError("Polarization selection not supported for unpolarized dataset.")

        # --- Convert to torch tensors ---
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        # --- Upsample Thorlabs input to (1024, 1024) ---
        # Add batch dimension for interpolation: (1, C, H, W)
        A = A.unsqueeze(0)
        A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False)
        A = A.squeeze(0)  # → (C, 1024, 1024)

        # --- Final shapes ---
        # A: (1, 1024, 1024)
        # B: (106, 187, 187)
        # During training, crop model output (106, 256, 256)
        # to (106, 187, 187) before loss calculation.

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


# -------------------------------------------------------------------------
# Simple Test Function
# -------------------------------------------------------------------------
def test_dataset():
    """Create fake data and test dataset pipeline."""
    import tempfile
    from types import SimpleNamespace

    # Temporary directories
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "train/thorlabs"), exist_ok=True)
    os.makedirs(os.path.join(root, "train/cubert"), exist_ok=True)

    # Create fake Thorlabs grayscale data (1008, 1008)
    fake_A = np.random.rand(1008, 1008).astype(np.float32)
    np.save(os.path.join(root, "train/thorlabs/fake_A.npy"), fake_A)

    # Create fake Cubert data (106, 187, 187)
    fake_B = np.random.rand(106, 187, 187).astype(np.float32)
    np.save(os.path.join(root, "train/cubert/fake_B.npy"), fake_B)

    # Define mock options
    opt = SimpleNamespace(
        dataroot=root,
        phase="train",
        max_dataset_size=float("inf"),
        polarization=None,       # unpolarized grayscale
        video_mode=False,
        norm_bitwise=False
    )

    # Initialize and test dataset
    dataset = AlignedDataset(opt)
    sample = dataset[0]

    print("=== Dataset Test ===")
    print(f"Thorlabs input shape (A): {tuple(sample['A'].shape)}")
    print(f"Cubert GT shape (B): {tuple(sample['B'].shape)}")
    print(f"A path: {sample['A_paths']}")
    print(f"B path: {sample['B_paths']}")
    print("====================")


# Run test if executed directly
if __name__ == "__main__":
    test_dataset()
