import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tifffile import imread
from torch.utils.data import Dataset

from path_utils import remap_dataset_path

RECON_SPECTRAL_BANDS = 106
RECON_TOTAL_BANDS = 212
RAW_POL_CHANNELS = 4
RAW_BIT_DEPTH_MAX = 4095.0
RANGE_TOLERANCE = 1e-4

REGIMES = {
    "recon_spatial": {"source": "recon"},
    "recon_spectral": {"source": "recon"},
    "recon_polarimetric": {"source": "recon"},
    "recon_specpol": {"source": "recon"},
    "recon_uncertainty_spectral": {"source": "recon"},
    "recon_uncertainty_specpol": {"source": "recon"},
    "raw_avgpol": {"source": "raw"},
    "raw_pol": {"source": "raw"},
}


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_float32(arr):
    return np.asarray(arr, dtype=np.float32)


def load_reconstruction_cube(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Reconstruction path not found: {path}. Confirm the dataset drive is mounted and the manifest paths are valid."
        )
    arr = ensure_float32(imread(path))
    arr = np.squeeze(arr)
    if arr.ndim != 4:
        raise ValueError(f"Expected reconstruction tensor with 4 dimensions, got {arr.shape} at {path}")

    if arr.shape[0] == 4 and arr.shape[1] == RECON_TOTAL_BANDS:
        return arr
    if arr.shape[-2:] == (128, 128) and arr.shape[-1] == RECON_TOTAL_BANDS and arr.shape[0] == 4:
        return np.moveaxis(arr, -1, 1)
    if arr.shape[-2:] == (128, 128) and arr.shape[0] == RECON_TOTAL_BANDS and arr.shape[1] == 4:
        return np.moveaxis(arr, 0, 1)

    raise ValueError(f"Unsupported reconstruction tensor shape {arr.shape} at {path}")


def load_raw_thorlabs(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw Thorlabs path not found: {path}. Confirm the dataset drive is mounted and the manifest paths are valid."
        )
    arr = ensure_float32(imread(path))
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 32 and arr.shape[2] > 32:
            pass
        elif arr.shape[-1] <= 8 and arr.shape[0] > 32 and arr.shape[1] > 32:
            arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unsupported raw Thorlabs tensor shape {arr.shape} at {path}")
    else:
        raise ValueError(f"Unsupported raw Thorlabs tensor shape {arr.shape} at {path}")

    if arr.shape[0] < RAW_POL_CHANNELS:
        raise ValueError(f"Expected at least {RAW_POL_CHANNELS} raw polarization channels at {path}, got {arr.shape}")
    return arr


def validate_tensor_range(tensor: torch.Tensor, *, task: str, regime: str, source_path: str, stage: str):
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"Non-finite values found during {stage}: task={task}, regime={regime}, path={source_path}"
        )

    tmin = float(tensor.min().item())
    tmax = float(tensor.max().item())
    if tmin < -RANGE_TOLERANCE or tmax > 1.0 + RANGE_TOLERANCE:
        raise ValueError(
            f"Out-of-range values during {stage}: task={task}, regime={regime}, path={source_path}, "
            f"min={tmin:.6f}, max={tmax:.6f}"
        )


def derive_recon_regime(cube: np.ndarray, regime: str):
    recon = cube[:, :RECON_SPECTRAL_BANDS, :, :]

    if regime == "recon_spatial":
        return np.mean(recon, axis=(0, 1), keepdims=False)[None, :, :]
    if regime == "recon_spectral":
        return np.mean(recon, axis=0)
    if regime == "recon_polarimetric":
        return np.mean(recon, axis=1)
    if regime == "recon_specpol":
        return recon.reshape(-1, recon.shape[-2], recon.shape[-1])
    if regime == "recon_uncertainty_spectral":
        return np.mean(cube, axis=0)
    if regime == "recon_uncertainty_specpol":
        return cube.reshape(-1, cube.shape[-2], cube.shape[-1])
    raise ValueError(f"Unsupported recon regime {regime}")


def derive_raw_regime(raw: np.ndarray, regime: str):
    raw = raw[:RAW_POL_CHANNELS]
    if regime == "raw_avgpol":
        return np.mean(raw, axis=0, keepdims=True)
    if regime == "raw_pol":
        return raw
    raise ValueError(f"Unsupported raw regime {regime}")


class PairedRegimeDataset(Dataset):
    def __init__(self, rows, task: str, regime: str, image_size=224, train=False, dataset_root=None):
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime {regime}")

        self.rows = list(rows)
        self.task = task
        self.regime = regime
        self.image_size = int(image_size)
        self.train = train
        self.dataset_root = dataset_root
        self.label_names = sorted({row["class_label"] for row in self.rows})
        self.label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}

    def __len__(self):
        return len(self.rows)

    def _normalize_source(self, row):
        if REGIMES[self.regime]["source"] == "recon":
            recon_path = remap_dataset_path(row["selected_recon_path"], self.dataset_root)
            cube = load_reconstruction_cube(recon_path)
            tensor = torch.from_numpy(derive_recon_regime(cube, self.regime).copy())
            validate_tensor_range(
                tensor,
                task=self.task,
                regime=self.regime,
                source_path=recon_path,
                stage="post_recon_derivation",
            )
            return tensor, recon_path

        raw_path = remap_dataset_path(row["raw_path"], self.dataset_root)
        raw = load_raw_thorlabs(raw_path)
        raw = raw / RAW_BIT_DEPTH_MAX
        tensor = torch.from_numpy(derive_raw_regime(raw, self.regime).copy())
        validate_tensor_range(
            tensor,
            task=self.task,
            regime=self.regime,
            source_path=raw_path,
            stage="post_raw_derivation",
        )
        return tensor, raw_path

    def __getitem__(self, index):
        row = self.rows[index]
        image, source_path = self._normalize_source(row)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.train:
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, dims=[-1])
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, dims=[-2])

        validate_tensor_range(
            image,
            task=self.task,
            regime=self.regime,
            source_path=source_path,
            stage="dataset_return",
        )

        label = self.label_to_idx[row["class_label"]]
        return image, label, row["sample_id"]


def summarize_dataset(dataset: PairedRegimeDataset):
    if len(dataset) == 0:
        return {
            "samples": 0,
            "channels": 0,
            "height": 0,
            "width": 0,
            "min": None,
            "max": None,
            "classes": dataset.label_names,
        }

    image, _, _ = dataset[0]
    return {
        "samples": len(dataset),
        "channels": int(image.shape[0]),
        "height": int(image.shape[1]),
        "width": int(image.shape[2]),
        "min": float(image.min().item()),
        "max": float(image.max().item()),
        "classes": dataset.label_names,
    }
