import csv
from pathlib import Path

import numpy as np
import torch
from tifffile import imread
from torch.utils.data import Dataset

from path_utils import remap_dataset_path
from regime_dataset import (
    RAW_BIT_DEPTH_MAX,
    RAW_POL_CHANNELS,
    RECON_SPECTRAL_BANDS,
    RECON_TOTAL_BANDS,
    ensure_float32,
    load_raw_thorlabs,
    load_reconstruction_cube,
)


ROI_REGIMES = {
    "roi_recon_spatial": {"source": "recon"},
    "roi_recon_spectral": {"source": "recon"},
    "roi_recon_polarimetric": {"source": "recon"},
    "roi_recon_specpol": {"source": "recon"},
    "roi_recon_uncertainty_spectral": {"source": "recon"},
    "roi_recon_uncertainty_specpol": {"source": "recon"},
    "roi_raw_avgpol": {"source": "raw"},
    "roi_raw_pol": {"source": "raw"},
}


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def crop_center_shift_up(arr: np.ndarray, roi: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    c, h, w = arr.shape
    crop_size = min(roi, h, w)
    y0 = (h - crop_size) // 2 - (crop_size // 2)
    x0 = (w - crop_size) // 2
    y0 = max(0, min(y0, h - crop_size))
    x0 = max(0, min(x0, w - crop_size))
    return arr[:, y0:y0 + crop_size, x0:x0 + crop_size]


def validate_feature_vector(feature: torch.Tensor, *, task: str, regime: str, source_path: str):
    if not torch.isfinite(feature).all():
        raise ValueError(
            f"Non-finite ROI features detected: task={task}, regime={regime}, path={source_path}"
        )


def roi_mean(arr: np.ndarray, roi: int):
    cropped = crop_center_shift_up(arr, roi)
    return cropped.reshape(cropped.shape[0], -1).mean(axis=1)


def derive_roi_recon_features(cube: np.ndarray, regime: str, roi_size: int):
    recon = cube[:, :RECON_SPECTRAL_BANDS, :, :]

    if regime == "roi_recon_spatial":
        image = np.mean(recon, axis=(0, 1), keepdims=False)
        return np.asarray([float(crop_center_shift_up(image, roi_size).mean())], dtype=np.float32)

    if regime == "roi_recon_spectral":
        spectral = np.mean(recon, axis=0)
        return roi_mean(spectral, roi_size).astype(np.float32)

    if regime == "roi_recon_polarimetric":
        polar = np.mean(recon, axis=1)
        return roi_mean(polar, roi_size).astype(np.float32)

    if regime == "roi_recon_specpol":
        chunks = []
        for pol_idx in range(recon.shape[0]):
            chunks.append(roi_mean(recon[pol_idx], roi_size))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    if regime == "roi_recon_uncertainty_spectral":
        spectral = np.mean(cube, axis=0)
        return roi_mean(spectral, roi_size).astype(np.float32)

    if regime == "roi_recon_uncertainty_specpol":
        chunks = []
        for pol_idx in range(cube.shape[0]):
            chunks.append(roi_mean(cube[pol_idx], roi_size))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported ROI reconstruction regime {regime}")


def derive_roi_raw_features(raw: np.ndarray, regime: str, roi_size: int):
    raw = raw[:RAW_POL_CHANNELS]
    if regime == "roi_raw_avgpol":
        avg = np.mean(raw, axis=0, keepdims=True)
        return np.asarray([float(crop_center_shift_up(avg[0], roi_size).mean())], dtype=np.float32)
    if regime == "roi_raw_pol":
        return roi_mean(raw, roi_size).astype(np.float32)
    raise ValueError(f"Unsupported ROI raw regime {regime}")


class RoiFeatureDataset(Dataset):
    def __init__(self, rows, task: str, regime: str, roi_size: int = 32, dataset_root=None):
        if regime not in ROI_REGIMES:
            raise ValueError(f"Unknown ROI regime {regime}")
        self.rows = list(rows)
        self.task = task
        self.regime = regime
        self.roi_size = int(roi_size)
        self.dataset_root = dataset_root
        self.label_names = sorted({row["class_label"] for row in self.rows})
        self.label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if ROI_REGIMES[self.regime]["source"] == "recon":
            recon_path = remap_dataset_path(row["selected_recon_path"], self.dataset_root)
            cube = load_reconstruction_cube(recon_path)
            feature = torch.from_numpy(derive_roi_recon_features(cube, self.regime, self.roi_size))
            source_path = recon_path
        else:
            raw_path = remap_dataset_path(row["raw_path"], self.dataset_root)
            raw = load_raw_thorlabs(raw_path) / RAW_BIT_DEPTH_MAX
            feature = torch.from_numpy(derive_roi_raw_features(raw, self.regime, self.roi_size))
            source_path = raw_path

        validate_feature_vector(feature, task=self.task, regime=self.regime, source_path=source_path)
        label = self.label_to_idx[row["class_label"]]
        return feature.float(), label, row["sample_id"]


def summarize_roi_dataset(dataset: RoiFeatureDataset):
    if len(dataset) == 0:
        return {
            "samples": 0,
            "features": 0,
            "min": None,
            "max": None,
            "classes": dataset.label_names,
        }

    feature, _, _ = dataset[0]
    return {
        "samples": len(dataset),
        "features": int(feature.shape[0]),
        "min": float(feature.min().item()),
        "max": float(feature.max().item()),
        "classes": dataset.label_names,
    }
