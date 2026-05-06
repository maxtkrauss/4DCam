import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tifffile import imread, imwrite

from augment_dataset import augment_dataset


WINDOWS_DATASET_ROOT = PureWindowsPath(r"D:\4DCam_Data\Textile-Camo Classification")
RECON_SPECTRAL_BANDS = 106
RECON_TOTAL_BANDS = 212
RAW_POL_CHANNELS = 4

SUPPORTED_REGIMES = (
    "raw_pol",
    "raw_avgpol",
    "recon_spatial",
    "recon_spectral",
    "recon_polarimetric",
    "recon_specpol",
    "raw_stokes",
    "recon_stokes_polarimetric",
    "recon_stokes_specpol",
)

SUPPORTED_TASKS = ("textile_3way", "camo_binary")


@dataclass(frozen=True)
class RegimeConfig:
    num_channels: int | None
    grayscale: bool
    batch_size: int


REGIME_CONFIG = {
    "raw_pol": RegimeConfig(num_channels=4, grayscale=False, batch_size=64),
    "raw_avgpol": RegimeConfig(num_channels=None, grayscale=True, batch_size=64),
    "recon_spatial": RegimeConfig(num_channels=None, grayscale=True, batch_size=64),
    "recon_polarimetric": RegimeConfig(num_channels=4, grayscale=False, batch_size=64),
    "recon_spectral": RegimeConfig(num_channels=106, grayscale=False, batch_size=32),
    "recon_specpol": RegimeConfig(num_channels=424, grayscale=False, batch_size=8),
    "raw_stokes": RegimeConfig(num_channels=3, grayscale=False, batch_size=64),
    "recon_stokes_polarimetric": RegimeConfig(num_channels=3, grayscale=False, batch_size=64),
    "recon_stokes_specpol": RegimeConfig(num_channels=318, grayscale=False, batch_size=8),
}


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_key(row):
    return f"{row.get('dataset', '')}|{row.get('image_number', '')}|{row.get('source_path', '')}"


def resolve_reconstruction_path(row, selection_by_key):
    match_count = int(row.get("recon_match_count", "0") or "0")
    recon_paths = [part.strip() for part in row.get("recon_paths", "").split(";") if part.strip()]

    if match_count == 1:
        if not recon_paths:
            raise ValueError(f"Missing reconstruction path for {row_key(row)}")
        return recon_paths[0]

    if match_count == 2:
        key = row_key(row)
        if key not in selection_by_key:
            raise ValueError(f"Missing manual selection for {key}")
        return selection_by_key[key]["selected_recon_path"]

    raise ValueError(f"Unsupported reconstruction match count {match_count} for {row_key(row)}")


def assign_folds(rows, n_splits=5, seed=42):
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    for task_rows in by_task.values():
        task_rows.sort(key=lambda item: (item["class_label"], item["dataset"], int(item["image_number"])))
        labels = np.asarray([row["class_label"] for row in task_rows])
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(task_rows)), labels)):
            for idx in test_idx:
                task_rows[idx]["fold"] = fold

    return rows


def build_canonical_manifest(paired_manifest_path: Path, selections_path: Path, output_path: Path, n_splits=5, seed=42):
    paired_rows = read_csv_rows(paired_manifest_path)
    selection_rows = read_csv_rows(selections_path)
    selection_by_key = {row_key(row): row for row in selection_rows}

    canonical_rows = []
    for row in paired_rows:
        if row.get("notes") != "paired":
            continue

        task = "textile_3way" if row["group"] == "textile" else "camo_binary"
        class_label = row["class"]
        sample_id = f"{task}|{row['dataset']}|{row['image_number']}"
        canonical_rows.append(
            {
                "task": task,
                "group": row["group"],
                "dataset": row["dataset"],
                "class_label": class_label,
                "sample_id": sample_id,
                "image_number": row["image_number"],
                "raw_path": row["source_path"],
                "selected_recon_path": resolve_reconstruction_path(row, selection_by_key),
            }
        )

    canonical_rows = assign_folds(canonical_rows, n_splits=n_splits, seed=seed)
    fieldnames = [
        "task",
        "group",
        "dataset",
        "class_label",
        "sample_id",
        "image_number",
        "raw_path",
        "selected_recon_path",
        "fold",
    ]
    write_csv_rows(output_path, canonical_rows, fieldnames)
    return canonical_rows


def ensure_manifest(manifest_path: Path, paired_manifest: Path, selections: Path, seed: int):
    if manifest_path.exists():
        return manifest_path
    build_canonical_manifest(
        paired_manifest_path=paired_manifest,
        selections_path=selections,
        output_path=manifest_path,
        n_splits=5,
        seed=seed,
    )
    return manifest_path


def remap_dataset_path(path_str: str, dataset_root: str | None = None) -> str:
    if not dataset_root:
        return path_str

    try:
        windows_path = PureWindowsPath(path_str)
    except Exception:
        return path_str

    try:
        relative = windows_path.relative_to(WINDOWS_DATASET_ROOT)
    except Exception:
        return path_str

    return str(Path(dataset_root) / Path(*relative.parts))


def ensure_float32(arr):
    return np.asarray(arr, dtype=np.float32)


def load_reconstruction_cube(path: str):
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


def derive_stokes_from_pol(pol: np.ndarray):
    if pol.shape[0] < RAW_POL_CHANNELS:
        raise ValueError(f"Expected at least {RAW_POL_CHANNELS} polarization channels, got {pol.shape}")
    i0, i45, i90, i135 = pol[:RAW_POL_CHANNELS]
    s0 = i0 + i90
    s1 = i0 - i90
    s2 = i45 - i135
    return np.stack([s0, s1, s2], axis=0)


def derive_array_for_regime(row, regime: str, dataset_root: str | None):
    if regime.startswith("raw_"):
        raw_path = remap_dataset_path(row["raw_path"], dataset_root)
        arr = load_raw_thorlabs(raw_path)[:RAW_POL_CHANNELS]
        if regime == "raw_stokes":
            return derive_stokes_from_pol(arr), raw_path
        return arr, raw_path

    recon_path = remap_dataset_path(row["selected_recon_path"], dataset_root)
    cube = load_reconstruction_cube(recon_path)
    recon = cube[:, :RECON_SPECTRAL_BANDS, :, :]
    stokes = None

    if regime == "recon_spatial":
        return np.mean(recon, axis=(0, 1), keepdims=False)[None, :, :], recon_path
    if regime == "recon_polarimetric":
        return np.mean(recon, axis=1), recon_path
    if regime == "recon_spectral":
        return np.mean(recon, axis=0), recon_path
    if regime == "recon_specpol":
        return recon.reshape(-1, recon.shape[-2], recon.shape[-1]), recon_path
    if regime in ("recon_stokes_polarimetric", "recon_stokes_specpol"):
        stokes = derive_stokes_from_pol(recon)
    if regime == "recon_stokes_polarimetric":
        return np.mean(stokes, axis=1), recon_path
    if regime == "recon_stokes_specpol":
        return stokes.reshape(-1, stokes.shape[-2], stokes.shape[-1]), recon_path

    raise ValueError(f"Unsupported regime {regime}")


def sanitize_sample_id(sample_id: str):
    return sample_id.replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")


def write_tiff(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    imwrite(str(path), arr.astype(np.float32), photometric="minisblack", metadata={"axes": "CYX"})


def is_scratch_path(path: Path):
    normalized = str(path.resolve()).replace("\\", "/").lower()
    markers = ("/scratch/", "/scratch", "/gscratch/", "/gscratch", "/scr1/", "/tmp/scratch/")
    return any(marker in normalized for marker in markers)


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _count_tiffs(path: Path):
    if not path.exists():
        return 0
    return len(list(path.rglob("*.tif"))) + len(list(path.rglob("*.tiff")))


def prepared_dataset_is_usable(prepared_root: Path):
    original_train_dir = prepared_root / "original_train"
    original_test_dir = prepared_root / "original_test"
    augmented_train_dir = prepared_root / "augmented_train"
    manifest_path = prepared_root / "materialized_manifest.csv"

    if not manifest_path.exists():
        return False
    if _count_tiffs(original_train_dir) == 0:
        return False
    if _count_tiffs(original_test_dir) == 0:
        return False
    if _count_tiffs(augmented_train_dir) == 0:
        return False
    return True


def materialize_fold_dataset(rows, task: str, regime: str, fold: int, dataset_root: str | None, scratch_work_dir: Path):
    task_rows = [row for row in rows if row["task"] == task]
    train_rows = [row for row in task_rows if int(row["fold"]) != fold]
    test_rows = [row for row in task_rows if int(row["fold"]) == fold]

    if not train_rows or not test_rows:
        raise ValueError(f"No train/test rows found for task={task} fold={fold}")

    prepared_root = scratch_work_dir / "prepared" / task / regime / f"fold_{fold}"
    original_train_dir = prepared_root / "original_train"
    original_test_dir = prepared_root / "original_test"
    augmented_train_dir = prepared_root / "augmented_train"

    if prepared_dataset_is_usable(prepared_root):
        return {
            "prepared_root": prepared_root,
            "train_dir": augmented_train_dir,
            "test_dir": original_test_dir,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "reused": True,
        }

    ensure_clean_dir(original_train_dir)
    ensure_clean_dir(original_test_dir)
    ensure_clean_dir(augmented_train_dir)

    manifest_rows = []
    for split_name, split_rows, target_dir in (
        ("train", train_rows, original_train_dir),
        ("test", test_rows, original_test_dir),
    ):
        for row in split_rows:
            arr, source_path = derive_array_for_regime(row, regime=regime, dataset_root=dataset_root)
            sample_name = sanitize_sample_id(row["sample_id"])
            output_path = target_dir / row["class_label"] / f"{sample_name}.tif"
            write_tiff(output_path, arr)
            manifest_rows.append(
                {
                    "task": task,
                    "regime": regime,
                    "fold": fold,
                    "split": split_name,
                    "sample_id": row["sample_id"],
                    "class_label": row["class_label"],
                    "source_path": source_path,
                    "materialized_path": str(output_path),
                    "channels": int(arr.shape[0] if arr.ndim == 3 else 1),
                }
            )

    augment_dataset(str(original_train_dir), str(augmented_train_dir), augmentation_factor=10, max_channels=None)
    write_csv_rows(
        prepared_root / "materialized_manifest.csv",
        manifest_rows,
        [
            "task",
            "regime",
            "fold",
            "split",
            "sample_id",
            "class_label",
            "source_path",
            "materialized_path",
            "channels",
        ],
    )

    return {
        "prepared_root": prepared_root,
        "train_dir": augmented_train_dir,
        "test_dir": original_test_dir,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "reused": False,
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def copy_tree_contents(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def summarize_materialized_channels(paths: Iterable[Path]):
    info = []
    for path in paths:
        arr = imread(str(path))
        arr = np.asarray(arr)
        if arr.ndim == 2:
            channels = 1
        elif arr.ndim == 3:
            channels = arr.shape[0]
        else:
            channels = -1
        info.append({"path": str(path), "shape": list(arr.shape), "channels": channels})
    return info
