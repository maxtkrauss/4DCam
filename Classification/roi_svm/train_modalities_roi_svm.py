#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import tifffile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import append_command_log, ensure_dir, write_json


MODALITIES = [
    "native_scatter_4ch",
    "recon_1ch_mean_all",
    "recon_4ch_pol_mean_wavelength",
    "recon_106ch_wavelength_mean_pol",
    "recon_424ch_full",
]

FEATURE_MODES = [
    "mean_std",
    "mean_std_q",
    "mean_std_q_minmax",
    "spatial2_mean_std",
    "spatial3_mean_std",
    "spatial4_mean_std_q",
    "downsample16_flat",
]


@dataclass(frozen=True)
class ModalityRecord:
    image_id: str
    image_key: str
    class_name: str
    label: int
    recon_path: Path
    native_path: Path
    predefined_fold: int | None = None


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train: list[ModalityRecord]
    val: list[ModalityRecord]
    test: list[ModalityRecord]


def parse_csv_strings(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-modality image and pixel-majority linear SVM analysis.")
    parser.add_argument("--dataset", choices=["textiles", "camo"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--textiles-recon-root",
        default="/media/al/samsung_2tb/mobilenet_CHPC_dataset/flat_textiles",
        help="Root containing textile reconstruction TIFFs arranged by class.",
    )
    parser.add_argument(
        "--textiles-native-root",
        default="/media/al/samsung_2tb/mobilenet_CHPC_dataset/textiles",
        help="Root containing native textile TIFFs arranged by class.",
    )
    parser.add_argument(
        "--camo-recon-root",
        default="/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/spectral_datasets/flat_camo_plant",
        help="Root containing camo/plant reconstruction TIFFs arranged by class.",
    )
    parser.add_argument(
        "--camo-native-root",
        default="/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/dataset_nosplit/Camo_Plants",
        help="Root containing native camo/plant TIFFs arranged by class.",
    )
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--fold-filter", default="0,1,2,3,4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-c-grid", default="0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="mean_std")
    parser.add_argument("--pixel-c-selection", choices=["image", "val_majority"], default="image")
    parser.add_argument("--pixel-train-samples-per-image", type=int, default=512)
    parser.add_argument("--pixel-plot-limit", type=int, default=12)
    parser.add_argument("--augment-train-copies", type=int, default=0)
    parser.add_argument("--augment-gain-range", type=float, default=0.0)
    parser.add_argument("--augment-channel-gain-range", type=float, default=0.0)
    parser.add_argument("--augment-offset-range", type=float, default=0.0)
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def image_key_from_name(name: str) -> str:
    match = re.search(r"(image_\d+)", name)
    if not match:
        raise ValueError(f"Could not parse image key from filename: {name}")
    return match.group(1)


def fold_from_name(name: str) -> int | None:
    match = re.match(r"fold(\d+)_", name)
    return None if not match else int(match.group(1))


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32, copy=False)
    lo = float(np.min(cube))
    hi = float(np.max(cube))
    if hi == lo:
        return np.zeros_like(cube, dtype=np.float32)
    return ((cube - lo) / (hi - lo)).astype(np.float32, copy=False)


def crop_cube(cube: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return cube
    x0, x1, y0, y1 = crop
    return cube[:, y0:y1, x0:x1]


def load_native(path: Path, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    cube = tifffile.imread(path)
    if cube.ndim != 3:
        raise ValueError(f"Expected native 3D TIFF, got {cube.shape}: {path}")
    if cube.shape[0] < 4 and cube.shape[-1] >= 4:
        cube = np.moveaxis(cube, -1, 0)
    cube = cube[:4].astype(np.float32, copy=False)
    return normalize_cube(crop_cube(cube, crop))


def load_reconstruction(path: Path, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    cube = tifffile.imread(path)
    if cube.ndim != 3:
        raise ValueError(f"Expected reconstruction 3D TIFF, got {cube.shape}: {path}")
    if cube.shape[0] != 424 and cube.shape[-1] == 424:
        cube = np.moveaxis(cube, -1, 0)
    if cube.shape[0] != 424:
        raise ValueError(f"Expected 424-channel reconstruction, got {cube.shape}: {path}")
    return normalize_cube(crop_cube(cube.astype(np.float32, copy=False), crop))


def transform_reconstruction(cube: np.ndarray, modality: str) -> np.ndarray:
    pol = cube.reshape(4, 106, cube.shape[1], cube.shape[2])
    if modality == "recon_1ch_mean_all":
        return cube.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    if modality == "recon_4ch_pol_mean_wavelength":
        return pol.mean(axis=1).astype(np.float32, copy=False)
    if modality == "recon_106ch_wavelength_mean_pol":
        return pol.mean(axis=0).astype(np.float32, copy=False)
    if modality == "recon_424ch_full":
        return cube.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported reconstruction modality: {modality}")


def load_modality_cube(record: ModalityRecord, modality: str, dataset: str) -> np.ndarray:
    if dataset == "textiles":
        recon_crop = (30, 90, 20, 80)
        native_crop = (155, 464, 103, 413)
    else:
        recon_crop = None
        native_crop = None

    if modality == "native_scatter_4ch":
        return load_native(record.native_path, native_crop)
    recon = load_reconstruction(record.recon_path, recon_crop)
    return transform_reconstruction(recon, modality)


def crop_policy(dataset: str) -> dict:
    if dataset == "textiles":
        return {
            "reconstruction_crop_xyxy": [30, 90, 20, 80],
            "native_crop_xyxy": [155, 464, 103, 413],
        }
    return {
        "reconstruction_crop_xyxy": None,
        "native_crop_xyxy": None,
    }


def class_config(
    dataset: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, int], dict[str, str], dict[str, str], Path, Path]:
    if dataset == "textiles":
        class_names = ["cotton", "felt", "nylon"]
        labels = {name: idx for idx, name in enumerate(class_names)}
        recon_dirs = {name: name for name in class_names}
        native_dirs = {name: name for name in class_names}
        recon_root = Path(args.textiles_recon_root)
        native_root = Path(args.textiles_native_root)
        return class_names, labels, recon_dirs, native_dirs, recon_root, native_root

    class_names = ["camo", "no_camo"]
    labels = {"camo": 0, "no_camo": 1}
    recon_dirs = {"camo": "camo", "no_camo": "no_camo"}
    native_dirs = {"camo": "camoflage", "no_camo": "plants"}
    recon_root = Path(args.camo_recon_root)
    native_root = Path(args.camo_native_root)
    return class_names, labels, recon_dirs, native_dirs, recon_root, native_root


def discover_records(dataset: str, args: argparse.Namespace) -> tuple[list[ModalityRecord], list[str], dict]:
    class_names, labels, recon_dirs, native_dirs, recon_root, native_root = class_config(dataset, args)
    records: list[ModalityRecord] = []
    unmatched = []
    for class_name in class_names:
        native_by_key = {
            image_key_from_name(path.name): path
            for path in sorted((native_root / native_dirs[class_name]).glob("*.tif"))
        }
        for recon_path in sorted((recon_root / recon_dirs[class_name]).glob("*.tif")):
            key = image_key_from_name(recon_path.name)
            native_path = native_by_key.get(key)
            if native_path is None:
                unmatched.append({"class_name": class_name, "image_key": key, "recon_path": str(recon_path)})
                continue
            records.append(
                ModalityRecord(
                    image_id=f"{class_name}/{key}",
                    image_key=key,
                    class_name=class_name,
                    label=labels[class_name],
                    recon_path=recon_path,
                    native_path=native_path,
                    predefined_fold=fold_from_name(recon_path.name),
                )
            )
    manifest = {
        "dataset": dataset,
        "recon_root": str(recon_root),
        "native_root": str(native_root),
        "class_counts": {name: sum(r.class_name == name for r in records) for name in class_names},
        "unmatched": unmatched,
    }
    return records, class_names, manifest


def write_run_audit(
    output_root: Path,
    dataset: str,
    records: list[ModalityRecord],
    class_names: list[str],
    folds: list[FoldSplit],
    modalities: list[str],
) -> None:
    if not records:
        raise RuntimeError("No records discovered.")
    example = records[0]
    modality_shapes = {}
    for modality in modalities:
        modality_shapes[modality] = list(load_modality_cube(example, modality, dataset).shape)
    fold_summary = []
    for fold in folds:
        fold_summary.append(
            {
                "fold": fold.fold_id,
                "train_count": len(fold.train),
                "val_count": len(fold.val),
                "test_count": len(fold.test),
                "test_class_counts": {
                    name: sum(r.class_name == name for r in fold.test)
                    for name in class_names
                },
            }
        )
    audit = {
        "dataset": dataset,
        "class_names": class_names,
        "record_count": len(records),
        "class_counts": {name: sum(r.class_name == name for r in records) for name in class_names},
        "modalities": modalities,
        "modality_shapes_from_first_record": modality_shapes,
        "polarization_layout": {
            "reconstruction_shape": "424,H,W",
            "blocks": {
                "0": "bands 0:106",
                "45": "bands 106:212",
                "90": "bands 212:318",
                "135": "bands 318:424",
            },
            "recon_1ch_mean_all": "mean over all 424 channels",
            "recon_4ch_pol_mean_wavelength": "reshape to 4,106,H,W then mean over wavelength axis",
            "recon_106ch_wavelength_mean_pol": "reshape to 4,106,H,W then mean over polarization axis",
            "recon_424ch_full": "unchanged 424-channel cube",
        },
        "native_scatter_policy": "use first four channels from native 5-channel TIFFs",
        "crop_policy": crop_policy(dataset),
        "fold_summary": fold_summary,
    }
    write_json(output_root / "run_audit.json", audit)


def make_folds(records: list[ModalityRecord], n_splits: int, seed: int, use_predefined: bool) -> list[FoldSplit]:
    y = np.asarray([r.label for r in records], dtype=np.int64)
    if use_predefined:
        folds: list[FoldSplit] = []
        for fold_id in range(n_splits):
            test = [r for r in records if r.predefined_fold == fold_id]
            train_val = [r for r in records if r.predefined_fold != fold_id]
            train_val_y = np.asarray([r.label for r in train_val], dtype=np.int64)
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=100 + fold_id)
            train_idx, val_idx = next(splitter.split(np.arange(len(train_val)), train_val_y))
            folds.append(
                FoldSplit(
                    fold_id=fold_id,
                    train=[train_val[i] for i in train_idx],
                    val=[train_val[i] for i in val_idx],
                    test=test,
                )
            )
        return folds

    idx = np.arange(len(records))
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_id, (train_val_idx, test_idx) in enumerate(outer.split(idx, y)):
        train_val_y = y[train_val_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=100 + fold_id)
        train_rel, val_rel = next(splitter.split(train_val_idx, train_val_y))
        folds.append(
            FoldSplit(
                fold_id=fold_id,
                train=[records[i] for i in train_val_idx[train_rel]],
                val=[records[i] for i in train_val_idx[val_rel]],
                test=[records[i] for i in test_idx],
            )
        )
    return folds


def channel_stats(cube: np.ndarray, quantiles: bool = False, minmax: bool = False) -> np.ndarray:
    flat = cube.reshape(cube.shape[0], -1)
    parts = [flat.mean(axis=1), flat.std(axis=1)]
    if quantiles:
        parts.extend(np.quantile(flat, [0.25, 0.50, 0.75], axis=1))
    if minmax:
        parts.extend([flat.min(axis=1), flat.max(axis=1)])
    return np.concatenate(parts).astype(np.float32, copy=False)


def spatial_grid_stats(cube: np.ndarray, grid_size: int, quantiles: bool = False) -> np.ndarray:
    parts = [channel_stats(cube, quantiles=quantiles)]
    y_bins = np.array_split(np.arange(cube.shape[1]), grid_size)
    x_bins = np.array_split(np.arange(cube.shape[2]), grid_size)
    for ys in y_bins:
        for xs in x_bins:
            parts.append(channel_stats(cube[:, ys[:, None], xs], quantiles=quantiles))
    return np.concatenate(parts).astype(np.float32, copy=False)


def downsample_flat(cube: np.ndarray, grid_size: int) -> np.ndarray:
    y_bins = np.array_split(np.arange(cube.shape[1]), grid_size)
    x_bins = np.array_split(np.arange(cube.shape[2]), grid_size)
    cells = []
    for ys in y_bins:
        for xs in x_bins:
            cells.append(cube[:, ys[:, None], xs].mean(axis=(1, 2)))
    return np.stack(cells, axis=1).reshape(-1).astype(np.float32, copy=False)


def compute_feature_vector(cube: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "mean_std":
        return channel_stats(cube)
    if feature_mode == "mean_std_q":
        return channel_stats(cube, quantiles=True)
    if feature_mode == "mean_std_q_minmax":
        return channel_stats(cube, quantiles=True, minmax=True)
    if feature_mode == "spatial2_mean_std":
        return spatial_grid_stats(cube, 2)
    if feature_mode == "spatial3_mean_std":
        return spatial_grid_stats(cube, 3)
    if feature_mode == "spatial4_mean_std_q":
        return spatial_grid_stats(cube, 4, quantiles=True)
    if feature_mode == "downsample16_flat":
        return downsample_flat(cube, 16)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def augment_cube(cube: np.ndarray, rng: np.random.Generator, args: argparse.Namespace) -> np.ndarray:
    aug = np.array(cube, copy=True)
    k = int(rng.integers(0, 4))
    if k:
        aug = np.rot90(aug, k=k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        aug = aug[:, :, ::-1].copy()
    if rng.random() < 0.5:
        aug = aug[:, ::-1, :].copy()
    if args.augment_gain_range > 0:
        gain = rng.uniform(1.0 - args.augment_gain_range, 1.0 + args.augment_gain_range)
        aug *= np.float32(gain)
    if args.augment_channel_gain_range > 0:
        gains = rng.uniform(
            1.0 - args.augment_channel_gain_range,
            1.0 + args.augment_channel_gain_range,
            size=(aug.shape[0], 1, 1),
        ).astype(np.float32)
        aug *= gains
    if args.augment_offset_range > 0:
        aug += np.float32(rng.uniform(-args.augment_offset_range, args.augment_offset_range))
    if args.augment_noise_std > 0:
        aug += rng.normal(0.0, args.augment_noise_std, size=aug.shape).astype(np.float32)
    return np.clip(aug, 0.0, 1.0).astype(np.float32, copy=False)


def augmentation_config(args: argparse.Namespace) -> dict:
    return {
        "augment_train_copies": int(args.augment_train_copies),
        "augment_gain_range": float(args.augment_gain_range),
        "augment_channel_gain_range": float(args.augment_channel_gain_range),
        "augment_offset_range": float(args.augment_offset_range),
        "augment_noise_std": float(args.augment_noise_std),
        "spatial_transforms": "random rot90 plus random horizontal/vertical flips for augmented copies only",
    }


def build_feature_matrix(
    records: list[ModalityRecord],
    cube_cache: dict[str, np.ndarray],
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([compute_feature_vector(cube_cache[r.image_id], feature_mode) for r in records]),
        np.asarray([r.label for r in records], dtype=np.int64),
    )


def build_augmented_feature_matrix(
    records: list[ModalityRecord],
    cube_cache: dict[str, np.ndarray],
    feature_mode: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if args.augment_train_copies <= 0:
        return build_feature_matrix(records, cube_cache, feature_mode)
    rng = np.random.default_rng(seed)
    x_parts = []
    y_parts = []
    for record in records:
        cube = cube_cache[record.image_id]
        x_parts.append(compute_feature_vector(cube, feature_mode))
        y_parts.append(record.label)
        for _ in range(args.augment_train_copies):
            x_parts.append(compute_feature_vector(augment_cube(cube, rng, args), feature_mode))
            y_parts.append(record.label)
    return np.stack(x_parts), np.asarray(y_parts, dtype=np.int64)


def build_pixel_training_set(
    records: list[ModalityRecord],
    cube_cache: dict[str, np.ndarray],
    samples_per_image: int,
    seed: int,
    args: argparse.Namespace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts = []
    y_parts = []
    for record in records:
        cubes = [cube_cache[record.image_id]]
        if args is not None and args.augment_train_copies > 0:
            cubes.extend(augment_cube(cubes[0], rng, args) for _ in range(args.augment_train_copies))
        for cube in cubes:
            flat = cube.reshape(cube.shape[0], -1).T
            keep = rng.choice(len(flat), size=min(samples_per_image, len(flat)), replace=False)
            x_parts.append(flat[keep].astype(np.float32, copy=False))
            y_parts.append(np.full(len(keep), record.label, dtype=np.int64))
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "per_class_recall": {class_names[i]: float(recalls[i]) for i in range(len(class_names))},
    }


def fit_best_image_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, c_grid: list[float]):
    best = None
    best_model = None
    for c in c_grid:
        model = make_pipeline(StandardScaler(), LinearSVC(C=c, dual=False, max_iter=10000))
        model.fit(x_train, y_train)
        score = accuracy_score(y_val, model.predict(x_val))
        if best is None or score > best[1]:
            best = (c, score)
            best_model = model
    assert best is not None and best_model is not None
    return best_model, best


def pixel_majority_accuracy(model, records: list[ModalityRecord], cube_cache: dict[str, np.ndarray]) -> float:
    y_true = []
    y_pred = []
    for record in records:
        cube = cube_cache[record.image_id]
        flat = cube.reshape(cube.shape[0], -1).T
        pred = model.predict(flat)
        unique, counts = np.unique(pred, return_counts=True)
        y_true.append(record.label)
        y_pred.append(int(unique[int(np.argmax(counts))]))
    return float(accuracy_score(y_true, y_pred))


def fit_best_pixel_model(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    val_records: list[ModalityRecord],
    cube_cache: dict[str, np.ndarray],
    c_grid: list[float],
    fallback_c: float,
    selection: str,
):
    if selection == "image":
        model = make_pipeline(StandardScaler(), LinearSVC(C=fallback_c, dual=False, max_iter=10000))
        model.fit(pixel_x, pixel_y)
        return model, fallback_c, None
    best = None
    best_model = None
    for c in c_grid:
        model = make_pipeline(StandardScaler(), LinearSVC(C=c, dual=False, max_iter=10000))
        model.fit(pixel_x, pixel_y)
        score = pixel_majority_accuracy(model, val_records, cube_cache)
        if best is None or score > best[1]:
            best = (c, score)
            best_model = model
    assert best is not None and best_model is not None
    return best_model, float(best[0]), float(best[1])


def save_pixel_plot(output_dir: Path, record: ModalityRecord, cube: np.ndarray, pred_map: np.ndarray, class_names: list[str]) -> None:
    preview = cube.mean(axis=0)
    colors = ["#e4572e", "#4caf50", "#3f88c5", "#f0c808"]
    cmap = ListedColormap(colors[: len(class_names)])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"{record.image_id}\ntrue={class_names[record.label]}")
    axes[0].axis("off")
    axes[1].imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(pred_map, cmap=cmap, alpha=0.55, vmin=0, vmax=len(class_names) - 1)
    axes[1].set_title("Pixel predictions")
    axes[1].axis("off")
    fig.savefig(output_dir / f"{record.class_name}_{record.image_key}_pixel_predictions.png", dpi=180)
    plt.close(fig)


def summarize(run_rows: list[dict], output_root: Path, class_names: list[str]) -> None:
    rows = []
    df = pd.DataFrame(run_rows)
    for modality in MODALITIES:
        sub = df[df["modality"] == modality]
        if sub.empty:
            continue
        row = {
            "modality": modality,
            "folds_completed": int(len(sub)),
            "image_accuracy_mean": float(sub["test_image_accuracy"].mean()),
            "image_accuracy_std": float(sub["test_image_accuracy"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "image_macro_f1_mean": float(sub["test_image_macro_f1"].mean()),
            "image_macro_f1_std": float(sub["test_image_macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "pixel_majority_accuracy_mean": float(sub["test_pixel_majority_accuracy"].mean()),
            "pixel_majority_accuracy_std": float(sub["test_pixel_majority_accuracy"].std(ddof=1)) if len(sub) > 1 else 0.0,
        }
        for name in class_names:
            row[f"image_recall_{name}_mean"] = float(np.mean([x[name] for x in sub["test_image_per_class_recall"]]))
            row[f"pixel_recall_{name}_mean"] = float(np.mean([x[name] for x in sub["test_pixel_majority_per_class_recall"]]))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "modality_results.csv", index=False)
    write_json(output_root / "modality_results.json", {"rows": rows})
    if not summary.empty:
        plt.figure(figsize=(10, 4.8))
        x = np.arange(len(summary))
        plt.bar(x - 0.18, summary["image_accuracy_mean"], width=0.36, label="image")
        plt.bar(x + 0.18, summary["pixel_majority_accuracy_mean"], width=0.36, label="pixel majority")
        plt.xticks(x, summary["modality"], rotation=30, ha="right")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("Modality accuracy")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / "modality_accuracy_plot.png", dpi=200)
        plt.savefig(output_root / "modality_accuracy_plot.pdf")
        plt.close()


def run_modality_fold(
    dataset: str,
    modality: str,
    fold: FoldSplit,
    records: list[ModalityRecord],
    class_names: list[str],
    c_grid: list[float],
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    fold_dir = ensure_dir(output_root / modality / f"fold_{fold.fold_id}")
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    cube_cache = {record.image_id: load_modality_cube(record, modality, dataset) for record in records}
    aug_seed = stable_seed(args.seed, dataset, modality, fold.fold_id, "train_augmentation")
    x_train, y_train = build_augmented_feature_matrix(fold.train, cube_cache, args.feature_mode, args, aug_seed)
    x_val, y_val = build_feature_matrix(fold.val, cube_cache, args.feature_mode)
    x_test, y_test = build_feature_matrix(fold.test, cube_cache, args.feature_mode)

    image_model, (best_c, best_val_acc) = fit_best_image_model(x_train, y_train, x_val, y_val, c_grid)
    test_pred = image_model.predict(x_test)
    image_metrics = metric_bundle(y_test, test_pred, class_names)

    rows = []
    for record, pred, feat in zip(fold.test, test_pred, x_test):
        decision = image_model.decision_function(feat[None, :])
        decision = np.asarray(decision).reshape(-1)
        if len(class_names) == 2 and decision.shape[0] == 1:
            decision = np.asarray([-decision[0], decision[0]], dtype=np.float32)
        rows.append(
            {
                "image_id": record.image_id,
                "image_key": record.image_key,
                "path": str(record.recon_path if modality != "native_scatter_4ch" else record.native_path),
                "true_label": int(record.label),
                "pred_label": int(pred),
                "correct": int(pred == record.label),
                **{f"score_{class_names[i]}": float(decision[i]) for i in range(len(class_names))},
            }
        )
    pd.DataFrame(rows).to_csv(fold_dir / "test_image_predictions.csv", index=False)
    pd.DataFrame(image_metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(
        fold_dir / "confusion_matrix_image.csv"
    )

    pixel_x, pixel_y = build_pixel_training_set(
        fold.train,
        cube_cache,
        args.pixel_train_samples_per_image,
        stable_seed(args.seed, dataset, modality, fold.fold_id, "pixel_augmentation"),
        args,
    )
    pixel_model, pixel_best_c, pixel_best_val_acc = fit_best_pixel_model(
        pixel_x,
        pixel_y,
        fold.val,
        cube_cache,
        c_grid,
        float(best_c),
        args.pixel_c_selection,
    )

    pixel_rows = []
    pixel_majority_true = []
    pixel_majority_pred = []
    pixel_dir = ensure_dir(fold_dir / "pixel_plots")
    for image_index, record in enumerate(fold.test):
        cube = cube_cache[record.image_id]
        flat = cube.reshape(cube.shape[0], -1).T
        pred_map = pixel_model.predict(flat).reshape(cube.shape[1], cube.shape[2])
        if image_index < args.pixel_plot_limit:
            save_pixel_plot(pixel_dir, record, cube, pred_map, class_names)
        unique, counts = np.unique(pred_map, return_counts=True)
        count_map = {class_names[int(k)]: int(v) for k, v in zip(unique, counts)}
        majority = int(unique[int(np.argmax(counts))])
        pixel_majority_true.append(record.label)
        pixel_majority_pred.append(majority)
        pixel_rows.append(
            {
                "image_id": record.image_id,
                "image_key": record.image_key,
                "true_label": int(record.label),
                "majority_pred_label": majority,
                "majority_correct": int(majority == record.label),
                **count_map,
            }
        )
    pd.DataFrame(pixel_rows).fillna(0).to_csv(fold_dir / "pixel_prediction_counts.csv", index=False)
    pixel_majority_metrics = metric_bundle(np.asarray(pixel_majority_true), np.asarray(pixel_majority_pred), class_names)

    with (fold_dir / "best_image_model.pkl").open("wb") as f:
        pickle.dump({"model": image_model, "best_c": best_c, "feature_mode": args.feature_mode}, f)
    with (fold_dir / "pixel_model.pkl").open("wb") as f:
        pickle.dump({"model": pixel_model, "best_c": pixel_best_c, "pixel_c_selection": args.pixel_c_selection}, f)

    example_cube = cube_cache[records[0].image_id]
    metrics = {
        "dataset": dataset,
        "modality": modality,
        "fold": int(fold.fold_id),
        "cube_shape": list(example_cube.shape),
        "feature_mode": args.feature_mode,
        "augmentation": augmentation_config(args),
        "image_model": "linear_svm",
        "pixel_model": "linear_svm",
        "best_c": float(best_c),
        "best_val_image_accuracy": float(best_val_acc),
        "pixel_best_c": float(pixel_best_c),
        "pixel_best_val_majority_accuracy": pixel_best_val_acc,
        "pixel_c_selection": args.pixel_c_selection,
        "test_image_accuracy": image_metrics["accuracy"],
        "test_image_macro_f1": image_metrics["macro_f1"],
        "test_image_per_class_recall": image_metrics["per_class_recall"],
        "test_pixel_majority_accuracy": pixel_majority_metrics["accuracy"],
        "test_pixel_majority_macro_f1": pixel_majority_metrics["macro_f1"],
        "test_pixel_majority_per_class_recall": pixel_majority_metrics["per_class_recall"],
        "pixel_train_samples_per_image": int(args.pixel_train_samples_per_image),
        "train_count": len(fold.train),
        "val_count": len(fold.val),
        "test_count": len(fold.test),
    }
    write_json(metrics_path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    modalities = parse_csv_strings(args.modalities)
    if args.smoke:
        modalities = ["recon_4ch_pol_mean_wavelength"]
        args.fold_filter = "0"
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")
    fold_filter = {int(x) for x in parse_csv_strings(args.fold_filter)}
    c_grid = parse_csv_floats(args.svm_c_grid)
    output_root = ensure_dir(args.output_root)
    append_command_log(output_root / "COMMANDS_RUN.md", [sys.executable, *sys.argv])

    records, class_names, manifest = discover_records(args.dataset, args)
    use_predefined = args.dataset == "camo"
    folds = make_folds(records, args.folds, args.seed, use_predefined=use_predefined)
    write_json(output_root / "dataset_manifest.json", manifest)
    write_run_audit(output_root, args.dataset, records, class_names, folds, modalities)

    run_rows: list[dict] = []
    for modality in modalities:
        for fold in folds:
            if fold.fold_id not in fold_filter:
                continue
            print(f"[run] dataset={args.dataset} modality={modality} fold={fold.fold_id}", flush=True)
            run_rows.append(run_modality_fold(args.dataset, modality, fold, records, class_names, c_grid, args, output_root))
            summarize(run_rows, output_root, class_names)
    pd.DataFrame(run_rows).to_csv(output_root / "results.csv", index=False)
    write_json(output_root / "results.json", {"rows": run_rows})
    summarize(run_rows, output_root, class_names)
    print(f"[done] wrote outputs under {output_root}", flush=True)


if __name__ == "__main__":
    main()
