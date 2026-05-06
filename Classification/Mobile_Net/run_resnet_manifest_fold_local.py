import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms

from build_canonical_manifest import DEFAULT_OUTPUT as DEFAULT_CANONICAL_MANIFEST
from manifest_local_runner_utils import (
    SUPPORTED_TASKS,
    copy_tree_contents,
    ensure_manifest,
    is_scratch_path,
    materialize_fold_dataset,
    read_csv_rows,
    save_json,
    summarize_materialized_channels,
)
from resnet_regime_model import CompactRegimeResNet
from utils.dataset import ImageFolder
from utils.transforms import (
    MultiChannelNormalize,
    MultiChannelRandomHorizontalFlip,
    MultiChannelRandomVerticalFlip,
    MultiChannelResize,
    ToTensorIfNeeded,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CLASSIFICATION_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CLASSIFICATION_ROOT.parent
DEFAULT_PAIRED_MANIFEST = REPO_ROOT / "paired_textile_camo_manifest_2026-04-09.csv"
DEFAULT_SELECTIONS = REPO_ROOT / "double_pair_manual_selections_2026-04-09.csv"
DEFAULT_MANIFEST = DEFAULT_CANONICAL_MANIFEST

RESNET_REGIMES = (
    "raw_pol",
    "recon_spatial",
    "recon_polarimetric",
    "recon_spectral",
    "recon_specpol",
)
STOKES_REGIMES = (
    "raw_stokes",
    "recon_stokes_polarimetric",
    "recon_stokes_specpol",
)
ALL_RESNET_REGIMES = RESNET_REGIMES + STOKES_REGIMES

REGIME_DISPLAY_NAMES = {
    "raw_pol": "4-channel scatterogram",
    "recon_spatial": "1-channel spectral-polarimetrically averaged reconstruction",
    "recon_polarimetric": "4-channel spectrally averaged reconstruction",
    "recon_spectral": "106-channel polarimetrically averaged reconstruction",
    "recon_specpol": "424-channel spectro-polarimetric reconstruction",
    "raw_stokes": "3-channel raw Stokes scatterogram",
    "recon_stokes_polarimetric": "3-channel spectrally averaged Stokes reconstruction",
    "recon_stokes_specpol": "318-channel spectro-Stokes reconstruction",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compact ResNet fold training against canonical paired textile/camo manifest data."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Scratch-backed dataset root used to remap Windows manifest paths.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--task", choices=list(SUPPORTED_TASKS) + ["all"], default="all")
    parser.add_argument("--regime", choices=list(ALL_RESNET_REGIMES) + ["all"], default="all")
    parser.add_argument(
        "--regime-set",
        choices=["original", "stokes", "all"],
        default="original",
        help="Which regimes to run when --regime all is selected. Default preserves the original five modalities.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--scratch-work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="Fixed batch size used for every modality.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--model-variant",
        choices=["baseline", "projection_stem", "factorized_latent", "hybrid_spectral"],
        default="baseline",
        help="Use projection_stem, factorized_latent, or hybrid_spectral before spatial ResNet blocks.",
    )
    parser.add_argument(
        "--projection-channels",
        type=int,
        default=32,
        help="Channel count after the 1x1 learned projection stem.",
    )
    parser.add_argument(
        "--channel-attention",
        action="store_true",
        help="Add squeeze-excitation attention after the projection or factorized latent stem.",
    )
    parser.add_argument(
        "--spectral-bands",
        type=int,
        default=106,
        help="Number of wavelength channels per polarization/Stokes group for factorized_latent.",
    )
    parser.add_argument(
        "--latent-channels-per-group",
        type=int,
        default=16,
        help="Latent channels learned per spectral group for factorized_latent.",
    )
    parser.add_argument(
        "--hybrid-fusion-channels",
        type=int,
        default=96,
        help="Output channels after grouped/full projection fusion for hybrid_spectral.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_scratch_paths(args):
    scratch_work_dir = args.scratch_work_dir.resolve()
    dataset_root = Path(args.dataset_root).resolve()

    if not is_scratch_path(scratch_work_dir):
        raise ValueError(f"--scratch-work-dir must point at scratch storage. Got: {scratch_work_dir}")
    if not is_scratch_path(dataset_root):
        raise ValueError(f"--dataset-root must point at scratch storage. Got: {dataset_root}")

    repo_root = SCRIPT_DIR.resolve()
    if repo_root == scratch_work_dir or repo_root in scratch_work_dir.parents:
        raise ValueError("--scratch-work-dir cannot live inside the repo checkout.")
    if repo_root == dataset_root or repo_root in dataset_root.parents:
        raise ValueError("--dataset-root cannot live inside the repo checkout.")

    args.scratch_work_dir = scratch_work_dir
    args.dataset_root = str(dataset_root)
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()


def candidate_manifest_roots():
    roots = [
        REPO_ROOT,
        PROJECT_ROOT,
        CLASSIFICATION_ROOT,
        SCRIPT_DIR,
        Path.cwd(),
    ]
    unique_roots = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(root)
    return unique_roots


def resolve_existing_csv(path: Path, filename: str):
    if path.exists():
        return path
    for root in candidate_manifest_roots():
        candidate = root / filename
        if candidate.exists():
            return candidate
    return path


def resolve_manifest_inputs(args):
    args.manifest = resolve_existing_csv(args.manifest, "canonical_paired_manifest_2026-04-09.csv")
    args.paired_manifest = resolve_existing_csv(args.paired_manifest, "paired_textile_camo_manifest_2026-04-09.csv")
    args.selections = resolve_existing_csv(args.selections, "double_pair_manual_selections_2026-04-09.csv")

    if args.manifest.exists():
        print(f"Using canonical manifest: {args.manifest}")
        return

    missing = []
    if not args.paired_manifest.exists():
        missing.append(("paired manifest", args.paired_manifest))
    if not args.selections.exists():
        missing.append(("manual selections", args.selections))
    if not missing:
        print(f"Canonical manifest not found; will build it at: {args.manifest}")
        print(f"Using paired manifest: {args.paired_manifest}")
        print(f"Using manual selections: {args.selections}")
        return

    searched = "\n  ".join(str(root / "canonical_paired_manifest_2026-04-09.csv") for root in candidate_manifest_roots())
    missing_text = "\n".join(f"- Missing {label}: {path}" for label, path in missing)
    raise FileNotFoundError(
        "Could not find an existing canonical manifest, and cannot build one because required source CSVs are missing.\n"
        f"{missing_text}\n\n"
        "Pass --manifest /path/to/canonical_paired_manifest_2026-04-09.csv, or place the CSV in one of these searched locations:\n"
        f"  {searched}"
    )


def append_summary(path: Path, rows):
    fieldnames = [
        "task",
        "regime",
        "fold",
        "best_test_acc",
        "final_test_acc",
        "final_train_acc",
        "best_macro_f1",
        "final_macro_f1",
        "time_minutes",
        "train_dir",
        "test_dir",
    ]
    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_promote_results(run_root: Path, output_dir: Path | None, task: str, regime: str, fold: int):
    if output_dir is None:
        return None
    target = output_dir / task / regime / f"fold_{fold}"
    copy_tree_contents(run_root, target)
    return target


def collect_example_tiffs(prepared_root: Path, limit: int = 3):
    return sorted(prepared_root.rglob("*.tif"))[:limit]


def make_transforms(image_size: int, train: bool):
    transform_list = [
        ToTensorIfNeeded(),
        MultiChannelResize((image_size, image_size)),
    ]
    if train:
        transform_list.extend(
            [
                MultiChannelRandomHorizontalFlip(p=0.5),
                MultiChannelRandomVerticalFlip(p=0.5),
            ]
        )
    transform_list.append(MultiChannelNormalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)


def detect_channels(dataset: ImageFolder):
    if len(dataset) == 0:
        raise ValueError("Cannot detect channels from an empty dataset.")
    image, _ = dataset[0]
    return int(image.shape[0])


def make_sampler(dataset: ImageFolder):
    counts = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def make_loaders(train_dir: str, test_dir: str, args):
    temp_dataset = ImageFolder(
        train_dir,
        transform=transforms.Compose([ToTensorIfNeeded(), MultiChannelResize((args.image_size, args.image_size))]),
        num_channels=None,
        normalize_per_channel=False,
    )
    in_channels = detect_channels(temp_dataset)

    train_dataset = ImageFolder(
        train_dir,
        transform=make_transforms(args.image_size, train=True),
        num_channels=in_channels,
        normalize_per_channel=False,
    )
    test_dataset = ImageFolder(
        test_dir,
        transform=make_transforms(args.image_size, train=False),
        num_channels=in_channels,
        normalize_per_channel=False,
    )
    train_eval_dataset = ImageFolder(
        train_dir,
        transform=make_transforms(args.image_size, train=False),
        num_channels=in_channels,
        normalize_per_channel=False,
    )

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=make_sampler(train_dataset),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_dataset, test_dataset, train_loader, train_eval_loader, test_loader, in_channels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size

    return total_loss / max(total_items, 1), 100.0 * total_correct / max(total_items, 1)


@torch.inference_mode()
def evaluate(model, loader, label_names, device):
    model.eval()
    probs = []
    preds = []
    trues = []
    sample_paths = []

    sample_offset = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)

        batch_size = labels.size(0)
        batch_samples = loader.dataset.samples[sample_offset : sample_offset + batch_size]
        sample_paths.extend(path for path, _ in batch_samples)
        sample_offset += batch_size

        probs.append(prob)
        preds.append(pred)
        trues.append(labels.numpy())

    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
    }
    if len(label_names) == 2:
        metrics["auroc"] = float(roc_auc_score(trues, probs[:, 1]))
        metrics["binary_f1"] = float(f1_score(trues, preds))
    return metrics, sample_paths, trues, preds, probs


def save_predictions(path: Path, sample_paths, label_names, trues, preds, probs):
    fieldnames = ["sample_path", "true_label", "pred_label"] + [f"prob_{name}" for name in label_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample_path in enumerate(sample_paths):
            row = {
                "sample_path": sample_path,
                "true_label": label_names[int(trues[idx])],
                "pred_label": label_names[int(preds[idx])],
            }
            for class_idx, class_name in enumerate(label_names):
                row[f"prob_{class_name}"] = float(probs[idx, class_idx])
            writer.writerow(row)


def save_checkpoint(path: Path, model, label_names, args, task: str, regime: str, fold: int, report, epoch: int):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": label_names,
            "task": task,
            "regime": regime,
            "regime_display_name": REGIME_DISPLAY_NAMES[regime],
            "fold": fold,
            "epoch": epoch,
            "model_name": "CompactRegimeResNet",
            "parameter_report": report.__dict__,
            "train_args": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
            "base_channels": args.base_channels,
            "model_variant": args.model_variant,
            "projection_channels": args.projection_channels,
            "channel_attention": args.channel_attention,
            "spectral_bands": args.spectral_bands,
            "latent_channels_per_group": args.latent_channels_per_group,
            "hybrid_fusion_channels": args.hybrid_fusion_channels,
            "dropout": args.dropout,
            "image_size": args.image_size,
            },
        },
        path,
    )


def run_single_fold(args, train_dir: str, test_dir: str, task: str, regime: str, fold: int, run_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_dataset, test_dataset, train_loader, train_eval_loader, test_loader, in_channels = make_loaders(
        train_dir, test_dir, args
    )
    label_names = train_dataset.classes
    model = CompactRegimeResNet(
        in_channels=in_channels,
        num_classes=len(label_names),
        base_channels=args.base_channels,
        dropout=args.dropout,
        projection_channels=args.projection_channels
        if args.model_variant in ("projection_stem", "factorized_latent", "hybrid_spectral")
        else None,
        use_channel_attention=args.channel_attention,
        factorized_latent=args.model_variant == "factorized_latent",
        hybrid_spectral=args.model_variant == "hybrid_spectral",
        spectral_bands=args.spectral_bands,
        latent_channels_per_group=args.latent_channels_per_group,
        hybrid_fusion_channels=args.hybrid_fusion_channels,
    ).to(device)
    report = model.parameter_report()
    print(
        f"[{task}][{regime}][fold={fold}] classes={label_names} input_channels={in_channels} "
        f"batch_size={args.batch_size} model_variant={args.model_variant} params={report.total}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_macro_f1 = -1.0
    best_test_acc = 0.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    last_train_acc = 0.0
    last_train_loss = None
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics, _, _, _, _ = evaluate(model, test_loader, label_names, device)
        scheduler.step()

        last_train_loss = train_loss
        last_train_acc = train_acc
        test_acc_pct = test_metrics["accuracy"] * 100.0
        macro_f1 = test_metrics["macro_f1"]
        best_test_acc = max(best_test_acc, test_acc_pct)

        print(
            f"[{task}][{regime}][fold={fold}] epoch={epoch:03d} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc_pct:.2f}% macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1 + args.early_stop_min_delta:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            print(
                f"[{task}][{regime}][fold={fold}] early_stop epoch={epoch} "
                f"best_epoch={best_epoch} best_macro_f1={best_macro_f1:.4f}"
            )
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    final_test_metrics, sample_paths, trues, preds, probs = evaluate(model, test_loader, label_names, device)
    final_train_metrics, _, _, _, _ = evaluate(model, train_eval_loader, label_names, device)

    checkpoint_path = run_root / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, label_names, args, task, regime, fold, report, best_epoch)
    save_predictions(run_root / "predictions.csv", sample_paths, label_names, trues, preds, probs)

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_metrics["accuracy"] * 100.0,
        "final_train_acc": final_train_metrics["accuracy"] * 100.0,
        "best_macro_f1": best_macro_f1,
        "final_macro_f1": final_test_metrics["macro_f1"],
        "final_metrics": final_test_metrics,
        "last_train_loss": last_train_loss,
        "last_train_acc": last_train_acc,
        "epochs_trained": epochs_trained,
        "stopped_early": epochs_trained < args.epochs,
        "checkpoint": str(checkpoint_path),
        "parameter_report": report.__dict__,
        "input_channels": in_channels,
        "label_names": label_names,
    }


def train_args_payload(args):
    return {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "base_channels": args.base_channels,
        "model_variant": args.model_variant,
        "projection_channels": args.projection_channels,
        "channel_attention": args.channel_attention,
        "spectral_bands": args.spectral_bands,
        "latent_channels_per_group": args.latent_channels_per_group,
        "hybrid_fusion_channels": args.hybrid_fusion_channels,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "workers": args.workers,
    }


def main():
    args = parse_args()
    validate_scratch_paths(args)
    resolve_manifest_inputs(args)
    seed_everything(args.seed)

    manifest_path = ensure_manifest(args.manifest, args.paired_manifest, args.selections, args.seed)
    all_rows = read_csv_rows(manifest_path)

    tasks = list(SUPPORTED_TASKS) if args.task == "all" else [args.task]
    if args.regime == "all":
        if args.regime_set == "stokes":
            regimes = list(STOKES_REGIMES)
        elif args.regime_set == "all":
            regimes = list(ALL_RESNET_REGIMES)
        else:
            regimes = list(RESNET_REGIMES)
    else:
        regimes = [args.regime]
    summary_rows = []

    for task in tasks:
        task_rows = [row for row in all_rows if row["task"] == task]
        available_folds = sorted({int(row["fold"]) for row in task_rows})
        if args.fold not in available_folds:
            raise ValueError(f"Requested fold {args.fold} not available for task {task}. Available folds: {available_folds}")

        for regime in regimes:
            run_root = args.scratch_work_dir / "runs" / task / regime / f"fold_{args.fold}"
            run_root.mkdir(parents=True, exist_ok=True)
            metrics_path = run_root / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                print(f"Skipping completed run for task={task} regime={regime} fold={args.fold} at {metrics_path}")
                continue

            print(f"Preparing task={task} regime={regime} fold={args.fold}")
            materialized = materialize_fold_dataset(
                rows=all_rows,
                task=task,
                regime=regime,
                fold=args.fold,
                dataset_root=args.dataset_root,
                scratch_work_dir=args.scratch_work_dir,
            )
            prepared_root = Path(materialized["prepared_root"])
            if materialized.get("reused"):
                print(f"Reusing prepared dataset at {prepared_root}")

            save_json(
                run_root / "materialization_summary.json",
                {
                    "task": task,
                    "regime": regime,
                    "regime_display_name": REGIME_DISPLAY_NAMES[regime],
                    "fold": args.fold,
                    "prepared_root": str(prepared_root),
                    "train_dir": str(materialized["train_dir"]),
                    "test_dir": str(materialized["test_dir"]),
                    "sample_count_train": len(materialized["train_rows"]),
                    "sample_count_test": len(materialized["test_rows"]),
                    "reused_prepared_dataset": bool(materialized.get("reused")),
                    "example_channel_summary": summarize_materialized_channels(collect_example_tiffs(prepared_root)),
                },
            )

            print(f"Training task={task} regime={regime} fold={args.fold}")
            start_time = time.time()
            cwd_before = Path.cwd()
            try:
                os.chdir(run_root)
                result = run_single_fold(
                    args=args,
                    train_dir=str(materialized["train_dir"]),
                    test_dir=str(materialized["test_dir"]),
                    task=task,
                    regime=regime,
                    fold=args.fold,
                    run_root=run_root,
                )
            finally:
                os.chdir(cwd_before)
            elapsed_minutes = (time.time() - start_time) / 60.0

            result_payload = {
                "task": task,
                "regime": regime,
                "regime_display_name": REGIME_DISPLAY_NAMES[regime],
                "fold": args.fold,
                "best_test_acc": result["best_test_acc"],
                "final_test_acc": result["final_test_acc"],
                "final_train_acc": result["final_train_acc"],
                "best_macro_f1": result["best_macro_f1"],
                "final_macro_f1": result["final_macro_f1"],
                "time_minutes": elapsed_minutes,
                "train_args": train_args_payload(args),
                "train_dir": str(materialized["train_dir"]),
                "test_dir": str(materialized["test_dir"]),
                "checkpoint": result["checkpoint"],
                "input_channels": result["input_channels"],
                "label_names": result["label_names"],
                "parameter_report": result["parameter_report"],
                "final_metrics": result["final_metrics"],
                "last_train_loss": result["last_train_loss"],
                "last_train_acc": result["last_train_acc"],
                "epochs_trained": result["epochs_trained"],
                "stopped_early": result["stopped_early"],
            }
            save_json(metrics_path, result_payload)

            promoted_to = maybe_promote_results(run_root, args.output_dir, task, regime, args.fold)
            if promoted_to is not None:
                print(f"Promoted results to {promoted_to}")

            summary_rows.append(
                {
                    "task": task,
                    "regime": regime,
                    "fold": args.fold,
                    "best_test_acc": result["best_test_acc"],
                    "final_test_acc": result["final_test_acc"],
                    "final_train_acc": result["final_train_acc"],
                    "best_macro_f1": result["best_macro_f1"],
                    "final_macro_f1": result["final_macro_f1"],
                    "time_minutes": elapsed_minutes,
                    "train_dir": str(materialized["train_dir"]),
                    "test_dir": str(materialized["test_dir"]),
                }
            )

            if not args.keep_temp:
                shutil.rmtree(prepared_root, ignore_errors=True)

    append_summary(args.scratch_work_dir / "runs" / "summary_resnet_manifest_runner.csv", summary_rows)
    print(f"Wrote scratch summary to {args.scratch_work_dir / 'runs' / 'summary_resnet_manifest_runner.csv'}")

    if args.output_dir is not None:
        append_summary(args.output_dir / "summary_resnet_manifest_runner.csv", summary_rows)
        print(f"Wrote promoted summary to {args.output_dir / 'summary_resnet_manifest_runner.csv'}")


if __name__ == "__main__":
    main()
