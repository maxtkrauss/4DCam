import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from build_canonical_manifest import (
    DEFAULT_OUTPUT as DEFAULT_CANONICAL_MANIFEST,
    build_canonical_manifest,
)
from regime_dataset import REGIMES, PairedRegimeDataset, read_manifest, summarize_dataset
from regime_model import RegimeMobileNetV3Large


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PAIRED_MANIFEST = REPO_ROOT / "paired_textile_camo_manifest_2026-04-09.csv"
DEFAULT_SELECTIONS = REPO_ROOT / "double_pair_manual_selections_2026-04-09.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "regime_results"
DEFAULT_REGIMES = [regime for regime in REGIMES.keys() if regime != "recon_spatial"]
DEFAULT_TASKS = ["textile_3way", "camo_binary"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNetV3-Large experiments across textile/camo regimes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset root used to remap Windows manifest paths on other systems.")
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--regime", choices=DEFAULT_REGIMES + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop training when train loss fails to improve by min delta for this many epochs. Set <0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum train loss improvement required to reset early stopping patience.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--folds", type=str, default="all", help="Comma-separated fold list or 'all'")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_manifest(args):
    if args.manifest.exists():
        return args.manifest

    build_canonical_manifest(
        paired_manifest_path=args.paired_manifest,
        selections_path=args.selections,
        output_path=args.manifest,
        n_splits=5,
        seed=args.seed,
    )
    return args.manifest


def filter_rows(rows, task):
    return [row for row in rows if row["task"] == task]


def folds_for_task(rows, fold_arg):
    available = sorted({int(row["fold"]) for row in rows})
    if fold_arg == "all":
        return available
    requested = [int(part) for part in fold_arg.split(",") if part.strip()]
    return [fold for fold in requested if fold in available]


def make_weighted_sampler(dataset: PairedRegimeDataset):
    counts = {}
    for row in dataset.rows:
        counts[row["class_label"]] = counts.get(row["class_label"], 0) + 1

    weights = []
    for row in dataset.rows:
        weights.append(1.0 / counts[row["class_label"]])

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def log_loader_summary(dataset: PairedRegimeDataset, task: str, regime: str, split_name: str):
    summary = summarize_dataset(dataset)
    print(
        f"[{task}][{regime}][{split_name}] samples={summary['samples']} "
        f"shape=({summary['channels']}, {summary['height']}, {summary['width']}) "
        f"min={summary['min']} max={summary['max']} classes={summary['classes']}"
    )


def log_first_batch_stats(loader, task: str, regime: str, split_name: str):
    batch = next(iter(loader), None)
    if batch is None:
        print(f"[{task}][{regime}][{split_name}] no batches available")
        return

    images, labels, _ = batch
    print(
        f"[{task}][{regime}][{split_name}] first_batch="
        f"batch_size={int(images.shape[0])} channels={int(images.shape[1])} "
        f"height={int(images.shape[2])} width={int(images.shape[3])} "
        f"min={float(images.min().item()):.6f} max={float(images.max().item()):.6f} "
        f"labels={sorted(set(labels.tolist()))}"
    )


def create_loaders(rows, task, regime, fold, batch_size, num_workers, image_size, dataset_root):
    train_rows = [row for row in rows if int(row["fold"]) != fold]
    eval_rows = [row for row in rows if int(row["fold"]) == fold]

    train_dataset = PairedRegimeDataset(
        train_rows, task=task, regime=regime, image_size=image_size, train=True, dataset_root=dataset_root
    )
    eval_dataset = PairedRegimeDataset(
        eval_rows, task=task, regime=regime, image_size=image_size, train=False, dataset_root=dataset_root
    )

    log_loader_summary(train_dataset, task, regime, "train")
    log_loader_summary(eval_dataset, task, regime, "eval")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_dataset),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    log_first_batch_stats(train_loader, task, regime, "train")
    log_first_batch_stats(eval_loader, task, regime, "eval")
    return train_dataset, eval_dataset, train_loader, eval_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.inference_mode()
def evaluate(model, loader, label_names, device):
    model.eval()
    probs = []
    preds = []
    trues = []
    sample_ids = []

    for images, labels, batch_ids in loader:
        images = images.to(device)
        logits = model(images)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)

        probs.append(prob)
        preds.append(pred)
        trues.append(labels.numpy())
        sample_ids.extend(batch_ids)

    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
    }

    if len(label_names) == 2:
        metrics["auroc"] = float(roc_auc_score(trues, probs[:, 1]))
        metrics["binary_f1"] = float(f1_score(trues, preds))

    metrics["confusion_matrix"] = confusion_matrix(trues, preds).tolist()
    return metrics, sample_ids, trues, preds, probs


def save_predictions(path: Path, sample_ids, label_names, trues, preds, probs):
    fieldnames = ["sample_id", "true_label", "pred_label"] + [f"prob_{name}" for name in label_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "true_label": label_names[int(trues[idx])],
                "pred_label": label_names[int(preds[idx])],
            }
            for class_idx, class_name in enumerate(label_names):
                row[f"prob_{class_name}"] = float(probs[idx, class_idx])
            writer.writerow(row)


def run_single_fold(args, rows, task, regime, fold, output_dir, device):
    train_dataset, eval_dataset, train_loader, eval_loader = create_loaders(
        rows=rows,
        task=task,
        regime=regime,
        fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        dataset_root=args.dataset_root,
    )

    in_channels = summarize_dataset(train_dataset)["channels"]
    num_classes = len(train_dataset.label_names)
    model = RegimeMobileNetV3Large(in_channels=in_channels, num_classes=num_classes, dropout=0.2).to(device)
    report = model.parameter_report()
    print(
        f"[{task}][{regime}][fold={fold}] params total={report.total} "
        f"adapter={report.adapter} body={report.body} head={report.head}"
    )

    if args.smoke_test:
        metrics, sample_ids, trues, preds, probs = evaluate(model, eval_loader, eval_dataset.label_names, device)
        metrics["train_loss"] = None
        metrics["best_train_loss"] = None
        metrics["epochs_trained"] = 0
        metrics["stopped_early"] = False
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

        last_loss = None
        best_loss = None
        best_state_dict = None
        best_epoch = 0
        stale_epochs = 0
        epochs_trained = 0
        for epoch in range(args.epochs):
            last_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            epochs_trained = epoch + 1
            print(f"[{task}][{regime}][fold={fold}] epoch={epoch + 1} train_loss={last_loss:.6f}")

            improved = best_loss is None or (best_loss - last_loss) > args.early_stop_min_delta
            if improved:
                best_loss = last_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1

            if args.early_stop_patience >= 0 and stale_epochs >= args.early_stop_patience:
                print(
                    f"[{task}][{regime}][fold={fold}] early_stop "
                    f"epoch={epoch + 1} best_epoch={best_epoch} "
                    f"best_train_loss={best_loss:.6f} patience={args.early_stop_patience} "
                    f"min_delta={args.early_stop_min_delta}"
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        metrics, sample_ids, trues, preds, probs = evaluate(model, eval_loader, eval_dataset.label_names, device)
        metrics["train_loss"] = last_loss
        metrics["best_train_loss"] = best_loss
        metrics["epochs_trained"] = epochs_trained
        metrics["stopped_early"] = epochs_trained < args.epochs

    fold_dir = output_dir / task / regime / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": train_dataset.label_names,
            "regime": regime,
            "task": task,
            "fold": fold,
            "parameter_report": report.__dict__,
        },
        fold_dir / "checkpoint.pt",
    )

    save_predictions(fold_dir / "predictions.csv", sample_ids, eval_dataset.label_names, trues, preds, probs)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def append_summary(summary_path: Path, rows):
    file_exists = summary_path.exists()
    fieldnames = [
        "task",
        "regime",
        "fold",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "binary_f1",
        "train_loss",
        "best_train_loss",
        "epochs_trained",
        "stopped_early",
    ]
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = ensure_manifest(args)
    all_rows = read_manifest(manifest_path)

    tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
    regimes = DEFAULT_REGIMES if args.regime == "all" else [args.regime]

    summary_rows = []
    for task in tasks:
        task_rows = filter_rows(all_rows, task)
        folds = folds_for_task(task_rows, args.folds)

        for regime in regimes:
            print(f"Running task={task} regime={regime} folds={folds}")
            for fold in folds:
                metrics = run_single_fold(
                    args=args,
                    rows=task_rows,
                    task=task,
                    regime=regime,
                    fold=fold,
                    output_dir=args.output_dir,
                    device=device,
                )
                summary_rows.append(
                    {
                        "task": task,
                        "regime": regime,
                        "fold": fold,
                        "accuracy": metrics.get("accuracy"),
                        "macro_f1": metrics.get("macro_f1"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "auroc": metrics.get("auroc"),
                        "binary_f1": metrics.get("binary_f1"),
                        "train_loss": metrics.get("train_loss"),
                        "best_train_loss": metrics.get("best_train_loss"),
                        "epochs_trained": metrics.get("epochs_trained"),
                        "stopped_early": metrics.get("stopped_early"),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    append_summary(args.output_dir / "summary_comparison.csv", summary_rows)
    print(f"Wrote summary to {args.output_dir / 'summary_comparison.csv'}")


if __name__ == "__main__":
    main()
