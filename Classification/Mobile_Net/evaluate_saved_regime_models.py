import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from build_canonical_manifest import (
    DEFAULT_OUTPUT as DEFAULT_CANONICAL_MANIFEST,
    build_canonical_manifest,
)
from regime_dataset import REGIMES, PairedRegimeDataset, read_manifest, summarize_dataset
from regime_model import RegimeMobileNetV3Large
from run_regime_experiments import (
    DEFAULT_PAIRED_MANIFEST,
    DEFAULT_SELECTIONS,
    DEFAULT_TASKS,
    evaluate,
    filter_rows,
    folds_for_task,
    save_predictions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "regime_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reload saved MobileNet regime checkpoints and evaluate them on their held-out fold."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--write-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional dataset root used to remap Windows manifest paths on other systems.",
    )
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--regime", choices=[r for r in REGIMES.keys() if r != "recon_spatial"] + ["all"], default="all")
    parser.add_argument("--folds", type=str, default="all", help="Comma-separated fold list or 'all'")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def ensure_manifest(args):
    if args.manifest.exists():
        return args.manifest

    build_canonical_manifest(
        paired_manifest_path=args.paired_manifest,
        selections_path=args.selections,
        output_path=args.manifest,
        n_splits=5,
        seed=42,
    )
    return args.manifest


def create_eval_loader(rows, task, regime, fold, batch_size, num_workers, image_size, dataset_root):
    eval_rows = [row for row in rows if int(row["fold"]) == fold]
    eval_dataset = PairedRegimeDataset(
        eval_rows,
        task=task,
        regime=regime,
        image_size=image_size,
        train=False,
        dataset_root=dataset_root,
    )
    summary = summarize_dataset(eval_dataset)
    print(
        f"[{task}][{regime}][eval] samples={summary['samples']} "
        f"shape=({summary['channels']}, {summary['height']}, {summary['width']}) "
        f"min={summary['min']} max={summary['max']} classes={summary['classes']}"
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return eval_dataset, eval_loader


def append_summary(summary_path: Path, rows):
    fieldnames = [
        "task",
        "regime",
        "fold",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "binary_f1",
        "checkpoint_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_checkpoint(checkpoint_path: Path, eval_dataset, eval_loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    in_channels = summarize_dataset(eval_dataset)["channels"]
    num_classes = len(eval_dataset.label_names)
    model = RegimeMobileNetV3Large(in_channels=in_channels, num_classes=num_classes, dropout=0.2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, sample_ids, trues, preds, probs = evaluate(model, eval_loader, eval_dataset.label_names, device)
    return checkpoint, metrics, sample_ids, trues, preds, probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = ensure_manifest(args)
    all_rows = read_manifest(manifest_path)

    tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
    regimes = [r for r in REGIMES.keys() if r != "recon_spatial"] if args.regime == "all" else [args.regime]
    write_dir = args.write_dir or (args.results_root / "re_evaluation")
    write_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for task in tasks:
        task_rows = filter_rows(all_rows, task)
        folds = folds_for_task(task_rows, args.folds)
        for regime in regimes:
            for fold in folds:
                checkpoint_path = args.results_root / task / regime / f"fold_{fold}" / "checkpoint.pt"
                if not checkpoint_path.exists():
                    print(f"Skipping missing checkpoint: {checkpoint_path}")
                    continue

                eval_dataset, eval_loader = create_eval_loader(
                    rows=task_rows,
                    task=task,
                    regime=regime,
                    fold=fold,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    image_size=args.image_size,
                    dataset_root=args.dataset_root,
                )
                checkpoint, metrics, sample_ids, trues, preds, probs = evaluate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    eval_dataset=eval_dataset,
                    eval_loader=eval_loader,
                    device=device,
                )

                fold_dir = write_dir / task / regime / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                save_predictions(fold_dir / "predictions.csv", sample_ids, eval_dataset.label_names, trues, preds, probs)
                with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2)
                with (fold_dir / "checkpoint_info.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "checkpoint_path": str(checkpoint_path),
                            "task": checkpoint.get("task", task),
                            "regime": checkpoint.get("regime", regime),
                            "fold": checkpoint.get("fold", fold),
                            "label_names": checkpoint.get("label_names", eval_dataset.label_names),
                            "parameter_report": checkpoint.get("parameter_report"),
                        },
                        handle,
                        indent=2,
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
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
                print(
                    f"[{task}][{regime}][fold={fold}] re_eval "
                    f"accuracy={metrics.get('accuracy'):.4f} macro_f1={metrics.get('macro_f1'):.4f}"
                )

    summary_path = write_dir / "summary_re_evaluated.csv"
    append_summary(summary_path, summary_rows)
    print(f"Wrote re-evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
