import argparse
import csv
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cross_validate import train_one_fold as train_textiles_one_fold
from cross_validate_foliage import train_one_fold as train_camo_one_fold
from manifest_local_runner_utils import (
    REGIME_CONFIG,
    SUPPORTED_REGIMES,
    SUPPORTED_TASKS,
    copy_tree_contents,
    ensure_manifest,
    is_scratch_path,
    materialize_fold_dataset,
    read_csv_rows,
    save_json,
    summarize_materialized_channels,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PAIRED_MANIFEST = REPO_ROOT.parent / "paired_textile_camo_manifest_2026-04-09.csv"
DEFAULT_SELECTIONS = REPO_ROOT.parent / "double_pair_manual_selections_2026-04-09.csv"
DEFAULT_MANIFEST = REPO_ROOT.parent / "canonical_paired_manifest_2026-04-09.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local-faithful MobileNet fold training against external manifest data using scratch-only prepared datasets."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Scratch-backed dataset root used to remap Windows manifest paths.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--task", choices=list(SUPPORTED_TASKS) + ["all"], default="all")
    parser.add_argument("--regime", choices=list(SUPPORTED_REGIMES) + ["all"], default="all")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--scratch-work-dir", type=Path, required=True, help="Required scratch root for prepared datasets and run artifacts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to receive promoted final artifacts.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override applied to every regime.")
    parser.add_argument(
        "--model-variant",
        choices=["baseline", "shared_projection_stem"],
        default="baseline",
        help="Model architecture variant to use.",
    )
    parser.add_argument(
        "--projection-channels",
        type=int,
        default=16,
        help="Channel count for the shared projection stem when enabled.",
    )
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop", help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-temp", action="store_true", help="Preserve scratch-prepared TIFF datasets after training.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip task/regime/fold runs that already have metrics.json in scratch runs.")
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
        raise ValueError(
            f"--scratch-work-dir must point at scratch storage. Got: {scratch_work_dir}"
        )
    if not is_scratch_path(dataset_root):
        raise ValueError(
            f"--dataset-root must point at scratch storage. Got: {dataset_root}"
        )

    repo_root = REPO_ROOT.resolve()
    if repo_root == scratch_work_dir or repo_root in scratch_work_dir.parents:
        raise ValueError("--scratch-work-dir cannot live inside the repo checkout.")

    if repo_root == dataset_root or repo_root in dataset_root.parents:
        raise ValueError("--dataset-root cannot live inside the repo checkout.")

    args.scratch_work_dir = scratch_work_dir
    args.dataset_root = str(dataset_root)
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()


def build_training_args(args, task: str, regime: str):
    config = REGIME_CONFIG[regime]
    epochs = args.epochs if args.epochs is not None else 50
    train_args = SimpleNamespace(
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        epochs=epochs,
        model_variant=args.model_variant,
        projection_channels=args.projection_channels,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        lr_step_size=20,
        lr_gamma=0.1,
        warmup_epochs=0,
        warmup_lr_init=0.0,
        dropout=0.0,
        label_smoothing=0.0,
        num_channels=config.num_channels,
        workers=args.workers,
        grayscale=config.grayscale,
        global_normalize=True,
        disable_ema=True,
    )
    return train_args


def trainer_for_task(task: str):
    if task == "textile_3way":
        return train_textiles_one_fold
    if task == "camo_binary":
        return train_camo_one_fold
    raise ValueError(f"Unsupported task {task}")


def append_summary(path: Path, rows):
    fieldnames = [
        "task",
        "regime",
        "fold",
        "best_test_acc",
        "final_test_acc",
        "final_train_acc",
        "time_minutes",
        "train_dir",
        "test_dir",
    ]
    file_exists = path.exists()
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
    candidates = sorted(prepared_root.rglob("*.tif"))
    return candidates[:limit]


def main():
    args = parse_args()
    validate_scratch_paths(args)
    seed_everything(args.seed)

    manifest_path = ensure_manifest(args.manifest, args.paired_manifest, args.selections, args.seed)
    all_rows = read_csv_rows(manifest_path)

    tasks = list(SUPPORTED_TASKS) if args.task == "all" else [args.task]
    regimes = list(SUPPORTED_REGIMES) if args.regime == "all" else [args.regime]

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

            example_tiffs = collect_example_tiffs(prepared_root)
            sample_channel_summary = summarize_materialized_channels(example_tiffs)
            save_json(
                run_root / "materialization_summary.json",
                {
                    "task": task,
                    "regime": regime,
                    "fold": args.fold,
                    "prepared_root": str(prepared_root),
                    "train_dir": str(materialized["train_dir"]),
                    "test_dir": str(materialized["test_dir"]),
                    "sample_count_train": len(materialized["train_rows"]),
                    "sample_count_test": len(materialized["test_rows"]),
                    "reused_prepared_dataset": bool(materialized.get("reused")),
                    "example_channel_summary": sample_channel_summary,
                },
            )

            train_args = build_training_args(args, task, regime)
            trainer = trainer_for_task(task)

            print(f"Training task={task} regime={regime} fold={args.fold}")
            start_time = time.time()
            cwd_before = Path.cwd()
            try:
                os.chdir(run_root)
                best_acc, final_train_acc, final_test_acc = trainer(
                    str(materialized["train_dir"]),
                    str(materialized["test_dir"]),
                    train_args,
                    args.fold,
                )
            finally:
                os.chdir(cwd_before)
            elapsed_minutes = (time.time() - start_time) / 60.0

            result_payload = {
                "task": task,
                "regime": regime,
                "fold": args.fold,
                "best_test_acc": best_acc,
                "final_test_acc": final_test_acc,
                "final_train_acc": final_train_acc,
                "time_minutes": elapsed_minutes,
                "train_args": vars(train_args),
                "train_dir": str(materialized["train_dir"]),
                "test_dir": str(materialized["test_dir"]),
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
                    "best_test_acc": best_acc,
                    "final_test_acc": final_test_acc,
                    "final_train_acc": final_train_acc,
                    "time_minutes": elapsed_minutes,
                    "train_dir": str(materialized["train_dir"]),
                    "test_dir": str(materialized["test_dir"]),
                }
            )

            if not args.keep_temp:
                shutil.rmtree(prepared_root, ignore_errors=True)

    append_summary(args.scratch_work_dir / "runs" / "summary_local_manifest_runner.csv", summary_rows)
    print(f"Wrote scratch summary to {args.scratch_work_dir / 'runs' / 'summary_local_manifest_runner.csv'}")

    if args.output_dir is not None:
        append_summary(args.output_dir / "summary_local_manifest_runner.csv", summary_rows)
        print(f"Wrote promoted summary to {args.output_dir / 'summary_local_manifest_runner.csv'}")


if __name__ == "__main__":
    main()
