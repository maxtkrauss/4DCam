import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize all completed scratch runs from run_manifest_fold_local.py."
    )
    parser.add_argument(
        "--scratch-work-dir",
        type=Path,
        required=True,
        help="Scratch root used with run_manifest_fold_local.py",
    )
    parser.add_argument(
        "--task",
        choices=["textile_3way", "camo_binary", "all"],
        default="all",
        help="Optional task filter",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        help="Optional regime filter",
    )
    return parser.parse_args()


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_pct(value):
    if value is None:
        return "n/a"
    return f"{value:6.2f}%"


def format_float(value, decimals=1):
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def average(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def load_rows(runs_root: Path, task_filter: str, regime_filter: str):
    rows = []
    for metrics_path in sorted(runs_root.glob("*/*/fold_*/metrics.json")):
        task = metrics_path.parent.parent.parent.name
        regime = metrics_path.parent.parent.name
        fold_name = metrics_path.parent.name

        if not fold_name.startswith("fold_"):
            continue
        fold = int(fold_name.split("_", 1)[1])

        if task_filter != "all" and task != task_filter:
            continue
        if regime_filter != "all" and regime != regime_filter:
            continue

        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        train_args = payload.get("train_args", {})
        rows.append(
            {
                "task": task,
                "regime": regime,
                "fold": fold,
                "best_test_acc": maybe_float(payload.get("best_test_acc")),
                "final_test_acc": maybe_float(payload.get("final_test_acc")),
                "final_train_acc": maybe_float(payload.get("final_train_acc")),
                "time_minutes": maybe_float(payload.get("time_minutes")),
                "batch_size": train_args.get("batch_size"),
                "epochs": train_args.get("epochs"),
                "optimizer": train_args.get("optimizer", "rmsprop"),
                "lr": maybe_float(train_args.get("lr")),
                "weight_decay": maybe_float(train_args.get("weight_decay")),
                "metrics_path": str(metrics_path),
            }
        )

    rows.sort(key=lambda row: (row["task"], row["regime"], row["fold"]))
    return rows


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["task"], row["regime"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (task, regime), group_rows in sorted(grouped.items()):
        folds_present = sorted({row["fold"] for row in group_rows})
        batch_sizes = sorted({row["batch_size"] for row in group_rows if row.get("batch_size") is not None})
        optimizers = sorted({row["optimizer"] for row in group_rows if row.get("optimizer") is not None})
        lrs = sorted({row["lr"] for row in group_rows if row.get("lr") is not None})
        wds = sorted({row["weight_decay"] for row in group_rows if row.get("weight_decay") is not None})

        summaries.append(
            {
                "task": task,
                "regime": regime,
                "folds": ",".join(str(fold) for fold in folds_present),
                "n": len(folds_present),
                "best_test_acc": average([row["best_test_acc"] for row in group_rows]),
                "final_test_acc": average([row["final_test_acc"] for row in group_rows]),
                "final_train_acc": average([row["final_train_acc"] for row in group_rows]),
                "time_minutes": average([row["time_minutes"] for row in group_rows]),
                "batch_size": ",".join(str(item) for item in batch_sizes) if batch_sizes else "",
                "optimizer": ",".join(optimizers),
                "lr": ",".join(f"{item:.6g}" for item in lrs) if lrs else "",
                "weight_decay": ",".join(f"{item:.6g}" for item in wds) if wds else "",
            }
        )
    return summaries


def print_run_config(rows):
    batch_sizes = sorted({row["batch_size"] for row in rows if row.get("batch_size") is not None})
    optimizers = sorted({row["optimizer"] for row in rows if row.get("optimizer") is not None})
    lrs = sorted({row["lr"] for row in rows if row.get("lr") is not None})
    wds = sorted({row["weight_decay"] for row in rows if row.get("weight_decay") is not None})
    epochs = sorted({row["epochs"] for row in rows if row.get("epochs") is not None})
    folds = sorted({row["fold"] for row in rows})

    print("=" * 96)
    print("Run Summary")
    print("=" * 96)
    print(f"Completed folds : {','.join(str(fold) for fold in folds)}")
    print(f"Optimizers      : {', '.join(optimizers) if optimizers else 'n/a'}")
    print(f"Batch sizes     : {', '.join(str(item) for item in batch_sizes) if batch_sizes else 'n/a'}")
    print(f"Epochs          : {', '.join(str(item) for item in epochs) if epochs else 'n/a'}")
    print(f"Learning rates  : {', '.join(f'{item:.6g}' for item in lrs) if lrs else 'n/a'}")
    print(f"Weight decay    : {', '.join(f'{item:.6g}' for item in wds) if wds else 'n/a'}")
    print("=" * 96)


def print_per_fold(rows):
    print("=" * 144)
    print("Per-Fold Results")
    print("=" * 144)
    print(
        f"{'Task':<14} {'Regime':<28} {'Fold':<6} {'Best Test':>10} {'Final Test':>11} "
        f"{'Final Train':>12} {'Min':>8} {'BS':>6} {'Opt':>10} {'LR':>10}"
    )
    print("-" * 144)
    for row in rows:
        print(
            f"{row['task']:<14} {row['regime']:<28} {row['fold']:<6} "
            f"{format_pct(row['best_test_acc']):>10} {format_pct(row['final_test_acc']):>11} "
            f"{format_pct(row['final_train_acc']):>12} {format_float(row['time_minutes'], 1):>8} "
            f"{str(row['batch_size']) if row.get('batch_size') is not None else 'n/a':>6} "
            f"{row.get('optimizer', 'n/a'):>10} "
            f"{format_float(row['lr'], 4):>10}"
        )
    print("=" * 144)


def print_averages(rows):
    print("=" * 160)
    print("Averaged Results")
    print("=" * 160)
    print(
        f"{'Task':<14} {'Regime':<28} {'Folds':<12} {'N':<4} {'Best Test':>10} "
        f"{'Final Test':>11} {'Final Train':>12} {'Min':>8} {'BS':>8} {'Opt':>10} {'LR':>12}"
    )
    print("-" * 160)
    for row in rows:
        print(
            f"{row['task']:<14} {row['regime']:<28} {row['folds']:<12} {row['n']:<4} "
            f"{format_pct(row['best_test_acc']):>10} {format_pct(row['final_test_acc']):>11} "
            f"{format_pct(row['final_train_acc']):>12} {format_float(row['time_minutes'], 1):>8} "
            f"{row['batch_size'] if row['batch_size'] else 'n/a':>8} "
            f"{row['optimizer'] if row['optimizer'] else 'n/a':>10} "
            f"{row['lr'] if row['lr'] else 'n/a':>12}"
        )
    print("=" * 160)


def main():
    args = parse_args()
    runs_root = args.scratch_work_dir / "runs"
    rows = load_rows(runs_root, args.task, args.regime)

    if not rows:
        print(f"No completed runs found under {runs_root}")
        return

    print_run_config(rows)
    print()
    print_per_fold(rows)
    print()
    print_averages(summarize_rows(rows))


if __name__ == "__main__":
    main()
