import argparse
import json
from pathlib import Path


METRICS = [
    "best_test_acc",
    "final_test_acc",
    "final_train_acc",
    "time_minutes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print per-fold and averaged scratch results from run_manifest_fold_local.py outputs."
    )
    parser.add_argument(
        "--scratch-work-dir",
        type=Path,
        required=True,
        help="Scratch root passed to run_manifest_fold_local.py",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1",
        help="Comma-separated folds to include. Default: 0,1",
    )
    parser.add_argument(
        "--task",
        choices=["textile_3way", "camo_binary", "all"],
        default="all",
        help="Task filter",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        help="Regime filter",
    )
    return parser.parse_args()


def parse_folds(text: str):
    folds = []
    for part in text.split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    return sorted(set(folds))


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(runs_root: Path, folds, task_filter: str, regime_filter: str):
    rows = []
    for metrics_path in sorted(runs_root.glob("*/*/fold_*/metrics.json")):
        fold_name = metrics_path.parent.name
        if not fold_name.startswith("fold_"):
            continue
        fold = int(fold_name.split("_", 1)[1])
        if fold not in folds:
            continue

        task = metrics_path.parent.parent.parent.name
        regime = metrics_path.parent.parent.name
        if task_filter != "all" and task != task_filter:
            continue
        if regime_filter != "all" and regime != regime_filter:
            continue

        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        rows.append(
            {
                "task": task,
                "regime": regime,
                "fold": fold,
                "best_test_acc": maybe_float(payload.get("best_test_acc")),
                "final_test_acc": maybe_float(payload.get("final_test_acc")),
                "final_train_acc": maybe_float(payload.get("final_train_acc")),
                "time_minutes": maybe_float(payload.get("time_minutes")),
                "metrics_path": str(metrics_path),
            }
        )

    rows.sort(key=lambda row: (row["task"], row["regime"], row["fold"]))
    return rows


def average(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def summarize(rows, requested_folds):
    grouped = {}
    for row in rows:
        key = (row["task"], row["regime"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (task, regime), group_rows in sorted(grouped.items()):
        folds_present = sorted({row["fold"] for row in group_rows})
        summaries.append(
            {
                "task": task,
                "regime": regime,
                "requested_folds": ",".join(str(fold) for fold in requested_folds),
                "folds_present": ",".join(str(fold) for fold in folds_present),
                "num_folds_found": len(folds_present),
                "best_test_acc": average([row["best_test_acc"] for row in group_rows]),
                "final_test_acc": average([row["final_test_acc"] for row in group_rows]),
                "final_train_acc": average([row["final_train_acc"] for row in group_rows]),
                "time_minutes": average([row["time_minutes"] for row in group_rows]),
            }
        )
    return summaries


def pct_text(value):
    if value is None:
        return "n/a"
    return f"{value:6.2f}%"


def min_text(value):
    if value is None:
        return "n/a"
    return f"{value:6.1f}"


def print_fold_table(rows):
    print("=" * 112)
    print("Per-Fold Results")
    print("=" * 112)
    print(f"{'Task':<14} {'Regime':<28} {'Fold':<6} {'Best Test':>10} {'Final Test':>11} {'Final Train':>12} {'Min':>8}")
    print("-" * 112)
    for row in rows:
        print(
            f"{row['task']:<14} {row['regime']:<28} {row['fold']:<6} "
            f"{pct_text(row['best_test_acc']):>10} {pct_text(row['final_test_acc']):>11} "
            f"{pct_text(row['final_train_acc']):>12} {min_text(row['time_minutes']):>8}"
        )
    print("=" * 112)


def print_average_table(rows):
    print("=" * 124)
    print("Averaged Results")
    print("=" * 124)
    print(
        f"{'Task':<14} {'Regime':<28} {'Folds':<10} {'N':<4} "
        f"{'Best Test':>10} {'Final Test':>11} {'Final Train':>12} {'Min':>8}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{row['task']:<14} {row['regime']:<28} {row['folds_present']:<10} {row['num_folds_found']:<4} "
            f"{pct_text(row['best_test_acc']):>10} {pct_text(row['final_test_acc']):>11} "
            f"{pct_text(row['final_train_acc']):>12} {min_text(row['time_minutes']):>8}"
        )
    print("=" * 124)


def main():
    args = parse_args()
    runs_root = args.scratch_work_dir / "runs"
    folds = parse_folds(args.folds)

    rows = load_rows(runs_root, folds, args.task, args.regime)
    if not rows:
        print(
            f"No scratch metrics found under {runs_root} for folds={','.join(str(fold) for fold in folds)} "
            f"task={args.task} regime={args.regime}"
        )
        return

    averages = summarize(rows, folds)
    print_fold_table(rows)
    print()
    print_average_table(averages)


if __name__ == "__main__":
    main()
