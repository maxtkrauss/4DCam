import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize scratch results from run_manifest_fold_local.py"
    )
    parser.add_argument(
        "--scratch-work-dir",
        type=Path,
        required=True,
        help="Scratch root passed to run_manifest_fold_local.py",
    )
    parser.add_argument(
        "--task",
        choices=["textile_3way", "camo_binary", "all"],
        default="textile_3way",
        help="Task to summarize",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold to summarize")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path):
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["metrics_path"] = str(metrics_path)
    return payload


def find_metric_files(runs_root: Path, task: str, fold: int):
    if task == "all":
        return sorted(runs_root.rglob(f"fold_{fold}/metrics.json"))
    return sorted((runs_root / task).glob(f"*/fold_{fold}/metrics.json"))


def summarize(rows):
    if not rows:
        return None
    best = [row["best_test_acc"] for row in rows]
    final_test = [row["final_test_acc"] for row in rows]
    final_train = [row["final_train_acc"] for row in rows]
    return {
        "count": len(rows),
        "best_test_acc_mean": sum(best) / len(best),
        "final_test_acc_mean": sum(final_test) / len(final_test),
        "final_train_acc_mean": sum(final_train) / len(final_train),
    }


def write_csv(path: Path, rows):
    fieldnames = [
        "task",
        "regime",
        "fold",
        "best_test_acc",
        "final_test_acc",
        "final_train_acc",
        "time_minutes",
        "metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    runs_root = args.scratch_work_dir / "runs"
    metric_files = find_metric_files(runs_root, args.task, args.fold)

    rows = []
    for metrics_path in metric_files:
        rows.append(load_metrics(metrics_path))

    rows.sort(key=lambda row: (-row["best_test_acc"], row["task"], row["regime"]))

    if not rows:
        print(f"No metrics found under {runs_root} for task={args.task} fold={args.fold}")
        return

    print("=" * 96)
    print(f"Manifest Local Runner Summary | task={args.task} | fold={args.fold}")
    print("=" * 96)
    print(f"{'Task':<14} {'Regime':<22} {'Best Test':>10} {'Final Test':>11} {'Final Train':>12} {'Min':>8}")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['task']:<14} {row['regime']:<22} "
            f"{row['best_test_acc']:>9.2f}% {row['final_test_acc']:>10.2f}% "
            f"{row['final_train_acc']:>11.2f}% {row['time_minutes']:>7.1f}"
        )

    totals = summarize(rows)
    print("-" * 96)
    print(
        f"{'MEAN':<37} "
        f"{totals['best_test_acc_mean']:>9.2f}% {totals['final_test_acc_mean']:>10.2f}% "
        f"{totals['final_train_acc_mean']:>11.2f}%"
    )
    print("=" * 96)

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"Wrote CSV summary to {args.output_csv}")


if __name__ == "__main__":
    main()
