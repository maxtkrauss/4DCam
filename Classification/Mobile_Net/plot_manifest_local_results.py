import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot scratch results from run_manifest_fold_local.py"
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
        default="all",
        help="Task to summarize",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold to summarize")
    parser.add_argument(
        "--metric",
        choices=["best_test_acc", "final_test_acc", "final_train_acc", "time_minutes"],
        default="best_test_acc",
        help="Metric to plot",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional output image path",
    )
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
        return sorted(runs_root.glob(f"*/**/fold_{fold}/metrics.json"))
    return sorted((runs_root / task).glob(f"*/fold_{fold}/metrics.json"))


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
    rows = [load_metrics(path) for path in metric_files]

    if not rows:
        print(f"No metrics found under {runs_root} for task={args.task} fold={args.fold}")
        return

    rows.sort(key=lambda row: (row["task"], -row[args.metric], row["regime"]))

    labels = [
        row["regime"] if args.task != "all" else f"{row['task']}:{row['regime']}"
        for row in rows
    ]
    values = [row[args.metric] for row in rows]

    fig_width = max(8, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(f"Manifest Local Runner | task={args.task} | fold={args.fold} | metric={args.metric}")
    ax.set_ylabel(args.metric)
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        if args.metric.endswith("_acc"):
            text = f"{value:.2f}%"
        else:
            text = f"{value:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_plot, dpi=200, bbox_inches="tight")
        print(f"Wrote plot to {args.output_plot}")
    else:
        default_name = f"{args.task}_fold{args.fold}_{args.metric}.png"
        default_path = args.scratch_work_dir / default_name
        fig.savefig(default_path, dpi=200, bbox_inches="tight")
        print(f"Wrote plot to {default_path}")

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"Wrote CSV summary to {args.output_csv}")


if __name__ == "__main__":
    main()
