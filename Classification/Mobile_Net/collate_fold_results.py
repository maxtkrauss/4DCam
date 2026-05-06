import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOTS = [
    SCRIPT_DIR / "regime_results",
    SCRIPT_DIR / "roi_feature_results",
]
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "fold01_averaged_results.csv"
METRIC_NAMES = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "binary_f1",
    "train_loss",
    "best_train_loss",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collate Mobile_Net experiment metrics and average them across selected folds."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        default=None,
        help="Results root to scan. Can be passed multiple times. Defaults to regime_results and roi_feature_results.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1",
        help="Comma-separated folds to include. Default: 0,1",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path for averaged results.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the individual per-fold rows used in the aggregation.",
    )
    return parser.parse_args()


def parse_folds(folds_arg: str):
    folds = []
    for part in folds_arg.split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    return sorted(set(folds))


def parse_fold_from_name(folder_name: str):
    if not folder_name.startswith("fold_"):
        return None
    try:
        return int(folder_name.split("_", 1)[1])
    except ValueError:
        return None


def metric_sort_key(row):
    return (
        row["source"],
        row["task"],
        row["regime"],
        row["fold"],
    )


def average_metric(rows, metric_name: str):
    values = []
    for row in rows:
        value = row.get(metric_name)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def load_metric_row(results_root: Path, metrics_path: Path):
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    fold_dir = metrics_path.parent
    regime_dir = fold_dir.parent
    task_dir = regime_dir.parent
    fold = parse_fold_from_name(fold_dir.name)
    if fold is None:
        return None

    row = {
        "source": results_root.name,
        "results_root": str(results_root),
        "task": task_dir.name,
        "regime": regime_dir.name,
        "fold": fold,
        "metrics_path": str(metrics_path),
    }
    for metric_name in METRIC_NAMES:
        row[metric_name] = metrics.get(metric_name)
    return row


def collect_rows(results_root: Path, selected_folds):
    rows = []
    for metrics_path in sorted(results_root.glob("*/*/fold_*/metrics.json")):
        row = load_metric_row(results_root, metrics_path)
        if row is None or row["fold"] not in selected_folds:
            continue
        rows.append(row)
    rows.sort(key=metric_sort_key)
    return rows


def aggregate_rows(rows, selected_folds):
    grouped = {}
    for row in rows:
        key = (row["source"], row["results_root"], row["task"], row["regime"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (source, results_root, task, regime), group_rows in sorted(grouped.items()):
        folds_present = sorted({row["fold"] for row in group_rows})
        summary = {
            "source": source,
            "results_root": results_root,
            "task": task,
            "regime": regime,
            "requested_folds": ",".join(str(fold) for fold in selected_folds),
            "folds_present": ",".join(str(fold) for fold in folds_present),
            "num_folds_found": len(folds_present),
        }
        for metric_name in METRIC_NAMES:
            summary[f"{metric_name}_mean"] = average_metric(group_rows, metric_name)
        aggregate_rows.append(summary)
    return aggregate_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def print_summary(rows):
    if not rows:
        print("No matching metrics files were found.")
        return

    print("=" * 120)
    print("Fold-Averaged Results")
    print("=" * 120)
    print(f"{'Source':<22} {'Task':<16} {'Regime':<32} {'Folds':<10} {'Accuracy':>10}")
    print("-" * 120)
    for row in rows:
        accuracy = row.get("accuracy_mean")
        accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "n/a"
        print(
            f"{row['source']:<22} {row['task']:<16} {row['regime']:<32} "
            f"{row['folds_present']:<10} {accuracy_text:>10}"
        )
    print("=" * 120)


def main():
    args = parse_args()
    selected_folds = parse_folds(args.folds)
    results_roots = args.results_root if args.results_root else DEFAULT_RESULTS_ROOTS

    per_fold_rows = []
    for results_root in results_roots:
        if not results_root.exists():
            print(f"Skipping missing results root: {results_root}")
            continue
        per_fold_rows.extend(collect_rows(results_root, selected_folds))

    averaged_rows = aggregate_rows(per_fold_rows, selected_folds)

    averaged_fieldnames = [
        "source",
        "results_root",
        "task",
        "regime",
        "requested_folds",
        "folds_present",
        "num_folds_found",
    ] + [f"{metric_name}_mean" for metric_name in METRIC_NAMES]
    write_csv(args.output_csv, averaged_rows, averaged_fieldnames)

    details_csv = args.details_csv
    if details_csv is not None:
        detail_fieldnames = [
            "source",
            "results_root",
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
            "metrics_path",
        ]
        write_csv(details_csv, per_fold_rows, detail_fieldnames)

    print_summary(averaged_rows)
    print(f"Wrote averaged results to {args.output_csv}")
    if details_csv is not None:
        print(f"Wrote per-fold details to {details_csv}")


if __name__ == "__main__":
    main()
