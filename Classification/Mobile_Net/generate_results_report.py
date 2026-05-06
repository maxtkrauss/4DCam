import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "regime_results"
DEFAULT_SUMMARY = DEFAULT_RESULTS_ROOT / "summary_comparison.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "analysis"

TASK_LABEL_ORDERS = {
    "textile_3way": ["cotton", "felt", "nylon"],
    "camo_binary": ["camo", "no_camo"],
}

PRIMARY_METRIC = {
    "textile_3way": "macro_f1",
    "camo_binary": "macro_f1",
}

SECONDARY_METRIC = {
    "textile_3way": "accuracy",
    "camo_binary": "accuracy",
}


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_prediction_files(results_root: Path, task: str, regime: str):
    regime_dir = results_root / task / regime
    return sorted(regime_dir.glob("fold_*/predictions.csv"))


def aggregate_predictions(results_root: Path, task: str, regime: str):
    prediction_files = collect_prediction_files(results_root, task, regime)
    if not prediction_files:
        return None

    frames = [pd.read_csv(path) for path in prediction_files]
    df = pd.concat(frames, ignore_index=True)
    labels = TASK_LABEL_ORDERS[task]
    matrix = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    return df, matrix, labels


def plot_confusion_matrix(matrix, labels, title, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(int(matrix.max()), 1)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = vmax / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(summary_df: pd.DataFrame):
    numeric_cols = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "binary_f1",
        "train_loss",
    ]
    rows = []
    for (task, regime), group in summary_df.groupby(["task", "regime"], sort=True):
        row = {"task": task, "regime": regime, "folds": int(group["fold"].nunique())}
        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else None
            row[f"{col}_sd"] = float(vals.std(ddof=0)) if not vals.empty else None
        rows.append(row)
    return rows


def select_best_rows(metric_rows, task: str):
    primary = PRIMARY_METRIC[task]
    secondary = SECONDARY_METRIC[task]
    rows = [row for row in metric_rows if row["task"] == task]
    rows = [row for row in rows if row.get(f"{primary}_mean") is not None]
    rows.sort(
        key=lambda row: (
            row.get(f"{primary}_mean", float("-inf")),
            row.get(f"{secondary}_mean", float("-inf")),
        ),
        reverse=True,
    )
    return rows


def build_markdown_report(metric_rows, confusion_outputs, output_path: Path):
    lines = []
    lines.append("# MobileNetV3 Classification Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    for task in ["textile_3way", "camo_binary"]:
        ranked = select_best_rows(metric_rows, task)
        best = ranked[0] if ranked else None
        if task == "textile_3way":
            lines.append("### Textile 3-Way")
            if best:
                lines.append(
                    f"- Best regime by Macro-F1: `{best['regime']}` "
                    f"with Accuracy `{best['accuracy_mean']:.3f} +/- {best['accuracy_sd']:.3f}` "
                    f"and Macro-F1 `{best['macro_f1_mean']:.3f} +/- {best['macro_f1_sd']:.3f}`."
                )
        else:
            lines.append("### Camo Binary")
            if best:
                lines.append(
                    f"- Best regime by Macro-F1: `{best['regime']}` "
                    f"with Accuracy `{best['accuracy_mean']:.3f} +/- {best['accuracy_sd']:.3f}` "
                    f"and Macro-F1 `{best['macro_f1_mean']:.3f} +/- {best['macro_f1_sd']:.3f}`."
                )
        if ranked:
            lines.append("- Top regimes (Accuracy and Macro-F1 across folds):")
            for row in ranked[:3]:
                lines.append(
                    f"  - `{row['regime']}`: Accuracy `{row['accuracy_mean']:.3f} +/- {row['accuracy_sd']:.3f}`, "
                    f"Macro-F1 `{row['macro_f1_mean']:.3f} +/- {row['macro_f1_sd']:.3f}`."
                )
        lines.append("")

    lines.append("## Confusion Matrices")
    lines.append("")
    for task in ["textile_3way", "camo_binary"]:
        lines.append(f"### {task}")
        relevant = [item for item in confusion_outputs if item["task"] == task]
        for item in relevant:
            lines.append(
                f"- `{item['regime']}`: [{item['png_name']}]({item['png_name']})"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Confusion matrices were aggregated across all completed folds by concatenating each fold's predictions.")
    lines.append("- Summary metrics were computed as mean and standard deviation across folds from `summary_comparison.csv`.")
    lines.append("- Headline reporting uses Accuracy (Mean +/- SD) and Macro-F1 (Mean +/- SD) for every regime.")
    lines.append("- Regimes are ranked by Macro-F1 with Accuracy used as the secondary sort key.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrices and a results writeup from finished MobileNet experiments.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(args.summary)
    metric_rows = summarize_metrics(summary_df)

    metric_csv = args.output_dir / "aggregated_metrics.csv"
    metric_fields = list(metric_rows[0].keys()) if metric_rows else []
    if metric_rows:
        save_csv(metric_csv, metric_rows, metric_fields)

    confusion_outputs = []
    for task, regime_group in summary_df.groupby(["task", "regime"], sort=True):
        task_name, regime = task
        aggregated = aggregate_predictions(args.results_root, task_name, regime)
        if aggregated is None:
            continue

        _, matrix, labels = aggregated
        png_name = f"{task_name}__{regime}__confusion.png"
        plot_confusion_matrix(
            matrix,
            labels,
            title=f"{task_name} | {regime}",
            output_path=args.output_dir / png_name,
        )

        confusion_outputs.append(
            {
                "task": task_name,
                "regime": regime,
                "png_name": png_name,
                "labels": labels,
                "matrix": matrix.tolist(),
            }
        )

    with (args.output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion_outputs, handle, indent=2)

    build_markdown_report(metric_rows, confusion_outputs, args.output_dir / "results_writeup.md")
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
