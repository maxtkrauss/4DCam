import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix


TASK_LABEL_ORDERS = {
    "textile_3way": ["cotton", "felt", "nylon"],
    "camo_binary": ["no_camo", "camo"],
}

DISPLAY_LABELS = {
    "textile_3way": {
        "cotton": "cotton",
        "felt": "felt",
        "nylon": "nylon",
    },
    "camo_binary": {
        "no_camo": "plant",
        "camo": "camo",
    },
}

CLASS_COLORS = {
    "cotton": "#2b7bba",
    "felt": "#d62728",
    "nylon": "#2ca02c",
    "plant": "#2ca02c",
    "camo": "#006d1f",
}

REGIME_TITLES = {
    "raw_pol": "Scatterogram Classification",
    "recon_spatial": "Spatial Reconstruction Classification",
    "recon_polarimetric": "Polarimetric Reconstruction Classification",
    "recon_spectral": "Spectral Reconstruction Classification",
    "recon_specpol": "Spectro-Polarimetric Classification",
    "raw_stokes": "Raw Stokes Classification",
    "recon_stokes_polarimetric": "Polarimetric Stokes Classification",
    "recon_stokes_specpol": "Spectro-Stokes Classification",
}

METRIC_NAMES = [
    "best_test_acc",
    "final_test_acc",
    "final_train_acc",
    "best_macro_f1",
    "final_macro_f1",
    "time_minutes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze scratch manifest-runner outputs with fold mean +/- SD and aggregate confusion matrices."
    )
    parser.add_argument("--scratch-work-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["textile_3way", "camo_binary", "all"], default="all")
    parser.add_argument("--regime", type=str, default="all")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cm-style",
        choices=["custom", "basic"],
        default="custom",
        help="Plot confusion matrices in the publication-style custom format by default.",
    )
    return parser.parse_args()


def parse_folds(text: str):
    return sorted({int(part.strip()) for part in text.split(",") if part.strip()})


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_metric_paths(runs_root: Path, task_filter: str, regime_filter: str, folds):
    for metrics_path in sorted(runs_root.glob("*/*/fold_*/metrics.json")):
        task = metrics_path.parent.parent.parent.name
        regime = metrics_path.parent.parent.name
        fold_name = metrics_path.parent.name
        if not fold_name.startswith("fold_"):
            continue
        fold = int(fold_name.split("_", 1)[1])
        if fold not in folds:
            continue
        if task_filter != "all" and task != task_filter:
            continue
        if regime_filter != "all" and regime != regime_filter:
            continue
        yield task, regime, fold, metrics_path


def load_fold_rows(runs_root: Path, task_filter: str, regime_filter: str, folds):
    rows = []
    for task, regime, fold, metrics_path in iter_metric_paths(runs_root, task_filter, regime_filter, folds):
        payload = load_json(metrics_path)
        train_args = payload.get("train_args", {})
        row = {
            "task": task,
            "regime": regime,
            "fold": fold,
            "metrics_path": str(metrics_path),
            "prediction_path": str(metrics_path.parent / "predictions.csv"),
            "batch_size": train_args.get("batch_size"),
            "optimizer": train_args.get("optimizer"),
            "lr": maybe_float(train_args.get("lr")),
            "model_variant": train_args.get("model_variant"),
        }
        for metric_name in METRIC_NAMES:
            row[metric_name] = maybe_float(payload.get(metric_name))
        rows.append(row)
    rows.sort(key=lambda row: (row["task"], row["regime"], row["fold"]))
    return rows


def mean_sd(values):
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None, None
    mean = float(np.mean(usable))
    sd = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    return mean, sd


def summarize_fold_rows(rows, requested_folds):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["regime"]), []).append(row)

    summaries = []
    for (task, regime), group_rows in sorted(grouped.items()):
        folds_present = sorted({row["fold"] for row in group_rows})
        summary = {
            "task": task,
            "regime": regime,
            "requested_folds": ",".join(str(fold) for fold in requested_folds),
            "folds_present": ",".join(str(fold) for fold in folds_present),
            "num_folds_found": len(folds_present),
            "batch_size": ",".join(
                str(item) for item in sorted({row["batch_size"] for row in group_rows if row["batch_size"] is not None})
            ),
            "optimizer": ",".join(
                str(item) for item in sorted({row["optimizer"] for row in group_rows if row["optimizer"]})
            ),
            "lr": ",".join(
                f"{item:.6g}" for item in sorted({row["lr"] for row in group_rows if row["lr"] is not None})
            ),
            "model_variant": ",".join(
                str(item) for item in sorted({row["model_variant"] for row in group_rows if row["model_variant"]})
            ),
        }
        for metric_name in METRIC_NAMES:
            mean, sd = mean_sd([row.get(metric_name) for row in group_rows])
            summary[f"{metric_name}_mean"] = mean
            summary[f"{metric_name}_sd"] = sd
        summaries.append(summary)
    return summaries


def load_aggregate_predictions(rows):
    grouped = {}
    for row in rows:
        path = Path(row["prediction_path"])
        if path.exists():
            grouped.setdefault((row["task"], row["regime"]), []).append(path)

    outputs = []
    for (task, regime), paths in sorted(grouped.items()):
        frames = []
        for path in paths:
            df = pd.read_csv(path)
            df["prediction_path"] = str(path)
            df["fold"] = int(path.parent.name.split("_", 1)[1])
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        labels = TASK_LABEL_ORDERS.get(task, sorted(set(combined["true_label"]) | set(combined["pred_label"])))
        matrix = confusion_matrix(combined["true_label"], combined["pred_label"], labels=labels)
        outputs.append(
            {
                "task": task,
                "regime": regime,
                "labels": labels,
                "matrix": matrix,
                "num_predictions": int(len(combined)),
                "folds": sorted(combined["fold"].unique().tolist()),
                "dataframe": combined,
            }
        )
    return outputs


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def plot_confusion_matrix(matrix, labels, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(int(matrix.max()), 1)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = vmax / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_custom_confusion_matrix(matrix, labels, task: str, regime: str, output_path: Path):
    display_labels = [DISPLAY_LABELS.get(task, {}).get(label, label) for label in labels]
    title = REGIME_TITLES.get(regime, f"{regime} Classification")

    fig_size = (6.4, 5.4) if len(labels) == 2 else (7.4, 6.0)
    fig, ax = plt.subplots(figsize=fig_size)
    vmax = max(int(matrix.max()), 1)
    image = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=vmax)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_yticklabels(display_labels, fontsize=12)

    for tick, label in zip(ax.get_xticklabels(), display_labels):
        tick.set_color(CLASS_COLORS.get(label, "black"))
    for tick, label in zip(ax.get_yticklabels(), display_labels):
        tick.set_color(CLASS_COLORS.get(label, "black"))

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = vmax / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if value > threshold else "black",
            )

    for i, label in enumerate(display_labels):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=CLASS_COLORS.get(label, "black"),
                linewidth=2.2,
                alpha=0.95,
            )
        )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Count", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pct_pm(mean, sd):
    if mean is None:
        return "n/a"
    return f"{mean:.2f} +/- {sd:.2f}"


def write_markdown(path: Path, summaries, confusion_outputs):
    lines = ["# Scratch Run Analysis", ""]
    lines.append("## Mean +/- SD Across Folds")
    lines.append("")
    lines.append("| Task | Regime | Folds | Best Test | Final Test | Final Train | Macro-F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['num_folds_found']} | "
            f"{pct_pm(row['best_test_acc_mean'], row['best_test_acc_sd'])} | "
            f"{pct_pm(row['final_test_acc_mean'], row['final_test_acc_sd'])} | "
            f"{pct_pm(row['final_train_acc_mean'], row['final_train_acc_sd'])} | "
            f"{pct_pm(row['final_macro_f1_mean'], row['final_macro_f1_sd'])} |"
        )

    lines.append("")
    lines.append("## Aggregate Confusion Matrices")
    lines.append("")
    for item in confusion_outputs:
        png_name = f"{item['task']}__{item['regime']}__confusion.png"
        lines.append(f"- `{item['task']} / {item['regime']}`: [{png_name}]({png_name})")
    lines.append("")
    lines.append("Confusion matrices are computed by concatenating test predictions across completed folds.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    folds = parse_folds(args.folds)
    runs_root = args.scratch_work_dir / "runs"
    output_dir = args.output_dir or args.scratch_work_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = load_fold_rows(runs_root, args.task, args.regime, folds)
    if not fold_rows:
        print(f"No metrics found under {runs_root}")
        return

    summaries = summarize_fold_rows(fold_rows, folds)
    summary_fields = list(summaries[0].keys())
    write_csv(output_dir / "fold_metric_summary.csv", summaries, summary_fields)

    detail_fields = list(fold_rows[0].keys())
    write_csv(output_dir / "fold_metric_details.csv", fold_rows, detail_fields)

    confusion_outputs = load_aggregate_predictions(fold_rows)
    confusion_json = []
    for item in confusion_outputs:
        png_name = f"{item['task']}__{item['regime']}__confusion.png"
        if args.cm_style == "custom":
            plot_custom_confusion_matrix(
                item["matrix"],
                item["labels"],
                task=item["task"],
                regime=item["regime"],
                output_path=output_dir / png_name,
            )
        else:
            plot_confusion_matrix(
                item["matrix"],
                item["labels"],
                title=f"{item['task']} | {item['regime']}",
                output_path=output_dir / png_name,
            )
        item["dataframe"].to_csv(output_dir / f"{item['task']}__{item['regime']}__predictions_all_folds.csv", index=False)
        confusion_json.append(
            {
                "task": item["task"],
                "regime": item["regime"],
                "labels": item["labels"],
                "matrix": item["matrix"].tolist(),
                "num_predictions": item["num_predictions"],
                "folds": item["folds"],
                "png_name": png_name,
            }
        )

    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion_json, handle, indent=2)

    write_markdown(output_dir / "analysis_report.md", summaries, confusion_json)
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
