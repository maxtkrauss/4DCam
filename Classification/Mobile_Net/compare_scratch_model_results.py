import argparse
import csv
import json
import statistics
from pathlib import Path


TASKS = ("textile_3way", "camo_binary")
TASK_LABELS = {
    "textile_3way": "Textile Classification",
    "camo_binary": "Camouflage Classification",
}

REGIME_ORDER = (
    "raw_pol",
    "raw_stokes",
    "recon_spatial",
    "recon_polarimetric",
    "recon_stokes_polarimetric",
    "recon_spectral",
    "recon_specpol",
    "recon_stokes_specpol",
)
REGIME_LABELS = {
    "raw_pol": "4-channel scatterogram",
    "raw_stokes": "3-channel raw Stokes scatterogram",
    "recon_spatial": "1-channel spectral-polarimetrically averaged reconstruction",
    "recon_polarimetric": "4-channel spectrally averaged reconstruction",
    "recon_stokes_polarimetric": "3-channel spectrally averaged Stokes reconstruction",
    "recon_spectral": "106-channel polarimetrically averaged reconstruction",
    "recon_specpol": "424-channel spectro-polarimetric reconstruction",
    "recon_stokes_specpol": "318-channel spectro-Stokes reconstruction",
}
CHANNELS = {
    "raw_pol": 4,
    "raw_stokes": 3,
    "recon_spatial": 1,
    "recon_polarimetric": 4,
    "recon_stokes_polarimetric": 3,
    "recon_spectral": 106,
    "recon_specpol": 424,
    "recon_stokes_specpol": 318,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare scratch model results across the five canonical MobileNet/ResNet modalities."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Model label and scratch root in the form Label=/scratch/path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--metric",
        choices=["best_test_acc", "final_test_acc", "final_train_acc"],
        default="final_test_acc",
    )
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated folds to include.")
    parser.add_argument(
        "--sort",
        choices=["modality", "textile", "camo"],
        default="modality",
        help="Row order for per-run tables.",
    )
    parser.add_argument(
        "--regime-set",
        choices=["original", "stokes", "all"],
        default="original",
        help="Regimes to include in the comparison table.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def parse_run(text: str):
    if "=" not in text:
        raise ValueError(f"--run must be Label=/path, got: {text}")
    label, root = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing run label in --run {text}")
    return label, Path(root).expanduser()


def parse_folds(text: str):
    return sorted({int(part.strip()) for part in text.split(",") if part.strip()})


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_path(run_root: Path, task: str, regime: str, fold: int):
    return run_root / "runs" / task / regime / f"fold_{fold}" / "metrics.json"


def load_metric(run_root: Path, task: str, regime: str, fold: int, metric: str):
    path = metric_path(run_root, task, regime, fold)
    if not path.exists():
        return None, path
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return maybe_float(payload.get(metric)), path


def summarize_values(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None, None, 0
    mean = sum(usable) / len(usable)
    stdev = statistics.stdev(usable) if len(usable) > 1 else 0.0
    return mean, stdev, len(usable)


def regimes_for_set(regime_set: str):
    if regime_set == "stokes":
        return ("raw_stokes", "recon_stokes_polarimetric", "recon_stokes_specpol")
    if regime_set == "all":
        return REGIME_ORDER
    return ("raw_pol", "recon_spatial", "recon_polarimetric", "recon_spectral", "recon_specpol")


def load_run(label: str, root: Path, folds, metric: str, regimes):
    rows = []
    for regime in regimes:
        for task in TASKS:
            values = []
            missing = []
            for fold in folds:
                value, path = load_metric(root, task, regime, fold, metric)
                values.append(value)
                if value is None:
                    missing.append(str(path))
            mean, stdev, n = summarize_values(values)
            rows.append(
                {
                    "model": label,
                    "root": str(root),
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "regime": regime,
                    "modality": REGIME_LABELS[regime],
                    "channels": CHANNELS[regime],
                    "metric": metric,
                    "mean": mean,
                    "stdev": stdev,
                    "n": n,
                    "folds_requested": ",".join(str(fold) for fold in folds),
                    "missing": ";".join(missing),
                }
            )
    return rows


def pct(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def pct_pm(mean, stdev, n):
    if mean is None:
        return "n/a"
    if n <= 1:
        return pct(mean)
    return f"{mean:.2f}% +/- {stdev:.2f}"


def pivot_run(rows):
    by_regime = {}
    for row in rows:
        by_regime.setdefault(row["regime"], {})[row["task"]] = row
    return by_regime


def row_sort_key(sort_mode: str, by_regime, regime: str):
    if sort_mode == "textile":
        value = by_regime[regime].get("textile_3way", {}).get("mean")
        return (float("inf") if value is None else value, REGIME_ORDER.index(regime))
    if sort_mode == "camo":
        value = by_regime[regime].get("camo_binary", {}).get("mean")
        return (float("inf") if value is None else value, REGIME_ORDER.index(regime))
    return (REGIME_ORDER.index(regime),)


def make_run_table(label: str, rows, sort_mode: str, regimes):
    by_regime = pivot_run(rows)
    regimes = sorted(regimes, key=lambda regime: row_sort_key(sort_mode, by_regime, regime))

    lines = [
        f"### {label}",
        "",
        "| Modality | Channels | Textile Classification | Camouflage Classification |",
        "|---|---:|---:|---:|",
    ]
    for regime in regimes:
        textile = by_regime[regime].get("textile_3way", {})
        camo = by_regime[regime].get("camo_binary", {})
        lines.append(
            f"| {REGIME_LABELS[regime]} | {CHANNELS[regime]} | "
            f"{pct_pm(textile.get('mean'), textile.get('stdev'), textile.get('n', 0))} | "
            f"{pct_pm(camo.get('mean'), camo.get('stdev'), camo.get('n', 0))} |"
        )
    return "\n".join(lines)


def make_delta_table(run_labels, all_rows, regimes):
    if len(run_labels) != 2:
        return None
    first, second = run_labels
    lookup = {(row["model"], row["task"], row["regime"]): row for row in all_rows}
    lines = [
        f"### Delta: {second} minus {first}",
        "",
        "| Modality | Textile Delta | Camouflage Delta |",
        "|---|---:|---:|",
    ]
    for regime in regimes:
        cells = []
        for task in TASKS:
            before = lookup.get((first, task, regime), {}).get("mean")
            after = lookup.get((second, task, regime), {}).get("mean")
            if before is None or after is None:
                cells.append("n/a")
            else:
                cells.append(f"{after - before:+.2f} pts")
        lines.append(f"| {REGIME_LABELS[regime]} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def write_csv(path: Path, rows):
    fieldnames = [
        "model",
        "root",
        "task",
        "task_label",
        "regime",
        "modality",
        "channels",
        "metric",
        "mean",
        "stdev",
        "n",
        "folds_requested",
        "missing",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main():
    args = parse_args()
    folds = parse_folds(args.folds)
    runs = [parse_run(item) for item in args.run]
    regimes = regimes_for_set(args.regime_set)

    all_rows = []
    sections = [f"# Scratch Result Comparison ({args.metric})"]
    for label, root in runs:
        rows = load_run(label, root, folds, args.metric, regimes)
        all_rows.extend(rows)
        sections.append(make_run_table(label, rows, args.sort, regimes))

    delta = make_delta_table([label for label, _ in runs], all_rows, regimes)
    if delta is not None:
        sections.append(delta)

    markdown = "\n\n".join(sections)
    print(markdown)

    if args.output_csv is not None:
        write_csv(args.output_csv, all_rows)
        print(f"\nWrote CSV to {args.output_csv}")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote Markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()
