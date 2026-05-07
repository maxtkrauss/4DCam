import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRED_MANIFEST = REPO_ROOT / "paired_textile_camo_manifest_2026-04-09.csv"
DEFAULT_SELECTIONS = REPO_ROOT / "double_pair_manual_selections_2026-04-09.csv"
DEFAULT_OUTPUT = REPO_ROOT / "canonical_paired_manifest_2026-04-09.csv"


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_key(row):
    return f"{row.get('dataset','')}|{row.get('image_number','')}|{row.get('source_path','')}"


def resolve_reconstruction_path(row, selection_by_key):
    match_count = int(row.get("recon_match_count", "0") or "0")
    recon_paths = [part.strip() for part in row.get("recon_paths", "").split(";") if part.strip()]

    if match_count == 1:
        if not recon_paths:
            raise ValueError(f"Missing reconstruction path for {row_key(row)}")
        return recon_paths[0]

    if match_count == 2:
        key = row_key(row)
        if key not in selection_by_key:
            raise ValueError(f"Missing manual selection for {key}")
        return selection_by_key[key]["selected_recon_path"]

    raise ValueError(f"Unsupported reconstruction match count {match_count} for {row_key(row)}")


def assign_folds(rows, n_splits=5, seed=42):
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    for task_rows in by_task.values():
        task_rows.sort(key=lambda item: (item["class_label"], item["dataset"], int(item["image_number"])))
        labels = np.asarray([row["class_label"] for row in task_rows])
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(task_rows)), labels)):
            for idx in test_idx:
                task_rows[idx]["fold"] = fold

    return rows


def build_canonical_manifest(paired_manifest_path: Path, selections_path: Path, output_path: Path, n_splits=5, seed=42):
    paired_rows = read_csv(paired_manifest_path)
    selection_rows = read_csv(selections_path)
    selection_by_key = {row_key(row): row for row in selection_rows}

    canonical_rows = []
    for row in paired_rows:
        if row.get("notes") != "paired":
            continue

        task = "textile_3way" if row["group"] == "textile" else "camo_binary"
        class_label = row["class"]
        sample_id = f"{task}|{row['dataset']}|{row['image_number']}"
        canonical_rows.append(
            {
                "task": task,
                "group": row["group"],
                "dataset": row["dataset"],
                "class_label": class_label,
                "sample_id": sample_id,
                "image_number": row["image_number"],
                "raw_path": row["source_path"],
                "selected_recon_path": resolve_reconstruction_path(row, selection_by_key),
            }
        )

    canonical_rows = assign_folds(canonical_rows, n_splits=n_splits, seed=seed)
    fieldnames = [
        "task",
        "group",
        "dataset",
        "class_label",
        "sample_id",
        "image_number",
        "raw_path",
        "selected_recon_path",
        "fold",
    ]
    write_csv(output_path, canonical_rows, fieldnames)
    return canonical_rows


def main():
    parser = argparse.ArgumentParser(description="Build a canonical paired manifest with manual camo selections applied.")
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_canonical_manifest(
        paired_manifest_path=args.paired_manifest,
        selections_path=args.selections,
        output_path=args.output,
        n_splits=args.folds,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
