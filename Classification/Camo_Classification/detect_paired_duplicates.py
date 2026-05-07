import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This tool requires the 'tifffile' package.") from exc


DEFAULT_MANIFEST = Path(r"Z:\4DCam\Software\paired_textile_camo_manifest_2026-04-09.csv")
DEFAULT_SELECTIONS = Path(r"Z:\4DCam\Software\double_pair_manual_selections_2026-04-09.csv")
DEFAULT_REPORT_JSON = Path(r"Z:\4DCam\Software\paired_duplicate_report_2026-04-09.json")
DEFAULT_REPORT_CSV = Path(r"Z:\4DCam\Software\paired_duplicate_candidates_2026-04-09.csv")


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_key(row):
    return f"{row.get('dataset','')}|{row.get('image_number','')}|{row.get('source_path','')}"


def resolve_selected_pairs(manifest_rows, selection_rows):
    selection_by_key = {build_key(row): row for row in selection_rows}
    resolved = []
    unresolved = []

    for row in manifest_rows:
        if row.get("notes") != "paired":
            continue

        key = build_key(row)
        match_count = int(row.get("recon_match_count", "0") or "0")
        recon_paths = [part.strip() for part in row.get("recon_paths", "").split(";") if part.strip()]

        if match_count == 1 and recon_paths:
            selected_path = recon_paths[0]
        elif match_count == 2 and key in selection_by_key:
            selected_path = selection_by_key[key]["selected_recon_path"]
        else:
            unresolved.append(row)
            continue

        resolved.append(
            {
                "group": row["group"],
                "dataset": row["dataset"],
                "class": row["class"],
                "image_number": row["image_number"],
                "source_path": row["source_path"],
                "selected_recon_path": selected_path,
            }
        )

    return resolved, unresolved


def load_tiff(path: Path):
    arr = tifffile.imread(path)
    return np.squeeze(np.asarray(arr))


def to_panchromatic(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return np.mean(arr, axis=0)
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return np.mean(arr, axis=-1)
        return np.mean(arr, axis=int(np.argmin(arr.shape)))
    if arr.ndim >= 4:
        return np.mean(arr, axis=tuple(range(arr.ndim - 2)))
    raise ValueError(f"Unsupported TIFF shape: {arr.shape}")


def normalize_image(img: np.ndarray):
    img = np.asarray(img, dtype=np.float32)
    finite_mask = np.isfinite(img)
    if not finite_mask.any():
        return np.zeros_like(img, dtype=np.float32)

    valid = img[finite_mask]
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def compute_hash(path: Path):
    arr = load_tiff(path)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def build_entries(resolved_rows):
    entries = {"source": [], "reconstruction": []}
    cache = {}

    for row in resolved_rows:
        source_path = Path(row["source_path"])
        recon_path = Path(row["selected_recon_path"])
        for modality, path in (("source", source_path), ("reconstruction", recon_path)):
            if path not in cache:
                arr = load_tiff(path)
                pan = normalize_image(to_panchromatic(arr))
                cache[path] = {
                    "shape": tuple(pan.shape),
                    "hash": compute_hash(path),
                    "pan": pan,
                }
            meta = cache[path]
            entries[modality].append(
                {
                    "group": row["group"],
                    "dataset": row["dataset"],
                    "class": row["class"],
                    "image_number": row["image_number"],
                    "path": str(path),
                    "shape": meta["shape"],
                    "hash": meta["hash"],
                    "pan": meta["pan"],
                }
            )
    return entries


def find_exact_duplicates(items):
    by_hash = {}
    for item in items:
        by_hash.setdefault(item["hash"], []).append(item)

    return [group for group in by_hash.values() if len(group) > 1]


def find_near_duplicates(items, mae_threshold):
    candidates = []
    for left, right in combinations(items, 2):
        if left["path"] == right["path"]:
            continue
        if left["shape"] != right["shape"]:
            continue
        mae = float(np.mean(np.abs(left["pan"] - right["pan"])))
        if mae <= mae_threshold:
            candidates.append(
                {
                    "dataset_a": left["dataset"],
                    "image_number_a": left["image_number"],
                    "path_a": left["path"],
                    "dataset_b": right["dataset"],
                    "image_number_b": right["image_number"],
                    "path_b": right["path"],
                    "shape": left["shape"],
                    "mae": mae,
                }
            )
    candidates.sort(key=lambda item: item["mae"])
    return candidates


def write_candidate_csv(path: Path, rows):
    fieldnames = [
        "modality",
        "dataset_a",
        "image_number_a",
        "path_a",
        "dataset_b",
        "image_number_b",
        "path_b",
        "shape",
        "mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_report(resolved_rows, unresolved_rows, entries, source_near, recon_near):
    exact_source = find_exact_duplicates(entries["source"])
    exact_recon = find_exact_duplicates(entries["reconstruction"])
    return {
        "resolved_pairs": len(resolved_rows),
        "unresolved_pairs": len(unresolved_rows),
        "exact_duplicate_groups": {
            "source": [
                {
                    "hash": group[0]["hash"],
                    "count": len(group),
                    "paths": [item["path"] for item in group],
                }
                for group in exact_source
            ],
            "reconstruction": [
                {
                    "hash": group[0]["hash"],
                    "count": len(group),
                    "paths": [item["path"] for item in group],
                }
                for group in exact_recon
            ],
        },
        "near_duplicate_counts": {
            "source": len(source_near),
            "reconstruction": len(recon_near),
        },
        "top_near_duplicates": {
            "source": source_near[:20],
            "reconstruction": recon_near[:20],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect exact and near-duplicate images in the resolved paired textile/camo dataset."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--mae-threshold", type=float, default=0.01)
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    selection_rows = read_csv(args.selections)
    resolved_rows, unresolved_rows = resolve_selected_pairs(manifest_rows, selection_rows)
    entries = build_entries(resolved_rows)

    source_near = find_near_duplicates(entries["source"], args.mae_threshold)
    recon_near = find_near_duplicates(entries["reconstruction"], args.mae_threshold)

    write_candidate_csv(
        args.report_csv,
        [{"modality": "source", **row} for row in source_near]
        + [{"modality": "reconstruction", **row} for row in recon_near],
    )

    report = make_report(resolved_rows, unresolved_rows, entries, source_near, recon_near)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
