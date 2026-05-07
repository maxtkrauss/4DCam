import argparse
import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This tool requires the 'tifffile' package.") from exc


DEFAULT_MANIFEST = Path(r"Z:\4DCam\Software\paired_textile_camo_manifest_2026-04-09.csv")
DEFAULT_SELECTIONS = Path(r"Z:\4DCam\Software\double_pair_manual_selections_2026-04-09.csv")
PREVIEW_SIZE = (420, 420)


def read_manifest(manifest_path: Path):
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    double_rows = []
    for row in rows:
        try:
            match_count = int(row.get("recon_match_count", "0") or "0")
        except ValueError:
            match_count = 0
        if row.get("group") != "camo" or match_count != 2:
            continue

        recon_paths = [part.strip() for part in row.get("recon_paths", "").split(";") if part.strip()]
        if len(recon_paths) != 2:
            continue
        row["candidate_paths"] = recon_paths
        double_rows.append(row)

    double_rows.sort(key=lambda item: (item.get("dataset", ""), int(item.get("image_number", "0"))))
    return double_rows


def load_existing_selections(output_path: Path):
    if not output_path.exists():
        return {}

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selections = {}
    for row in rows:
        key = build_key(row)
        selections[key] = row
    return selections


def build_key(row):
    return f"{row.get('dataset','')}|{row.get('image_number','')}|{row.get('source_path','')}"


def load_tiff(path: Path):
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    return arr


def to_panchromatic(arr: np.ndarray):
    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        arr = arr.astype(np.float32)
        # Handle both channel-first cubes (C, H, W) and channel-last images (H, W, C).
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return np.mean(arr, axis=0)
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return np.mean(arr, axis=-1)
        # Fall back to collapsing the smallest axis as the spectral/channel axis.
        collapse_axis = int(np.argmin(arr.shape))
        return np.mean(arr, axis=collapse_axis)

    if arr.ndim >= 4:
        arr = arr.astype(np.float32)
        # Reconstructions can arrive as stacks such as (pol, spec, H, W).
        # Collapse every non-spatial dimension to produce a 2D panchromatic preview.
        return np.mean(arr, axis=tuple(range(arr.ndim - 2)))

    raise ValueError(f"Unsupported TIFF shape: {arr.shape}")


def normalize_image(img: np.ndarray):
    img = np.asarray(img, dtype=np.float32)
    finite_mask = np.isfinite(img)
    if not finite_mask.any():
        return np.zeros_like(img, dtype=np.uint8)

    valid = img[finite_mask]
    lo = float(np.percentile(valid, 1))
    hi = float(np.percentile(valid, 99))
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    scaled = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def render_preview(path: Path):
    arr = load_tiff(path)
    pan = to_panchromatic(arr)
    norm = normalize_image(pan)
    image = Image.fromarray(norm, mode="L")
    image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


class ManualSelectorApp:
    def __init__(self, root: tk.Tk, rows, output_path: Path):
        self.root = root
        self.rows = rows
        self.output_path = output_path
        self.selections = load_existing_selections(output_path)
        self.index = 0
        self.preview_refs = []

        self.root.title("Manual Reconstruction Match Selector")
        self.root.geometry("1400x900")

        self.header_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.choice_var = tk.IntVar(value=-1)

        self._build_layout()
        self._jump_to_first_unselected()
        self._render_current()

    def _build_layout(self):
        main = tk.Frame(self.root, padx=12, pady=12)
        main.pack(fill="both", expand=True)

        tk.Label(main, textvariable=self.header_var, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(main, textvariable=self.status_var, justify="left", anchor="w").pack(anchor="w", pady=(4, 12))

        image_frame = tk.Frame(main)
        image_frame.pack(fill="both", expand=True)

        self.source_panel = self._make_panel(image_frame, "Raw Thorlabs Source")
        self.source_panel.grid(row=0, column=0, padx=6, sticky="nsew")

        self.cand1_panel = self._make_panel(image_frame, "Candidate A")
        self.cand1_panel.grid(row=0, column=1, padx=6, sticky="nsew")

        self.cand2_panel = self._make_panel(image_frame, "Candidate B")
        self.cand2_panel.grid(row=0, column=2, padx=6, sticky="nsew")

        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.columnconfigure(2, weight=1)

        button_row = tk.Frame(main, pady=10)
        button_row.pack(fill="x")

        tk.Button(button_row, text="Previous", command=self.prev_item, width=14).pack(side="left")
        tk.Button(button_row, text="Save Selection", command=self.save_current, width=16).pack(side="left", padx=8)
        tk.Button(button_row, text="Next", command=self.next_item, width=14).pack(side="left")
        tk.Button(button_row, text="Jump to First Unselected", command=self._jump_to_first_unselected, width=22).pack(side="left", padx=8)
        tk.Button(button_row, text="Export and Close", command=self.close_app, width=16).pack(side="right")

    def _make_panel(self, parent, title):
        frame = tk.LabelFrame(parent, text=title, padx=8, pady=8)
        label = tk.Label(frame)
        label.pack(pady=(0, 8))
        info = tk.Label(frame, justify="left", anchor="w", wraplength=380)
        info.pack(anchor="w")
        radio = None
        if "Candidate" in title:
            value = 0 if "A" in title else 1
            radio = tk.Radiobutton(frame, text=f"Select {title}", variable=self.choice_var, value=value)
            radio.pack(anchor="w", pady=(8, 0))
        frame.image_label = label
        frame.info_label = info
        frame.radio = radio
        return frame

    def _jump_to_first_unselected(self):
        for idx, row in enumerate(self.rows):
            if build_key(row) not in self.selections:
                self.index = idx
                break

    def _render_current(self):
        if not self.rows:
            self.header_var.set("No double-paired camo images found in the manifest.")
            self.status_var.set("")
            return

        row = self.rows[self.index]
        key = build_key(row)
        saved = self.selections.get(key)
        if saved:
            try:
                self.choice_var.set(int(saved["selected_candidate_index"]))
            except Exception:
                self.choice_var.set(-1)
        else:
            self.choice_var.set(-1)

        self.header_var.set(
            f"Item {self.index + 1} of {len(self.rows)} | {row['dataset']} | image_{row['image_number']}"
        )
        self.status_var.set(
            f"Source: {row['source_path']}\n"
            f"Saved choice: {saved['selected_recon_path'] if saved else 'none'}"
        )

        source_path = Path(row["source_path"])
        cand_paths = [Path(path) for path in row["candidate_paths"]]

        self.preview_refs = [
            render_preview(source_path),
            render_preview(cand_paths[0]),
            render_preview(cand_paths[1]),
        ]

        self.source_panel.image_label.configure(image=self.preview_refs[0])
        self.source_panel.info_label.configure(text=source_path.as_posix())

        self.cand1_panel.image_label.configure(image=self.preview_refs[1])
        self.cand1_panel.info_label.configure(text=cand_paths[0].as_posix())

        self.cand2_panel.image_label.configure(image=self.preview_refs[2])
        self.cand2_panel.info_label.configure(text=cand_paths[1].as_posix())

    def save_current(self):
        if self.choice_var.get() not in (0, 1):
            messagebox.showwarning("Selection Required", "Choose Candidate A or Candidate B before saving.")
            return

        row = self.rows[self.index]
        selected_idx = self.choice_var.get()
        selected_path = row["candidate_paths"][selected_idx]
        key = build_key(row)
        self.selections[key] = {
            "group": row["group"],
            "dataset": row["dataset"],
            "class": row["class"],
            "image_number": row["image_number"],
            "source_name": row["source_name"],
            "source_path": row["source_path"],
            "candidate_a_path": row["candidate_paths"][0],
            "candidate_b_path": row["candidate_paths"][1],
            "selected_candidate_index": str(selected_idx),
            "selected_recon_path": selected_path,
        }
        self._write_selections()
        self.status_var.set(
            f"Source: {row['source_path']}\nSaved choice: {selected_path}"
        )

    def _write_selections(self):
        fieldnames = [
            "group",
            "dataset",
            "class",
            "image_number",
            "source_name",
            "source_path",
            "candidate_a_path",
            "candidate_b_path",
            "selected_candidate_index",
            "selected_recon_path",
        ]
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(self.selections):
                writer.writerow(self.selections[key])

    def next_item(self):
        if not self.rows:
            return
        if self.index < len(self.rows) - 1:
            self.index += 1
            self._render_current()

    def prev_item(self):
        if not self.rows:
            return
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def close_app(self):
        self._write_selections()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Review double-paired camo reconstruction matches and select the correct mapping manually."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to the paired manifest CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SELECTIONS,
        help="CSV path for saved manual selections.",
    )
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    root = tk.Tk()
    app = ManualSelectorApp(root, rows, args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
