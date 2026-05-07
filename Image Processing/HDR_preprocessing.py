from pathlib import Path
import shutil

# -----------------------------
# USER SETTINGS
# -----------------------------
ROOT = Path(r"D:\Trees 2.1")                 # contains hdr_seq_YYYYMMDD_HHMMSS folders
OUT = Path(r"D:\Trees_2.1_HDR")              # new destination root

DO_MOVE = True   # True = move files, False = copy files

# Expected filenames inside each hdr_seq_* folder
CUBERT_NAME = "hdr_result_cubert.tif"
THORLABS_NAME = "hdr_result_thorlabs.tif"

# Folder prefix pattern
SEQ_PREFIX = "hdr_seq_"

# -----------------------------
# OUTPUT FOLDERS
# -----------------------------
OUT_CUBERT = OUT / "cubert"
OUT_THORLABS = OUT / "thorlabs"
OUT_CUBERT.mkdir(parents=True, exist_ok=True)
OUT_THORLABS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# HELPERS
# -----------------------------
def transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if DO_MOVE:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

# -----------------------------
# MAIN
# -----------------------------
seq_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(SEQ_PREFIX)])

print(f"Found {len(seq_dirs)} seq folders under: {ROOT}")
moved_cubert = 0
moved_thorlabs = 0
skipped = 0

for d in seq_dirs:
    # d.name example: hdr_seq_20260201_170359
    suffix = d.name[len(SEQ_PREFIX):]  # -> 20260201_170359

    cubert_src = d / CUBERT_NAME
    thorlabs_src = d / THORLABS_NAME

    if cubert_src.exists():
        cubert_dst = OUT_CUBERT / f"cubert_{suffix}.tif"
        transfer(cubert_src, cubert_dst)
        moved_cubert += 1
        print(f"[cubert]   {cubert_src}  ->  {cubert_dst}")
    else:
        print(f"[missing]  cubert not found in {d}")
        skipped += 1

    if thorlabs_src.exists():
        thorlabs_dst = OUT_THORLABS / f"thorlabs_{suffix}.tif"
        transfer(thorlabs_src, thorlabs_dst)
        moved_thorlabs += 1
        print(f"[thorlabs] {thorlabs_src} ->  {thorlabs_dst}")
    else:
        print(f"[missing]  thorlabs not found in {d}")
        skipped += 1

print("\nDone.")
print(f"  cubert moved/copied:   {moved_cubert}")
print(f"  thorlabs moved/copied: {moved_thorlabs}")
print(f"  missing entries:       {skipped}")
print(f"Output folder: {OUT}")
