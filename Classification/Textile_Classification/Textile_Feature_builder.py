#!/usr/bin/env python3
"""
Build features from ALL images (no splits).

Choose polarimetric representation:
  - POL_MODE = "4": Pol = [m0, m45, m90, m135] (panchromatic ROI means)
                    SpecPol = per-band [m0, m45, m90, m135] (concat over bands)
  - POL_MODE = "3": Pol = [S0, S1/S0, S2/S0]
                    SpecPol = per-band [S0, S1/S0, S2/S0]

Outputs (in OUTDIR):
  X_spec_all.npy        # (N, B)      spectral per band, averaged over pols
  X_pol{POL_MODE}_all.npy
  X_specpol{POL_MODE}_all.npy
  y_all.npy
  order_all.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread

# ====== USER SETTINGS ======
PATHS_CSV = Path("/scratch/general/nfs1/u1528328/img_dir/textile_aggregate/paths_textile_FIXED.csv")
OUTDIR    = Path("/uufs/chpc.utah.edu/common/home/u1528328/Classification/Textile_Classification/features_all")
ROI        = 32
KEEP_BANDS = 106
EPS        = 1e-8

# Choose "4" to match verbal spec; set to "3" for normalized Stokes
POL_MODE   = "4"   # "4" or "3"
# ===========================

def ensure_ch_first(arr: np.ndarray) -> np.ndarray:
    """Return (B,H,W) from either (B,H,W) or (H,W,B)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    # If first dim looks like bands, keep; if last dim looks like bands, moveaxis
    if arr.shape[0] in (KEEP_BANDS, 212) or (arr.shape[0] > 16 and arr.shape[0] >= arr.shape[-1]):
        return arr
    if arr.shape[-1] in (KEEP_BANDS, 212) or (arr.shape[-1] > 16 and arr.shape[-1] > arr.shape[0]):
        return np.moveaxis(arr, -1, 0)
    # Fallback: if first dim > last dim, assume (B,H,W)
    return arr if arr.shape[0] >= arr.shape[-1] else np.moveaxis(arr, -1, 0)

def crop_center_shift_up(arr: np.ndarray, roi: int) -> np.ndarray:
    """Crop (C,H,W) to ROI×ROI: center horizontally, shift up by ROI/2 (clamped)."""
    C, H, W = arr.shape
    h = w = min(roi, H, W)
    y0 = (H - h) // 2 - (h // 2)
    x0 = (W - w) // 2
    y0 = max(0, min(y0, H - h))
    x0 = max(0, min(x0, W - w))
    return arr[:, y0:y0 + h, x0:x0 + w]

def pol3_from_means(m0: float, m45: float, m90: float, m135: float) -> np.ndarray:
    """[S0, S1/S0, S2/S0] from scalar per-pol ROI means."""
    S0 = m0 + m90
    S1 = m0 - m90
    S2 = m45 - m135
    return np.array([S0, S1 / (S0 + EPS), S2 / (S0 + EPS)], dtype=np.float32)

def pol4_from_means(m0: float, m45: float, m90: float, m135: float) -> np.ndarray:
    """[m0, m45, m90, m135] from scalar per-pol ROI means."""
    return np.array([m0, m45, m90, m135], dtype=np.float32)

def pol14_from_means(m0: float, m45: float, m90: float, m135: float) -> np.ndarray:
    """
    Return 14 polarimetric features from per-pol ROI means (m0, m45, m90, m135):

      [ m0, m45, m90, m135,             # 4 raw means
        S0, S1, S2,                     # 3 Stokes-like surrogates
        DoLP, sin2AoLP, cos2AoLP,       # 3 polarization descriptors
        m0/S0, m45/S0, m90/S0, m135/S0  # 4 angle-normalized means
      ]
    """
    S0 = m0 + m90
    S1 = m0 - m90
    S2 = m45 - m135

    amp = np.sqrt(S1**2 + S2**2)
    DoLP = amp / (S0 + EPS)
    sin2AoLP = S2 / (amp + EPS)
    cos2AoLP = S1 / (amp + EPS)

    return np.array([
        m0, m45, m90, m135,          # 4
        S0, S1, S2,                  # +3 = 7
        DoLP, sin2AoLP, cos2AoLP,    # +3 = 10
        m0/(S0 + EPS), m45/(S0 + EPS), m90/(S0 + EPS), m135/(S0 + EPS)  # +4 = 14
    ], dtype=np.float32)


def load_cubes(row):
    """Load four (B,H,W) cubes; truncate to KEEP_BANDS; guard against GT paths."""
    paths = [row["recon_pol0"], row["recon_pol45"], row["recon_pol90"], row["recon_pol135"]]
    for p in paths:
        ps = str(p).lower()
        if "cb_raw_" in ps:
            raise ValueError(f"GT path used in recon field: {p}")
        if "tl_gen_" not in ps:
            print(f"[WARN] recon path does not look like tl_gen_*: {p}")

    cubes = []
    for p in paths:
        cube = imread(p)
        cube = ensure_ch_first(cube).astype(np.float32)
        cube = cube[:KEEP_BANDS] if cube.shape[0] >= KEEP_BANDS else cube
        if cube.ndim != 3:
            raise ValueError(f"Expected (B,H,W), got {cube.shape} at {p}")
        cubes.append(cube)
    return cubes

def build_features_for_row(row):
    """
    Returns:
      X_spec     : (B,)       per-band mean over ROI, averaged over pols
      X_pol*     : (3,) or (4,) broadband polarimetric features
      X_specpol* : (B*3,) or (B*4,) per-band polarimetric features
    """
    cubes = load_cubes(row)  # list of 4 arrays: pol0,45,90,135

    # --- SPEC: per band, ROI mean per pol -> average over pols
    per_pol_spectra = []
    for cube in cubes:
        crop = crop_center_shift_up(cube, ROI)                  # (B,roi,roi)
        band_means = crop.reshape(crop.shape[0], -1).mean(1)    # (B,)
        per_pol_spectra.append(band_means)
    X_spec = np.mean(np.stack(per_pol_spectra, 0), 0).astype(np.float32)  # (B,)

    # --- POL (broadband): ROI-mean per pol on the broadband image
    mvals = []
    for cube in cubes:
        broad = cube.mean(axis=0)                                # (H,W)
        crop = crop_center_shift_up(broad[None, ...], ROI)[0]    # (roi,roi)
        mvals.append(float(crop.mean()))
    if POL_MODE == "4":
        X_pol = pol4_from_means(*mvals)                          # (4,)
    elif POL_MODE == "14":
        X_pol = pol14_from_means(*mvals)                          # (4,)
    else:
        X_pol = pol3_from_means(*mvals)                          # (3,)

    # --- SPECPOL (per-band): ROI-mean per pol per band -> 3 or 4 features per band
    chunks = []
    B = X_spec.shape[0]
    for b in range(B):
        m_b = []
        for cube in cubes:
            band_img = cube[b][None, ...]                        # (1,H,W)
            crop = crop_center_shift_up(band_img, ROI)[0]        # (roi,roi)
            m_b.append(float(crop.mean()))
        chunks.append(pol4_from_means(*m_b) if POL_MODE == "4" else pol3_from_means(*m_b))
    X_specpol = np.concatenate(chunks, 0).astype(np.float32)     # (B*4,) or (B*3,)

    return X_spec, X_pol, X_specpol

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PATHS_CSV)

    need = {"swatch_id","class","recon_pol0","recon_pol45","recon_pol90","recon_pol135"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    keep_cols = ["swatch_id","class"] + (["fold"] if "fold" in df.columns else [])
    df = df.sort_values(keep_cols).reset_index(drop=True)

    classes = sorted(df["class"].unique())
    class_to_int = {c:i for i,c in enumerate(classes)}
    print("[INFO] class_to_int:", class_to_int)

    Xs_list, Xp_list, Xsp_list, y_list = [], [], [], []
    kept, skipped = 0, 0
    for _, r in df.iterrows():
        try:
            xs, xp, xsp = build_features_for_row(r)
        except Exception as e:
            print(f"[WARN] Skipping row due to error: {e}")
            skipped += 1
            continue
        Xs_list.append(xs); Xp_list.append(xp); Xsp_list.append(xsp)
        y_list.append(class_to_int[r["class"]]); kept += 1

    if kept == 0:
        raise SystemExit("No rows processed. Check file paths in CSV.")

    X_spec     = np.stack(Xs_list)                  # (N, B)
    X_pol      = np.stack(Xp_list)                  # (N, 3 or 4)
    X_specpol  = np.stack(Xsp_list)                 # (N, B*3 or B*4)
    y          = np.asarray(y_list, dtype=np.int64)

    # Filenames reflect the mode
    np.save(OUTDIR / "X_spec_all.npy", X_spec)
    np.save(OUTDIR / f"X_pol{POL_MODE}_all.npy", X_pol)
    np.save(OUTDIR / f"X_specpol{POL_MODE}_all.npy", X_specpol)
    np.save(OUTDIR / "y_all.npy", y)

    meta_cols = ["swatch_id","class"] + (["fold"] if "fold" in df.columns else [])
    df[meta_cols].to_csv(OUTDIR / "order_all.csv", index=False)

    B = X_spec.shape[1]
    print(f"[OK] Features saved to {OUTDIR}  |  kept={kept}, skipped={skipped}")
    print(f"SPEC        : {X_spec.shape}   (B={B})")
    print(f"POL-{POL_MODE}     : {X_pol.shape}")
    print(f"SPECPOL-{POL_MODE} : {X_specpol.shape}  (B*{POL_MODE}={B*int(POL_MODE)})")
    print(f"Labels (y)  : {y.shape}  | classes: {classes}")

if __name__ == "__main__":
    main()
