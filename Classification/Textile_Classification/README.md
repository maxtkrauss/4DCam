Textile Classification — README

Overview
--------
This repository contains code and precomputed features used for the textile classification experiment that compares three modalities:

- SPEC (spectral only)
- POL (broadband polarimetric features)
- SPECPOL (spectro-polarimetric features: polarimetric features for each spectral band)

The `features_all/` folder stores the generated numeric feature arrays and the corresponding labels/metadata needed for the classification models. This README explains what each file in `features_all/` is, and how to load them.

files in `features_all/`
------------------------
Note: "N" is the number of samples (swatches), and "B" is the number of spectral bands (106).

- `X_spec_all.npy`  : NumPy array with shape (N, B).
    - Per-row: the per-band spectral feature for a swatch (band means averaged over polarizations inside the ROI).
    - Typical shape: (N, 106) if 106 bands were kept.

- `X_pol3_all.npy` / `X_pol4_all.npy` : NumPy array with shape (N, 3) or (N, 4).
    - If `pol3` (3 features): columns are [S0, S1/S0, S2/S0], where S0,S1,S2 are Stokes-like surrogates derived from per-pol ROI means.
    - If `pol4` (4 features): columns are the four broadband polarization mean values [m0, m45, m90, m135].

- `X_specpol3_all.npy` / `X_specpol4_all.npy` : NumPy array with shape (N, B*3) or (N, B*4).
    - Per-band polarimetric features concatenated across bands. e.g., for `pol3`, for every band you have 3 polarimetric features, resulting in `B * 3` columns.

- `X_pol3_scatter.npy`, `X_pol4_scatter.npy`, `X_specpol3_scatter.npy`, etc., and `X_pol3_scatter.npy` : These are the same types of features but for the raw scatterograms. They pair with `order_scatter.csv` and `y_all_scatterograms.npy` for those experiments.

- `y_all.npy` : NumPy 1-D integer array with shape (N,). These are integer labels for each row in the corresponding `X_*.npy` arrays.

- `y_all_scatterograms.npy` : Labels for the scatterogram variants (paired with `*_scatter.npy` files).

- `order_all.csv` : CSV table (N rows) listing `swatch_id`, `class` (human-readable class name), and optionally `fold` if folds were pre-assigned. Use this to map rows to swatch IDs and class names.

- `order_scatter.csv` : Same as above but for the scatterograms dataset/subset.

- `classes_scatter.txt` : Text file containing class names (one per line) used in the scatter experiment (if present).

Quick commands: inspect files (copy/paste)
-----------------------------------------
Open a Python REPL, or run this as a small script. These commands only use NumPy and will not change any files:

```python
import numpy as np
X_spec = np.load('features_all/X_spec_all.npy')
X_pol3 = np.load('features_all/X_pol3_all.npy')
y = np.load('features_all/y_all.npy', allow_pickle=True)
print('X_spec', X_spec.shape)
print('X_pol3', X_pol3.shape)
print('y', y.shape)
print('example row (spec):', X_spec[0][:8])  # first 8 band values
```

How shapes map to modalities (simple rules)
-------------------------------------------
If a feature vector has
- length 106 -> it is a spectral vector (SPEC) with 106 bands; shape (N,106)
- length 3 or 4 -> broadband POL features (N,3) or (N,4)
- length 318 (== 3*106) or 424 (== 4*106) -> spectro-polarimetric SPECPOL features

The training script `Textile_ResNet.py` contains the helper `infer_channels_and_bands()` to automatically interpret the number of channels/bands.

How the features were made
-------------------------
The script `Textile_Feature_builder.py` creates `features_all/`. Key points:
- It assumes you have a CSV (which maps to my CHPC directories) giving paths to reconstructions for four polarization angles (0,45,90,135) per swatch.
- For each swatch, it:
  - crops a centered ROI and computes per-band means to produce `X_spec` (average over polarizations)
  - computes broadband (ROI mean of the band-averaged image) polarimetric features for `X_pol` (mode `3` or `4` selected by `POL_MODE` at the top of the script)
  - computes per-band polarimetric features (concatenated) for `X_specpol`.
- Output numpy arrays and `order_all.csv` metadata are saved to `OUTDIR`.

How to regenerate features on new data:
--------------------------
Edit the constants at the top of `Textile_Feature_builder.py` (especially `PATHS_CSV`, `OUTDIR`, `POL_MODE`, and `KEEP_BANDS`) and then run:
(for the paper's current data, keep the paths the same are run the following on a CHPC node)

```bash
python Textile_Feature_builder.py
```

Required Python packages: `numpy`, `pandas`, `tifffile` (the script also uses standard library `pathlib`).

How to regenerate the final classification results
-------------------------------------------
The training + plotting script `Textile_ResNet.py`:
- loads `X_spec_all.npy`, then attempts to load `X_pol4_all.npy` (if missing it falls back to `X_pol3_all.npy`) and similarly for specpol
- runs 5-fold stratified cross-validation, trains a small CNN-like encoder, and computes out-of-fold predictions for each modality
- saves a combined plot called `Textile_Triple_CM.png` (three confusion matrices side-by-side) in the current working directory

Run it with (will use GPU if available):

```bash
python Textile_ResNet.py
```

Reconstruction images (textile_aggregates) — role in feature creation
-----------------------------------------------------------------
There is a companion folder, `textile_aggregates/`, that contains example images from the reconstruction experiments. These saved images are a subset of the original dataset, and they are important because they are the actual images that were used to compute the features in `features_all/`.

- Folder layout (example):
    - `textile_aggregates/results/textile_fold1_pol0/validation_latest/images/`
    - and the same structure for `pol45`, `pol90`, `pol135`.

- What you will find and why it matters:
    - Each `images/` folder contains the **ground-truth (GT)** image files (named with `cb_raw_`) and the corresponding **reconstruction** images (named with `tl_gen_`).
    - These GT + reconstruction images were used to compute the ROI means and the per-band polarimetric quantities that become the rows in `X_spec_all.npy`, `X_pol*_all.npy`, and `X_specpol*_all.npy` (see `Textile_Feature_builder.py`). In other words, they are not just illustrative—these visuals are a saved subset of the actual inputs used to generate the feature matrices for sequential classification.

- Notes:
    - Only one fold (`fold1`) is included here (`textile_fold1_*`) to keep the archive small; the full experiment used 5 folds in total (four additional folds were not saved here due to space constraints).
    - The classifier training scripts use the feature arrays (numpy files) rather than the images directly; the images are provided so you can inspect the original inputs that produced those features and verify ROI cropping, reconstruction quality, or pick examples for figures.



