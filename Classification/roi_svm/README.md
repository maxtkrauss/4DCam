# ROI SVM Classification

This module contains the classification code used for the current paper-ready reconstructed-modality results exported by Al on 2026-05-05.

## What It Runs

`train_modalities_roi_svm.py` trains five-modality linear SVM classifiers for:

- `textiles`: image-level result is the recommended paper number.
- `camo`: pixel-majority result is the recommended paper number.

The reconstructed modalities are:

- `recon_1ch_mean_all`
- `recon_4ch_pol_mean_wavelength`
- `recon_106ch_wavelength_mean_pol`
- `recon_424ch_full`

The native scatterogram comparator is:

- `native_scatter_4ch`

## Reference Results

Compact exported results live under `reference_results/`:

- `reference_results/results/`: fold and modality CSVs from the 2026-05-05 export.
- `reference_results/tables/`: paper-ready CSV and LaTeX tables.
- `reference_results/plots/`: monotonic trend figures.

## Dependencies

```bash
pip install -r requirements.txt
```

## Example Commands

Textiles, iter6-style paper result:

```bash
python train_modalities_roi_svm.py ^
  --dataset textiles ^
  --output-root outputs/research_iter6_textiles_spatial2_aug ^
  --feature-mode spatial2_mean_std ^
  --pixel-c-selection val_majority ^
  --augment-train-copies 4 ^
  --augment-gain-range 0.08 ^
  --augment-channel-gain-range 0.05 ^
  --augment-offset-range 0.03 ^
  --augment-noise-std 0.01
```

Camo, iter6-style paper result:

```bash
python train_modalities_roi_svm.py ^
  --dataset camo ^
  --output-root outputs/research_iter6_camo_spatial2_aug ^
  --feature-mode spatial2_mean_std ^
  --pixel-c-selection val_majority ^
  --augment-train-copies 4 ^
  --augment-gain-range 0.08 ^
  --augment-channel-gain-range 0.05 ^
  --augment-offset-range 0.03 ^
  --augment-noise-std 0.01
```

Use the dataset-root overrides if your data is not in Al's original local paths:

```bash
python train_modalities_roi_svm.py --dataset textiles --output-root outputs/textiles ^
  --textiles-recon-root D:\path\to\flat_textiles ^
  --textiles-native-root D:\path\to\textiles
```

```bash
python train_modalities_roi_svm.py --dataset camo --output-root outputs/camo ^
  --camo-recon-root D:\path\to\flat_camo_plant ^
  --camo-native-root D:\path\to\Camo_Plants
```

## Notes

The script expects 424-channel reconstruction TIFFs and native 4-or-more-channel TIFFs. Textiles use fixed ROI crops matching the export; camo uses full images.
