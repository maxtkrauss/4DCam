# Paper Edit Map

## Main Text

Use:
[paper_main_updated/Main_text_classification_updated.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/paper_main_updated/Main_text_classification_updated.tex)

Classification-specific changes:

- Abstract:
  - Removed the MobileNet/direct-compressed-measurement claim.
  - Replaced it with the monotonic reconstructed-channel results.
- Fabric classification section:
  - Replaced the old spectral-vs-polarization-vs-fused wording with 1/4/106/424 reconstructed modality results.
  - Replaced the figure with a new monotonic trend plot.
- Camouflage classification section:
  - Replaced the old fused-vs-direct-MobileNet wording with the monotonic pixel-majority trend.
  - Replaced the figure with a new monotonic trend plot.
- Conclusion:
  - Replaced the reconstruction-free classification conclusion with the increasing-channel-count conclusion.

## Supplement

Use:
[paper_supplement_updated/main_edited_classification_updated.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/paper_supplement_updated/main_edited_classification_updated.tex)

Classification-specific changes:

- Added new `S12. Classification on reconstructed modalities`.
- Documented:
  - dataset sizes
  - modality construction
  - per-image normalization
  - textile crop
  - linear-SVM feature construction
  - augmentation
  - image-level and pixel-majority readouts
- Added the summary figure and the two paper-result tables.

## Figures Added

- Main paper:
  - `paper_main_updated/figures/Textiles_monotonic_updated.png`
  - `paper_main_updated/figures/Camo_monotonic_updated.png`
- Supplement:
  - `paper_supplement_updated/recommended_monotonic_summary.png`

## Tables Added

- `tables/recommended_paper_results.csv`
- `tables/recommended_paper_results.tex`
- `tables/textiles_image_accuracy_summary.csv`
- `tables/textiles_image_accuracy_summary.tex`
- `tables/camo_pixel_majority_accuracy_summary.csv`
- `tables/camo_pixel_majority_accuracy_summary.tex`
