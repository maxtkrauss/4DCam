# Classification Export Package

This folder packages the classification work that should replace the older MobileNet-focused manuscript text.

## Start Here

1. Main paper update:
   [paper_main_updated/Main_text_classification_updated.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/paper_main_updated/Main_text_classification_updated.tex)
2. Supplement update:
   [paper_supplement_updated/main_edited_classification_updated.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/paper_supplement_updated/main_edited_classification_updated.tex)
3. Full handoff report:
   [report/classification_export_report.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/report/classification_export_report.tex)
4. Quick edit map:
   [notes/paper_edit_map.md](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/notes/paper_edit_map.md)

## Recommended Paper Story

- Replace the old MobileNet/direct-scatterogram classification narrative.
- Emphasize monotonic improvement with increasing reconstructed channel count.
- Use:
  - `textiles`: image-level linear-SVM result
  - `camo`: pixel-majority linear-SVM result
- Keep native 4-channel scatterogram as a separate comparator, not part of the reconstructed-channel monotonic trend.

## Key Numbers

- Textiles, reconstructed image-level accuracy:
  - `1ch`: `58.7 ± 8.0%`
  - `4ch`: `60.5 ± 6.2%`
  - `106ch`: `75.2 ± 10.7%`
  - `424ch`: `80.2 ± 8.6%`
- Camo, reconstructed pixel-majority accuracy:
  - `1ch`: `76.0 ± 7.4%`
  - `4ch`: `76.0 ± 7.4%`
  - `106ch`: `87.0 ± 6.7%`
  - `424ch`: `89.0 ± 5.5%`

## Important Files

- Main plots:
  - [plots/textiles_reconstructed_image_monotonic.png](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/plots/textiles_reconstructed_image_monotonic.png)
  - [plots/camo_reconstructed_pixel_monotonic.png](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/plots/camo_reconstructed_pixel_monotonic.png)
  - [plots/recommended_monotonic_summary.png](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/plots/recommended_monotonic_summary.png)
- Paper tables:
  - [tables/recommended_paper_results.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/tables/recommended_paper_results.csv)
  - [tables/recommended_paper_results.tex](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/tables/recommended_paper_results.tex)
- Source code:
  - [code/train_modalities_roi_svm.py](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/code/train_modalities_roi_svm.py)
- Source results:
  - [results/textiles_iter6_modality_results.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/results/textiles_iter6_modality_results.csv)
  - [results/camo_iter6_modality_results.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/results/camo_iter6_modality_results.csv)
  - [results/conversation_results_condensed.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/max_export/classification_export_20260505/results/conversation_results_condensed.csv)

## If A Single Unified Deep Model Is Required

The only deep architecture that was monotonic on both datasets in the same run was `polar_factorized_attn`, but it was much weaker on textiles. It is documented in the report, but it is not the recommended paper replacement.
