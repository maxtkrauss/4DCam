# Conversation Results Guide

## Main files

- Master CSV: [conversation_results_master.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/conversation_results_master.csv)
- Builder script: [build_conversation_results_index.py](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/build_conversation_results_index.py)

The master CSV contains all result tables collected during this Codex conversation:

- classical ROI-SVM / modality-SVM runs,
- research iteration summaries,
- deep architecture sweep summaries,
- deep architecture fold-level results,
- dense-TTA reevaluations,
- no-augmentation follow-ups,
- older SpectralFormer / ResNet pyramid result tables,
- smoke runs.

The export currently has `2363` rows.

## How to read the CSV

The most important columns are:

- `record_type`: what kind of table the row came from.
- `run_name`: the output directory the row came from.
- `dataset_guess`: `textiles` or `camo`.
- `model_family`: broad branch such as `modalities_roi_svm`, `deep_architecture_sweep`, `resnet50`, or `spectralformer_caf`.
- `modality`: one of the reconstructed channel modalities when present.
- `channels_guess`: inferred channel count when present.
- `image_accuracy_mean`, `pixel_majority_accuracy_mean`, `accuracy_mean`, `test_accuracy`: the main performance columns, depending on the source table.
- `acc_1ch`, `acc_4ch`, `acc_106ch`, `acc_424ch`: trend summary columns.
- `nondecreasing_with_channels`, `strictly_increasing_with_channels`, `r2_vs_log2_channels`: monotonic-trend diagnostics.

## Where to look first

### 1. Best classical SVM trends

Look at rows where:

- `record_type == research_iteration_trend_summary`

Primary source file:

- [generated_channel_trend_fit.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/research_iterations_summary/generated_channel_trend_fit.csv)

Most important conclusions from this table:

- Best paper-ready textile result is `iter6_textiles`, image-level SVM:
  - `0.5866 -> 0.6055 -> 0.7521 -> 0.8017`
  - monotonic: `True`
- Best classical camo monotonic result is the pixel-majority SVM branch:
  - baseline camo: `0.7600 -> 0.7600 -> 0.8900 -> 0.9000`
  - iter6 camo: `0.7600 -> 0.7600 -> 0.8700 -> 0.8900`

### 2. Deep overnight architecture trends

Look at rows where:

- `record_type == architecture_trend_summary`

Primary source files:

- [base sweep trend summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_overnight_20260504_163137/monotonic_trend_summary.csv)
- [high-capacity follow-up trend summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_high_capacity_20260504_230806/monotonic_trend_summary.csv)
- [no-augmentation follow-up trend summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_noaug_20260505_003917/monotonic_trend_summary.csv)
- [base dense-TTA reevaluation trend summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_overnight_20260504_163137_dense_tta/monotonic_trend_summary.csv)
- [high-capacity dense-TTA reevaluation trend summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_high_capacity_20260504_230806_dense_tta/monotonic_trend_summary.csv)

Most important conclusions from these tables:

- The only architecture that was strictly monotonic on both datasets in the same run was the base `polar_factorized_attn` run:
  - camo: `0.67 -> 0.74 -> 0.82 -> 0.84`
  - textiles: `0.4316 -> 0.4441 -> 0.4563 -> 0.5061`
  - this is clean but not strong enough on textiles.
- Best deep textile monotonic run is `patch_cnn_attn` with no augmentation:
  - `0.4258 -> 0.4877 -> 0.5869 -> 0.6612`
- Best deep camo monotonic run with the strongest 424-channel endpoint is `token_mixer_mil` from the base dense-TTA reevaluation:
  - `0.50 -> 0.68 -> 0.88 -> 0.91`
  - caveat: low 1-channel and 4-channel accuracy.

### 3. Deep architecture mean accuracies by modality

Look at rows where:

- `record_type == architecture_modality_summary`

These rows are the easiest way to compare deep architectures without reasoning about monotonicity yet.

Useful source files:

- [base sweep modality summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_overnight_20260504_163137/architecture_modality_summary.csv)
- [high-capacity modality summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_high_capacity_20260504_230806/architecture_modality_summary.csv)
- [no-augmentation modality summary](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_noaug_20260505_003917/architecture_modality_summary.csv)

### 4. Per-fold diagnostics

Look at rows where:

- `record_type == architecture_fold_result`
- `record_type == run_results_table`

These are the rows to inspect if you want fold-level stability, not just fold means.

## Conclusions to draw

### Overall conclusion

There is still no single deep model that is both:

- strong on both textiles and camo, and
- strictly monotonic across `1 -> 4 -> 106 -> 424` channels on both datasets.

### Strongest current paper-ready conclusion

The strongest paper-ready results remain the classical SVM-based analyses:

- textiles: image-level SVM monotonic trend with good accuracy,
- camo: pixel-majority SVM monotonic trend with good accuracy.

That is still the cleanest and most defensible story today.

### Best deep-learning conclusion

The deep sweep did learn something useful:

- `patch_cnn_attn` is the best textile-side deep trend candidate.
- `token_mixer_mil` is the best camo-side deep trend candidate when reevaluated with dense TTA.
- `polar_factorized_attn` is the only deep model that clearly showed strict monotonicity on both datasets in one run, but it underperformed on textiles.

### What not to conclude

Do not conclude that we already found a unified deep winner for the paper. We did not.

The deep sweep narrowed the search, but the classical SVM branch still gives the strongest final numbers and the clearest monotonic story at this point.

## Shortlist to cite

If you want the shortest list of files to inspect:

1. [generated_channel_trend_fit.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/research_iterations_summary/generated_channel_trend_fit.csv)
2. [arch_sweep_overnight_20260504_163137/monotonic_trend_summary.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_overnight_20260504_163137/monotonic_trend_summary.csv)
3. [arch_sweep_followup_noaug_20260505_003917/monotonic_trend_summary.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_followup_noaug_20260505_003917/monotonic_trend_summary.csv)
4. [arch_sweep_overnight_20260504_163137_dense_tta/monotonic_trend_summary.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/outputs/arch_sweep_overnight_20260504_163137_dense_tta/monotonic_trend_summary.csv)
5. [conversation_results_master.csv](/mnt/wwn-0x5000c500b66356b8-part1/20251029_max_classification_dataset/patch_classification_code/conversation_results_master.csv)
