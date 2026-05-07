CHPC Pathing

Dataset root on CHPC:
- `/scratch/general/nfs1/u1528328/Textile-Camo Classification`

The manifests in this repo store Windows-style dataset paths under:
- `D:\4DCam_Data\Textile-Camo Classification`

The experiment runners now support automatic remapping with `--dataset-root`.

Image-based MobileNet runs:
```bash
cd /path/to/4DCam/Classification/Mobile_Net
python run_regime_experiments.py \
  --dataset-root "/scratch/general/nfs1/u1528328/Textile-Camo Classification" \
  --task all \
  --regime all \
  --folds all \
  --epochs 20 \
  --batch-size 1 \
  --lr 3e-4 \
  --weight-decay 1e-4
```

ROI-feature runs:
```bash
cd /path/to/4DCam/Classification/Mobile_Net
python run_roi_feature_experiments.py \
  --dataset-root "/scratch/general/nfs1/u1528328/Textile-Camo Classification" \
  --task all \
  --regime all \
  --folds all \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

If you want to regenerate reports on CHPC:
```bash
python generate_results_report.py --results-root ./regime_results
python generate_results_report.py --results-root ./roi_feature_results
```
