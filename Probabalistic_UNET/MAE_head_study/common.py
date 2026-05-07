import os
import random
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = [
    {
        'name': 'banknotes_augmented',
        'display_name': 'Banknotes',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/banknotes',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_banknotes_augmented_prob_retrain',
    },
    {
        'name': 'fossils&fauna',
        'display_name': 'Fossils/Flora',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/fossils&fauna',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_fossils&fauna_prob_retrain',
    },
    {
        'name': 'produce_augmented',
        'display_name': 'Produce',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/produce',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/Diff_nll_modelsmodels/checkpoints_produce_augmented_prob_retrain',
    },
    {
        'name': 'invertebrates_augmented',
        'display_name': 'Invertebrates',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/invertebrates',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_invertebrates_augmented_prob_retrain',
    },
    {
        'name': 'rescharts_augmented',
        'display_name': 'Resolution charts',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/rescharts',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_rescharts_augmented_prob_retrain',
    },
    {
        'name': 'cumulative_augmented',
        'display_name': 'Unified',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/cumulative_split',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_cumulative_augmented_prob_retrain',
    },
]

DEFAULT_EXPERT_DATASETS = [
    'banknotes_augmented',
    'fossils&fauna',
    'produce_augmented',
    'invertebrates_augmented',
    'rescharts_augmented',
]
DEFAULT_TIER1_DATASETS = DEFAULT_EXPERT_DATASETS + ['cumulative_augmented']

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / 'train.py'
TEST_SCRIPT = REPO_ROOT / 'test.py'
RECON_EVAL_SCRIPT = REPO_ROOT / 'HSI_comparison_probabalistic.py'
RECON_PER_IMAGE_SCRIPT = REPO_ROOT / 'HSI_comparison_probabalistic_per_image.py'
SCALAR_TRAIN_SCRIPT = REPO_ROOT / 'scalar_mae_reliability' / 'train_scalar_mae_predictor.py'
SCALAR_EVAL_SCRIPT = REPO_ROOT / 'scalar_mae_reliability' / 'evaluate_scalar_mae_predictor.py'

SCRATCH_RESULTS_DIR = '/scratch/general/nfs1/u1528328/paper_mae_head_study_results'
METRICS_DIR = '/uufs/chpc.utah.edu/common/home/u1528328/4DCam/Probabalistic_UNET/metrics_paper_mae_head_study'

TEST_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '212',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise',
    '--use_nll',
    '--lambda_l1', '0',
    '--norm', 'instance',
    '--no_dropout',
    '--eval',
    '--save_eval_metadata',
]


def dataset_by_name(name):
    for ds in DATASETS:
        if ds['name'] == name:
            return ds
    raise ValueError(f'Unknown dataset: {name}')


def sanitize(name):
    return str(name).replace('&', 'and').replace('_augmented', '').replace(' ', '_')


def model_name_for_dataset(dataset_name, pol):
    return f'{dataset_name}_pol{pol}'


def run_cmd(cmd):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")


def build_result_dir(scratch_root, model_name):
    return os.path.join(scratch_root, model_name, 'validation_latest')


def count_eval_images(eval_img_dir):
    if not os.path.isdir(eval_img_dir):
        return 0
    return len([f for f in os.listdir(eval_img_dir) if f.startswith('cb_raw_') and f.lower().endswith(('.tif', '.tiff'))])


def run_reconstruction_eval(result_dir, metrics_root):
    os.makedirs(metrics_root, exist_ok=True)
    eval_img_dir = os.path.join(result_dir, 'images')
    num_images = max(1, count_eval_images(eval_img_dir))
    metrics_csv = os.path.join(metrics_root, 'reconstruction_metrics.csv')
    per_image_csv = os.path.join(metrics_root, 'reconstruction_per_image.csv')
    run_cmd(['python', str(RECON_EVAL_SCRIPT), '--results_dir', eval_img_dir, '--num_images', str(num_images), '--metrics_csv', metrics_csv])
    run_cmd(['python', str(RECON_PER_IMAGE_SCRIPT), '--results_dir', eval_img_dir, '--num_images', str(num_images), '--per_image_csv', per_image_csv])
    return metrics_csv, per_image_csv


def split_result_dir(result_dir, train_dir, eval_dir, train_fraction=0.5, seed=0):
    result_dir = Path(result_dir)
    train_dir = Path(train_dir)
    eval_dir = Path(eval_dir)
    for out_dir in [train_dir, eval_dir]:
        (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    metadata_path = result_dir / 'prediction_metadata.csv'
    metadata_df = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    gt_files = sorted((result_dir / 'images').glob('cb_raw_*.tif'))
    if not gt_files:
        return 0, 0

    entries = []
    for idx, gt_path in enumerate(gt_files):
        suffix = gt_path.name.replace('cb_raw_', '')
        paths = [
            gt_path,
            result_dir / 'images' / f'tl_gen_{suffix}',
            result_dir / 'images' / f'tl_risk_{suffix}',
            result_dir / 'images' / f'tl_ood_{suffix}',
        ]
        paths = [p for p in paths if p.exists()]
        metadata_row = metadata_df.iloc[[idx]].copy() if idx < len(metadata_df) else None
        entries.append({'paths': paths, 'metadata_row': metadata_row})

    rng = random.Random(seed)
    rng.shuffle(entries)
    split_idx = int(round(len(entries) * float(train_fraction)))
    split_idx = min(max(split_idx, 1), len(entries) - 1) if len(entries) > 1 else len(entries)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]

    for subset_entries, subset_dir in [(train_entries, train_dir), (eval_entries, eval_dir)]:
        subset_metadata = []
        for entry in subset_entries:
            for src in entry['paths']:
                shutil.copy2(src, subset_dir / 'images' / src.name)
            if entry['metadata_row'] is not None:
                subset_metadata.append(entry['metadata_row'])
        if subset_metadata:
            pd.concat(subset_metadata, ignore_index=True).to_csv(subset_dir / 'prediction_metadata.csv', index=False)
        elif metadata_df.columns.tolist():
            pd.DataFrame(columns=metadata_df.columns).to_csv(subset_dir / 'prediction_metadata.csv', index=False)

    return len(train_entries), len(eval_entries)


def run_frozen_reconstructor(dataset, pol, scratch_root, force_is_ood_label=None):
    model_name = model_name_for_dataset(dataset['name'], pol)
    ckpt_dir = os.path.join(dataset['checkpoints_dir'], f'pol{pol}')
    cmd = [
        'python', str(TEST_SCRIPT),
        '--dataroot', dataset['dataroot'],
        '--name', model_name,
        '--checkpoints_dir', ckpt_dir,
        '--polarization', str(pol),
        '--results_dir', scratch_root,
    ] + TEST_OPTS
    if force_is_ood_label in (0, 1):
        cmd += ['--force_is_ood_label', str(int(force_is_ood_label))]
    run_cmd(cmd)
    return build_result_dir(scratch_root, model_name)


def matched_head_output_dir(tier_root, dataset_name, pol):
    return os.path.join(tier_root, f'{sanitize(dataset_name)}_pol{pol}')


def save_bar_summary(df, path, metric, title):
    if df.empty:
        return
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(plot_df)), plot_df[metric], color='#4c78a8')
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df['dataset_name'], rotation=25, ha='right')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
