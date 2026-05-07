import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_EXPERT_DATASETS,
    METRICS_DIR,
    SCRATCH_RESULTS_DIR,
    dataset_by_name,
    matched_head_output_dir,
    model_name_for_dataset,
    run_cmd,
    run_reconstruction_eval,
    sanitize,
    TEST_OPTS,
    TEST_SCRIPT,
    SCALAR_EVAL_SCRIPT,
)


def save_heatmap(matrix_df, value_col, title, output_path):
    if matrix_df.empty:
        return
    pivot = matrix_df.pivot(index='expert_display_name', columns='target_display_name', values=value_col)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha='right')
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Tier 2: cross-domain stress test for frozen expert reconstructors plus matched MAE heads.')
    parser.add_argument('--expert_datasets', nargs='*', default=DEFAULT_EXPERT_DATASETS)
    parser.add_argument('--target_datasets', nargs='*', default=DEFAULT_EXPERT_DATASETS)
    parser.add_argument('--polarizations', nargs='*', type=int, default=[0])
    parser.add_argument('--include_matched', action='store_true')
    parser.add_argument('--keep_generated_images', action='store_true')
    args = parser.parse_args()

    tier_metrics_root = os.path.join(METRICS_DIR, 'tier2_cross_domain')
    os.makedirs(tier_metrics_root, exist_ok=True)

    all_rows = []
    for pol in args.polarizations:
        for expert_name in args.expert_datasets:
            expert_ds = dataset_by_name(expert_name)
            expert_model_name = model_name_for_dataset(expert_ds['name'], pol)
            head_model_dir = matched_head_output_dir(os.path.join(METRICS_DIR, 'tier1_matched_heads', 'models'), expert_ds['name'], pol)
            head_checkpoint = os.path.join(head_model_dir, 'scalar_mae_predictor_best.pt')
            if not os.path.exists(head_checkpoint):
                raise FileNotFoundError(f'Matched Tier 1 MAE head not found for {expert_name} pol{pol}: {head_checkpoint}')

            for target_name in args.target_datasets:
                if not args.include_matched and target_name == expert_name:
                    continue
                target_ds = dataset_by_name(target_name)
                run_tag = f'expert_{sanitize(expert_name)}_pol{pol}__on__{sanitize(target_name)}'
                scratch_root = os.path.join(SCRATCH_RESULTS_DIR, 'tier2', run_tag)

                cmd = [
                    'python', str(TEST_SCRIPT),
                    '--dataroot', target_ds['dataroot'],
                    '--name', expert_model_name,
                    '--checkpoints_dir', os.path.join(expert_ds['checkpoints_dir'], f'pol{pol}'),
                    '--polarization', str(pol),
                    '--results_dir', scratch_root,
                ] + TEST_OPTS
                run_cmd(cmd)
                result_dir = os.path.join(scratch_root, expert_model_name, 'validation_latest')

                recon_metrics_root = os.path.join(tier_metrics_root, 'reconstruction', run_tag)
                recon_metrics_csv, _ = run_reconstruction_eval(result_dir, recon_metrics_root)
                recon_metrics = pd.read_csv(recon_metrics_csv).iloc[0].to_dict() if os.path.exists(recon_metrics_csv) else {}

                eval_output_dir = os.path.join(tier_metrics_root, 'evals', run_tag)
                eval_mode = 'matched' if target_name == expert_name else 'cross_domain'
                run_cmd([
                    'python', str(SCALAR_EVAL_SCRIPT),
                    '--checkpoint', head_checkpoint,
                    '--result_dirs', result_dir,
                    '--dataset_names', target_ds['name'],
                    '--eval_modes', eval_mode,
                    '--output_dir', eval_output_dir,
                    '--polarization', str(pol),
                    '--norm_bitwise',
                ])
                pooled_summary_path = os.path.join(eval_output_dir, 'scalar_mae_pooled_summary.csv')
                if os.path.exists(pooled_summary_path):
                    row = pd.read_csv(pooled_summary_path).iloc[0].to_dict()
                    row.update(recon_metrics)
                    row.update({
                        'expert_dataset': expert_ds['name'],
                        'expert_display_name': expert_ds['display_name'],
                        'expert_model_name': expert_model_name,
                        'target_dataset': target_ds['name'],
                        'target_display_name': target_ds['display_name'],
                        'eval_mode': eval_mode,
                        'polarization': pol,
                    })
                    all_rows.append(row)

                if not args.keep_generated_images and os.path.isdir(scratch_root):
                    shutil.rmtree(scratch_root)
                    print(f'Deleted generated images in {scratch_root}')

    summary_df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(tier_metrics_root, 'tier2_cross_domain_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    if not summary_df.empty:
        save_heatmap(summary_df, 'true_mae_mean', 'Tier 2 Cross-Domain: True Reconstruction MAE', os.path.join(tier_metrics_root, 'tier2_true_mae_heatmap.png'))
        save_heatmap(summary_df, 'predicted_mae_mean', 'Tier 2 Cross-Domain: Predicted Reconstruction MAE', os.path.join(tier_metrics_root, 'tier2_predicted_mae_heatmap.png'))
        save_heatmap(summary_df, 'spearman_predicted_vs_true_mae', 'Tier 2 Cross-Domain: Spearman(predicted vs true MAE)', os.path.join(tier_metrics_root, 'tier2_spearman_heatmap.png'))
        save_heatmap(summary_df, 'avg_ssim_3d', 'Tier 2 Cross-Domain: Reconstruction SSIM3D', os.path.join(tier_metrics_root, 'tier2_ssim_heatmap.png'))

        md_lines = [
            '# Tier 2 Cross-Domain Stress Test',
            '',
            'Each row corresponds to one frozen expert reconstructor plus its matched scalar-MAE head evaluated on a target dataset.',
            '',
        ]
        for _, row in summary_df.iterrows():
            md_lines.append(
                f"- `{row['expert_display_name']}` on `{row['target_display_name']}` ({row['eval_mode']}): "
                f"true_MAE={row['true_mae_mean']:.6f}, pred_MAE={row['predicted_mae_mean']:.6f}, "
                f"Spearman={row['spearman_predicted_vs_true_mae']:.4f}, SSIM3D={row.get('avg_ssim_3d', np.nan):.4f}"
            )
        Path(os.path.join(tier_metrics_root, 'tier2_cross_domain_summary.md')).write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
