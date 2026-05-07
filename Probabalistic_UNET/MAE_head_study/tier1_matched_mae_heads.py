import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    DEFAULT_TIER1_DATASETS,
    METRICS_DIR,
    SCRATCH_RESULTS_DIR,
    dataset_by_name,
    matched_head_output_dir,
    model_name_for_dataset,
    run_cmd,
    run_frozen_reconstructor,
    run_reconstruction_eval,
    sanitize,
    save_bar_summary,
    split_result_dir,
    SCALAR_EVAL_SCRIPT,
    SCALAR_TRAIN_SCRIPT,
)


def main():
    parser = argparse.ArgumentParser(description='Tier 1: train matched scalar-MAE heads for each frozen paper reconstructor.')
    parser.add_argument('--datasets', nargs='*', default=DEFAULT_TIER1_DATASETS)
    parser.add_argument('--polarizations', nargs='*', type=int, default=[0])
    parser.add_argument('--train_fraction', type=float, default=0.5)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--skip_head_train', action='store_true')
    parser.add_argument('--keep_generated_images', action='store_true')
    args = parser.parse_args()

    tier_metrics_root = os.path.join(METRICS_DIR, 'tier1_matched_heads')
    os.makedirs(tier_metrics_root, exist_ok=True)

    all_rows = []
    for pol in args.polarizations:
        for ds_name in args.datasets:
            ds = dataset_by_name(ds_name)
            model_name = model_name_for_dataset(ds['name'], pol)
            run_tag = f'{sanitize(ds["name"])}_pol{pol}_matched'
            scratch_root = os.path.join(SCRATCH_RESULTS_DIR, 'tier1', run_tag)

            result_dir = run_frozen_reconstructor(ds, pol, scratch_root, force_is_ood_label=0)
            recon_metrics_root = os.path.join(tier_metrics_root, 'reconstruction', run_tag)
            run_reconstruction_eval(result_dir, recon_metrics_root)

            head_train_dir = os.path.join(scratch_root, model_name, 'mae_head_train_split')
            head_eval_dir = os.path.join(scratch_root, model_name, 'mae_head_eval_split')
            split_result_dir(
                result_dir=result_dir,
                train_dir=head_train_dir,
                eval_dir=head_eval_dir,
                train_fraction=args.train_fraction,
                seed=args.split_seed + pol + len(all_rows),
            )

            head_model_dir = matched_head_output_dir(os.path.join(tier_metrics_root, 'models'), ds['name'], pol)
            if not args.skip_head_train:
                run_cmd([
                    'python', str(SCALAR_TRAIN_SCRIPT),
                    '--train_results_dirs', head_train_dir,
                    '--val_results_dirs', head_eval_dir,
                    '--output_dir', head_model_dir,
                    '--polarization', str(pol),
                    '--norm_bitwise',
                    '--target_transform', 'log1p',
                    '--target_scale', '100.0',
                ])

            eval_output_dir = matched_head_output_dir(os.path.join(tier_metrics_root, 'evals'), ds['name'], pol)
            run_cmd([
                'python', str(SCALAR_EVAL_SCRIPT),
                '--checkpoint', os.path.join(head_model_dir, 'scalar_mae_predictor_best.pt'),
                '--result_dirs', head_eval_dir,
                '--dataset_names', ds['name'],
                '--eval_modes', 'matched',
                '--output_dir', eval_output_dir,
                '--polarization', str(pol),
                '--norm_bitwise',
            ])

            pooled_summary_path = os.path.join(eval_output_dir, 'scalar_mae_pooled_summary.csv')
            if os.path.exists(pooled_summary_path):
                row = pd.read_csv(pooled_summary_path).iloc[0].to_dict()
                row.update({
                    'dataset_name': ds['name'],
                    'display_name': ds['display_name'],
                    'reconstructor_name': model_name,
                    'eval_mode': 'matched',
                    'polarization': pol,
                })
                all_rows.append(row)

            if not args.keep_generated_images and os.path.isdir(scratch_root):
                shutil.rmtree(scratch_root)
                print(f'Deleted generated images in {scratch_root}')

    summary_df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(tier_metrics_root, 'tier1_matched_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    if not summary_df.empty:
        save_bar_summary(summary_df, os.path.join(tier_metrics_root, 'tier1_spearman_by_model.png'), 'spearman_predicted_vs_true_mae', 'Tier 1 Matched: Spearman(predicted vs true MAE)')
        save_bar_summary(summary_df, os.path.join(tier_metrics_root, 'tier1_mae_prediction_error.png'), 'mae_prediction_mae', 'Tier 1 Matched: MAE of the MAE Prediction')

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(summary_df['predicted_mae_mean'], summary_df['true_mae_mean'], s=55, color='#4c78a8')
        for _, row in summary_df.iterrows():
            ax.text(row['predicted_mae_mean'], row['true_mae_mean'], row['display_name'], fontsize=8)
        lim = max(summary_df['predicted_mae_mean'].max(), summary_df['true_mae_mean'].max()) * 1.1
        ax.plot([0, lim], [0, lim], '--', color='black', linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('Mean Predicted MAE')
        ax.set_ylabel('Mean True MAE')
        ax.set_title('Tier 1 Matched Heads: Mean True vs Predicted MAE')
        plt.tight_layout()
        plt.savefig(os.path.join(tier_metrics_root, 'tier1_mean_true_vs_predicted_mae.png'), dpi=160)
        plt.close(fig)

        md_lines = [
            '# Tier 1 Matched Scalar-MAE Heads',
            '',
            'Each row corresponds to one frozen probabilistic reconstructor evaluated with its own matched scalar-MAE head.',
            '',
        ]
        for _, row in summary_df.iterrows():
            md_lines.append(
                f"- `{row['display_name']}`: true_MAE={row['true_mae_mean']:.6f}, pred_MAE={row['predicted_mae_mean']:.6f}, "
                f"Spearman={row['spearman_predicted_vs_true_mae']:.4f}, Pearson={row['pearson_predicted_vs_true_mae']:.4f}, "
                f"slope={row['fit_slope_true_vs_predicted']:.4f}"
            )
        Path(os.path.join(tier_metrics_root, 'tier1_matched_summary.md')).write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
