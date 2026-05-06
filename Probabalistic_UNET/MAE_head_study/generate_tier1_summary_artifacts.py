import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_ROOT = REPO_ROOT / 'metrics_paper_mae_head_study' / 'tier1_matched_heads' / 'evals'
OUTPUT_ROOT = REPO_ROOT / 'metrics_paper_mae_head_study' / 'tier1_matched_heads'

DATASETS = [
    ('banknotes_pol0', 'Banknotes'),
    ('fossilsandfauna_pol0', 'Fossils/Fauna'),
    ('invertebrates_pol0', 'Invertebrates'),
    ('produce_pol0', 'Produce'),
    ('rescharts_pol0', 'Rescharts'),
]


def build_summary_table():
    rows = []
    for folder_name, display_name in DATASETS:
        csv_path = EVALS_ROOT / folder_name / 'scalar_mae_pooled_summary.csv'
        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        rows.append({
            'dataset': display_name,
            'num_images': int(row['num_images']),
            'true_mae_mean': float(row['true_mae_mean']),
            'predicted_mae_mean': float(row['predicted_mae_mean']),
            'mae_prediction_mae': float(row['mae_prediction_mae']),
            'mae_prediction_rmse': float(row['mae_prediction_rmse']),
            'spearman': float(row['spearman_predicted_vs_true_mae']),
            'pearson': float(row['pearson_predicted_vs_true_mae']),
            'r2': float(row['r2_predicted_vs_true_mae']),
            'fit_slope': float(row['fit_slope_true_vs_predicted']),
            'fit_intercept': float(row['fit_intercept_true_vs_predicted']),
        })
    return pd.DataFrame(rows)


def write_table_files(summary_df: pd.DataFrame):
    csv_path = OUTPUT_ROOT / 'tier1_matched_dataset_table.csv'
    md_path = OUTPUT_ROOT / 'tier1_matched_dataset_table.md'

    summary_df.to_csv(csv_path, index=False)

    display_df = summary_df.copy()
    for col in ['true_mae_mean', 'predicted_mae_mean', 'mae_prediction_mae', 'mae_prediction_rmse',
                'spearman', 'pearson', 'r2', 'fit_slope', 'fit_intercept']:
        display_df[col] = display_df[col].map(lambda x: f'{x:.6f}')
    display_df['num_images'] = display_df['num_images'].map(str)

    md_lines = [
        '# Tier 1 Matched Scalar-MAE Summary',
        '',
        '| Dataset | N | True MAE | Pred MAE | MAE Acc. (MAE) | RMSE | Spearman | Pearson | R^2 | Fit Slope | Fit Intercept |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, row in display_df.iterrows():
        md_lines.append(
            f"| {row['dataset']} | {row['num_images']} | {row['true_mae_mean']} | {row['predicted_mae_mean']} | "
            f"{row['mae_prediction_mae']} | {row['mae_prediction_rmse']} | {row['spearman']} | {row['pearson']} | "
            f"{row['r2']} | {row['fit_slope']} | {row['fit_intercept']} |"
        )
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    return csv_path, md_path


def make_tiled_figure():
    images = []
    titles = []
    for folder_name, display_name in DATASETS:
        img_path = EVALS_ROOT / folder_name / 'pooled_true_vs_predicted_mae.png'
        images.append(Image.open(img_path).convert('RGB'))
        titles.append(display_name)

    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 4.6), constrained_layout=True)
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    output_path = OUTPUT_ROOT / 'tier1_pooled_true_vs_predicted_mae_5panel.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary_table()
    csv_path, md_path = write_table_files(summary_df)
    figure_path = make_tiled_figure()

    print(f'Wrote table CSV to: {csv_path}')
    print(f'Wrote table Markdown to: {md_path}')
    print(f'Wrote tiled figure to: {figure_path}')


if __name__ == '__main__':
    main()
