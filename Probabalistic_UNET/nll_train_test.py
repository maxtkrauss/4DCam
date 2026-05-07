import os
import subprocess
import shutil
import pandas as pd
import argparse


### NOTES AND USAGE INSTRUCTIONS: ###
# DATASETS is the only section most users should edit.

# Required keys:
#   name: model/result prefix
#   dataroot: dataset root
#   checkpoints_dir: checkpoint root; each polarization writes under checkpoints_dir/pol*
#
# Expected aligned dataset layout:
#   <dataroot>/<phase>/thorlabs/*.tif
#   <dataroot>/<phase>/cubert/*.tif
#
# Training uses phase='training' by default; testing uses phase='validation'
# by default. If a dataset uses different folder names, set:
#   'phase': 'train'                 # use same phase for train/test
#   'train_phase': 'training_split'  # train only
#   'test_phase': 'validation_split' # test only
#
# Architecture selection:
#   default datasets use unet_1024
#   'full_field': True selects unet_2048 for 2048x2048 -> 410x410 full-field data
#   'netG': 'nafnet_512' explicitly selects an architecture
#
# Video-only/full-field inference:
#   'video_mode': True allows thorlabs-only data by creating dummy cubert targets.
#   Use this for inference/evaluation frame generation, not for training.
#   Pair it with 'skip_train': True unless real cubert targets are present.


DATASETS = [
    {
        'name': 'banknotes_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/banknotes',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_banknotes_augmented_prob_retrain',
    },
    {
        'name': 'fossils&fauna',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/fossils&fauna',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_fossils&fauna_prob_retrain',
    },
    {
        'name': 'produce_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/produce',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/Diff_nll_modelsmodels/checkpoints_produce_augmented_prob_retrain',
    },
    {
        'name': 'invertebrates_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/invertebrates',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_invertebrates_augmented_prob_retrain',
    },
    {
        'name': 'rescharts_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/rescharts',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_rescharts_augmented_prob_retrain',
    },
    {
        'name': 'cumulative_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/datasets/cumulative_split_rebuilt',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_cumulative_augmented_prob_retrain',
    },
    # {
    #     'name': 'Umbrella_Video_aligned',
    #     'dataroot': '/scratch/general/nfs1/u1528328/Umbrella_Video/split',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_Umbrella_Video_retrain',
    #     'full_field': True,
    #     'video_mode': True,  # Use video mode for processing frames w/ no GT
    # },
]

POL_ANGLES = [0,45,90,135]
TRAIN_SCRIPT = 'train.py'
TEST_SCRIPT = 'test.py'
EVAL_SCRIPT = 'HSI_comparison_probabalistic.py'
PER_IMAGE_SCRIPT = 'HSI_comparison_probabalistic_per_image.py'
RESULTS_DIR = '/scratch/general/nfs1/u1528328/nll_models/results'  # Directory where test images are saved
METRICS_DIR = '/uufs/chpc.utah.edu/common/home/u1528328/4DCam/Probabalistic_UNET/metrics_prob_nll_retrain' 

# Fixed options for training and testing, matching banknotes_training.sh.
# Dataset-specific architecture selection happens in resolve_dataset_netG().
TRAIN_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '212',
    '--n_epochs', '10',
    '--n_epochs_decay', '10',
    '--save_epoch_freq', '5',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise',
    '--use_nll',
    '--lambda_l1', '0',
    '--norm', 'instance',
    '--no_dropout'          # turn dropout off
]
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
    '--norm','instance',    # InstanceNorm
    '--no_dropout',
    '--eval',                # standard eval; with IN, outputs match train/eval
    '--results_dir', RESULTS_DIR
]

DEFAULT_NETG = 'unet_1024'
FULL_FIELD_NETG = 'unet_2048'
KNOWN_NETG = {
    'unet_128',
    'unet_256',
    'unet_512',
    'unet_1024',
    'unet_1024_mod',
    'unet_1024_to_256',
    'unet_2048',
    'nafnet_128',
    'nafnet_256',
    'nafnet_512',
    'nafnet_1024',
}


def resolve_dataset_netG(ds, cli_netG=None):
    if cli_netG:
        return cli_netG
    if 'netG' in ds:
        return ds['netG']
    if ds.get('full_field', False):
        return FULL_FIELD_NETG
    return DEFAULT_NETG


def validate_dataset_configs(datasets, cli_netG=None):
    if cli_netG and cli_netG not in KNOWN_NETG:
        raise ValueError(f"Unknown --netG '{cli_netG}'. Known options: {', '.join(sorted(KNOWN_NETG))}")

    required_keys = {'name', 'dataroot', 'checkpoints_dir'}
    for ds in datasets:
        missing = sorted(required_keys - set(ds))
        if missing:
            raise ValueError(f"DATASETS entry is missing required keys {missing}: {ds}")

        ds_netG = resolve_dataset_netG(ds, cli_netG)
        if ds_netG not in KNOWN_NETG:
            raise ValueError(
                f"DATASETS entry '{ds['name']}' resolves to unknown netG '{ds_netG}'. "
                f"Known options: {', '.join(sorted(KNOWN_NETG))}"
            )

        if ds.get('full_field', False) and ds.get('netG') and ds['netG'] != FULL_FIELD_NETG:
            raise ValueError(
                f"DATASETS entry '{ds['name']}' sets full_field=True but netG='{ds['netG']}'. "
                f"Use netG='{FULL_FIELD_NETG}' or remove one of those fields."
            )


def with_option(opts, flag, value):
    opts = list(opts)
    try:
        opts[opts.index(flag) + 1] = str(value)
    except ValueError as exc:
        raise ValueError(f"Expected '{flag}' in fixed option list.") from exc
    return opts


def with_flag(opts, flag, enabled=True, value=None):
    opts = list(opts)
    if not enabled:
        return opts
    if flag in opts:
        if value is not None:
            idx = opts.index(flag)
            if idx + 1 < len(opts) and not str(opts[idx + 1]).startswith('--'):
                opts[idx + 1] = str(value)
        return opts
    opts.append(flag)
    if value is not None:
        opts.append(str(value))
    return opts


def apply_dataset_runtime_options(opts, ds, phase_kind):
    opts = list(opts)
    phase = ds.get(f'{phase_kind}_phase', ds.get('phase'))
    if phase:
        opts = with_flag(opts, '--phase', True, phase)
    if ds.get('video_mode', False):
        opts = with_flag(opts, '--video_mode', True, 'True')
    if ds.get('GT_upsample', False):
        opts = with_flag(opts, '--GT_upsample', True, 'True')
    return opts


def dataset_opts(base_opts, ds, cli_netG=None, phase_kind='train'):
    opts = with_option(base_opts, '--netG', resolve_dataset_netG(ds, cli_netG))
    return apply_dataset_runtime_options(opts, ds, phase_kind)

# Helper to run a command and print output
def run_cmd(cmd):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to process (from DATASETS)')
    parser.add_argument(
        '--netG',
        type=str,
        default=None,
        help='Optional override for all datasets. If omitted, DATASETS entries decide via netG or full_field.',
    )
    args = parser.parse_args()

    selected_datasets = DATASETS
    if args.dataset:
        selected_datasets = [ds for ds in DATASETS if ds['name'] == args.dataset]
        if not selected_datasets:
            raise ValueError(f"Dataset {args.dataset} not found in DATASETS.")
    validate_dataset_configs(selected_datasets, args.netG)
        
    all_metrics = []         # rows from HSI_comparison_probabalistic.py (averages)
    all_metrics_per_image = []   # rows from HSI_comparison_prob_per_image.py (one row per image)
    for ds in selected_datasets:
        dataset_name = ds['name']
        ds_netG = resolve_dataset_netG(ds, args.netG)
        train_opts = dataset_opts(TRAIN_OPTS, ds, args.netG, phase_kind='train')
        test_opts = dataset_opts(TEST_OPTS, ds, args.netG, phase_kind='test')
        print(
            f"Dataset '{dataset_name}' using netG='{ds_netG}' "
            f"(full_field={bool(ds.get('full_field', False))}, "
            f"train_phase={ds.get('train_phase', ds.get('phase', 'training'))}, "
            f"test_phase={ds.get('test_phase', ds.get('phase', 'validation'))}, "
            f"video_mode={bool(ds.get('video_mode', False))})"
        )
        for pol in POL_ANGLES:
            # Append polarization to model name and checkpoint dir
            model_name = f"{ds['name']}_pol{pol}"
            ckpt_dir = os.path.join(ds['checkpoints_dir'], f"pol{pol}")

            # 1. Train
            if ds.get('skip_train', False):
                print(f"Skipping training for {model_name}; using existing checkpoint in {ckpt_dir}")
            else:
                train_cmd = [
                    'python', TRAIN_SCRIPT,
                    '--dataroot', ds['dataroot'],
                    '--name', model_name,
                    '--checkpoints_dir', ckpt_dir,
                    '--polarization', str(pol),
                ] + train_opts
                run_cmd(train_cmd)

            # 2. Test
            test_cmd = [
                'python', TEST_SCRIPT,
                '--dataroot', ds['dataroot'],
                '--name', model_name,
                '--checkpoints_dir', ckpt_dir,
                '--polarization', str(pol),
            ] + test_opts
            run_cmd(test_cmd)

            # 3. Evaluate
            eval_img_dir = os.path.join(RESULTS_DIR, model_name, 'validation_latest', 'images')
            # Count number of images for --num_images
            num_images = 0
            if os.path.exists(eval_img_dir):
                num_images = len([f for f in os.listdir(eval_img_dir) if f.startswith('cb_raw_') and f.endswith('.tif')])
            # Write metrics CSV outside of results folder to avoid deletion
            metrics_csv = os.path.join(METRICS_DIR, f'metrics_prob_{model_name}.csv')
            print(f"Writing metrics to: {metrics_csv}")
            eval_cmd = [
                'python', EVAL_SCRIPT,
                '--results_dir', eval_img_dir,
                '--num_images', str(num_images if num_images > 0 else 50),
                '--metrics_csv', metrics_csv
            ]
            run_cmd(eval_cmd)

            # --- 3b. Per-image evaluation (new) ---
            per_image_csv = os.path.join(METRICS_DIR, f'per_image_prob_{model_name}.csv')
            per_image_cmd = [
                'python', PER_IMAGE_SCRIPT,
                '--results_dir', eval_img_dir,
                '--num_images', str(num_images if num_images > 0 else 50),
                '--per_image_csv', per_image_csv,
            ]
            run_cmd(per_image_cmd)

            # Read per-image CSV and tag
            if os.path.exists(per_image_csv):
                df_pi = pd.read_csv(per_image_csv)
                df_pi['dataset'] = ds['name']
                df_pi['polarization'] = pol
                all_metrics_per_image.append(df_pi)
            else:
                print(f"Warning: Per-image metrics file not found for {model_name} pol {pol}")

            # 4. Read metrics and append to master CSV
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                df['dataset'] = ds['name']
                df['polarization'] = pol
                all_metrics.append(df)
            else:
                print(f"Warning: Metrics file not found for {model_name} pol {pol}")

            # 5. Delete test images to save space
            test_img_dir = os.path.join(RESULTS_DIR, model_name)
            if os.path.exists(test_img_dir):
                shutil.rmtree(test_img_dir)
                print(f"Deleted test images in {test_img_dir}")
                
    # Save all metrics to unique master CSVs per dataset
    if all_metrics:
        master_csv = f"master_metrics_{dataset_name}.csv"
        master_df = pd.concat(all_metrics, ignore_index=True)
        master_df.to_csv(master_csv, index=False)
        print(f"Master metrics written to {master_csv}")
    else:
        print("No metrics collected.")

    if all_metrics_per_image:
        master_csv_pi = f"master_metrics_per_image_{dataset_name}.csv"
        master_df_pi = pd.concat(all_metrics_per_image, ignore_index=True)
        master_df_pi.to_csv(master_csv_pi, index=False)
        print(f"Master PER-IMAGE metrics written to {master_csv_pi}")
    else:
        print("No PER-IMAGE metrics collected.")


if __name__ == "__main__":
    main()
