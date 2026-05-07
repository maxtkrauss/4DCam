import argparse
import subprocess
from pathlib import Path


RECON_TEST_OPTS = [
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


def run_cmd(cmd):
    print('Running:', ' '.join(str(part) for part in cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def resolve_checkpoint(template_or_path, pol):
    raw = str(template_or_path)
    if '{pol}' in raw:
        return raw.format(pol=pol)
    return raw


def main():
    parser = argparse.ArgumentParser(
        description='Run a trained reconstructor and MAE/reliability head on one dataset for one or more polarizations.'
    )
    parser.add_argument('--dataset_name', required=True, help='Label used in outputs, e.g. Umbrella_Video_aligned_eval')
    parser.add_argument('--dataroot', required=True, help='Dataset root passed to test.py')
    parser.add_argument('--recon_name', required=True, help='Model name used when the reconstructor was trained')
    parser.add_argument('--recon_checkpoints_dir', required=True, help='Checkpoint directory that contains the trained reconstructor')
    parser.add_argument(
        '--mae_checkpoint',
        required=True,
        help='Path to the MAE/reliability checkpoint, or a template containing {pol}',
    )
    parser.add_argument(
        '--polarizations',
        nargs='*',
        type=int,
        default=[0, 45, 90, 135],
        help='Polarizations to evaluate',
    )
    parser.add_argument(
        '--results_root',
        default='/scratch/general/nfs1/u1528328/mae_head_eval_results',
        help='Root folder for reconstruction outputs',
    )
    parser.add_argument(
        '--output_root',
        default='/scratch/general/nfs1/u1528328/model_dir/mae_head_eval_outputs',
        help='Root folder for MAE-head summaries',
    )
    parser.add_argument(
        '--eval_mode',
        default='unseen_ood',
        help='Evaluation mode label written into the MAE-head outputs; non-id modes are treated as OOD',
    )
    parser.add_argument(
        '--max_viz',
        type=int,
        default=12,
        help='How many visualizations evaluate_reliability_model.py should save per polarization',
    )
    parser.add_argument(
        '--skip_reconstruction',
        action='store_true',
        help='Reuse existing reconstruction outputs under results_root instead of rerunning test.py',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    test_script = script_dir / 'test.py'
    eval_script = script_dir / 'evaluate_reliability_model.py'

    for pol in args.polarizations:
        run_tag = f'{args.dataset_name}_pol{pol}'
        scratch_root = Path(args.results_root) / run_tag
        result_dir = scratch_root / args.recon_name / 'validation_latest'
        mae_output_dir = Path(args.output_root) / run_tag
        checkpoint_path = resolve_checkpoint(args.mae_checkpoint, pol)

        if not args.skip_reconstruction:
            recon_cmd = [
                'python',
                str(test_script),
                '--dataroot', args.dataroot,
                '--name', args.recon_name,
                '--checkpoints_dir', args.recon_checkpoints_dir,
                '--polarization', str(pol),
                '--results_dir', str(scratch_root),
                '--phase', 'validation',
            ] + RECON_TEST_OPTS
            run_cmd(recon_cmd)

        eval_cmd = [
            'python',
            str(eval_script),
            '--checkpoint', checkpoint_path,
            '--result_dirs', str(result_dir),
            '--dataset_names', args.dataset_name,
            '--eval_modes', args.eval_mode,
            '--output_dir', str(mae_output_dir),
            '--polarization', str(pol),
            '--norm_bitwise',
            '--max_viz', str(args.max_viz),
        ]
        run_cmd(eval_cmd)


if __name__ == '__main__':
    main()
