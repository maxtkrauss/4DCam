import argparse
import os
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f'Command failed: {" ".join(cmd)}')


def main():
    parser = argparse.ArgumentParser(description='Run corruption sweeps for the risk-aware reconstruction model.')
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoints_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--polarization', type=int, default=0)
    parser.add_argument('--corruption', type=str, default='poisson')
    parser.add_argument('--levels', type=str, default='0.0,0.25,0.5,0.75,1.0')
    parser.add_argument('--base_args', type=str, default='', help='extra args passed through to test.py')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    level_values = [float(x) for x in args.levels.split(',') if x.strip()]

    sweep_root = Path(args.results_dir)
    sweep_root.mkdir(parents=True, exist_ok=True)

    for level in level_values:
        exp_name = f'{args.name}__{args.corruption}_lvl{str(level).replace(".", "p")}'
        test_cmd = [
            'python', str(repo_root / 'test.py'),
            '--dataroot', args.dataroot,
            '--name', args.name,
            '--checkpoints_dir', args.checkpoints_dir,
            '--model', 'pix2pix',
            '--input_nc', '1',
            '--output_nc', '212',
            '--netG', 'unet_1024',
            '--norm', 'instance',
            '--use_nll',
            '--use_risk_model',
            '--lambda_l1', '0',
            '--no_dropout',
            '--eval',
            '--save_eval_metadata',
            '--input_corruption_eval',
            '--input_corruption_prob', '1.0',
            '--input_corruption_types', args.corruption,
            '--polarization', str(args.polarization),
            '--results_dir', str(sweep_root / exp_name),
        ]

        if args.corruption == 'poisson':
            peak = max(1.0, 1024.0 * (1.0 - level))
            test_cmd.extend(['--input_poisson_peak_min', str(peak), '--input_poisson_peak_max', str(peak)])
        elif args.corruption == 'gaussian':
            std = max(1e-4, 0.03 * max(level, 1e-3))
            test_cmd.extend(['--input_gaussian_std_min', str(std), '--input_gaussian_std_max', str(std)])
        elif args.corruption == 'brightness':
            scale = 1.0 + (0.8 * level)
            test_cmd.extend(['--input_brightness_scale_min', str(scale), '--input_brightness_scale_max', str(scale)])
        elif args.corruption == 'blur':
            kernel = int(3 + 8 * level)
            if kernel % 2 == 0:
                kernel += 1
            test_cmd.extend(['--input_blur_kernel_min', str(kernel), '--input_blur_kernel_max', str(kernel)])

        if args.base_args.strip():
            test_cmd.extend(args.base_args.split())
        run_cmd(test_cmd)

        eval_cmd = [
            'python', str(repo_root / 'evaluate_risk_assessment.py'),
            '--results_dir', str(sweep_root / exp_name / args.name / 'validation_latest'),
            '--output_dir', str(sweep_root / exp_name / 'risk_eval'),
        ]
        run_cmd(eval_cmd)


if __name__ == '__main__':
    main()
