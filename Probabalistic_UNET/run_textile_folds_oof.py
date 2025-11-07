#!/usr/bin/env python3
# run_textile_folds_oof.py
import os, subprocess, shutil, pandas as pd

# ---- Hard-coded paths ----
BASE = "/scratch/general/nfs1/u1528328/img_dir/textile_aggregate"                
CHECKPOINTS_ROOT = "/scratch/general/nfs1/u1528328/img_dir/textile_aggregate/checkpoints"
RESULTS_DIR_SCRATCH = "/scratch/general/nfs1/u1528328/img_dir/textile_aggregate/results"
METRICS_DIR = "/scratch/general/nfs1/u1528328/img_dir/textile_aggregate/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_SCRATCH, exist_ok=True)

# Your existing scripts
TRAIN_SCRIPT = "train.py"
TEST_SCRIPT  = "test.py"
EVAL_SCRIPT  = "HSI_comparison_probabalistic.py"
POL_ANGLES = [0, 45, 90, 135]

TRAIN_OPTS = [
    "--model","pix2pix","--input_nc","1","--output_nc","212",
    "--n_epochs","10","--n_epochs_decay","10","--save_epoch_freq","5",
    "--netG","unet_1024","--netG_reps","2","--netD_mult","0",
    "--norm_bitwise","--use_nll","--lambda_l1","0",
    "--norm","instance","--no_dropout"
]
TEST_OPTS = [
    "--model","pix2pix","--input_nc","1","--output_nc","212",
    "--netG","unet_1024","--netG_reps","2","--netD_mult","0",
    "--norm_bitwise","--use_nll","--lambda_l1","0",
    "--norm","instance","--no_dropout","--eval"
]

def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    r = subprocess.run(cmd, text=True, capture_output=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"Command failed: {' '.join(str(x) for x in cmd)}")

def ensure_validation_alias(dataroot:str, val_split:str):
    """Your pipeline expects a folder named 'validation'. If different, alias it.""" 
    if val_split == "validation":
        return
    src = os.path.join(dataroot, val_split)
    dst = os.path.join(dataroot, "validation")
    if not os.path.isdir(src):
        raise SystemExit(f"Expected split folder missing: {src}")
    # Replace any pre-existing alias/dir
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            if os.path.islink(dst): os.unlink(dst)
            else: shutil.rmtree(dst)
        except Exception:
            pass
    os.symlink(src, dst)

def test_and_eval(dataroot: str, checkpoints_dir: str, model_name: str, pol: int, split_label="validation"):
    # 1) TEST
    test_cmd = [
        "python", TEST_SCRIPT,
        "--dataroot", dataroot,
        "--name", model_name,
        "--checkpoints_dir", checkpoints_dir,
        "--polarization", str(pol),
    ] + TEST_OPTS
    run(test_cmd)

    # 2) Locate results produced by your framework (usually under ~/DiffuserNET/results)
    #    We’ll move them to RESULTS_DIR_SCRATCH to keep home tidy.
    home_results_root = os.path.join(os.path.expanduser("~"), "DiffuserNET", "results")
    eval_img_dir = os.path.join(home_results_root, model_name, f"{split_label}_latest", "images")
    if not os.path.isdir(eval_img_dir):
        # Fallback (some setups drop the split name)
        eval_img_dir = os.path.join(home_results_root, model_name, "validation_latest", "images")

    # 3) Count images for eval
    num_images = 0
    if os.path.isdir(eval_img_dir):
        num_images = len([f for f in os.listdir(eval_img_dir) if f.lower().endswith(".tif")])

    # 4) EVAL (writes metrics csv)
    metrics_csv = os.path.join(METRICS_DIR, f"metrics_prob_{model_name}.csv")
    eval_cmd = [
        "python", EVAL_SCRIPT,
        "--results_dir", eval_img_dir,
        "--num_images", str(num_images if num_images>0 else 50),
        "--metrics_csv", metrics_csv
    ]
    run(eval_cmd)
    print(f"Finished {model_name}: {num_images} images, metrics -> {metrics_csv}")

    # 5) Move the whole model results from home → scratch aggregate
    src_model_results = os.path.join(home_results_root, model_name)
    dst_model_results = os.path.join(RESULTS_DIR_SCRATCH, model_name)
    if os.path.isdir(src_model_results):
        if os.path.exists(dst_model_results):
            shutil.rmtree(dst_model_results)
        shutil.move(src_model_results, dst_model_results)
        print(f"Moved results to {dst_model_results}")

    return metrics_csv

def main():
    all_metrics = []

    # Loop folds 1..5, each fold already has training/validation created in step4
    for k in range(1, 6):
        ds_name = f"textile_fold{k}"
        dataroot = os.path.join(BASE, "folds", f"fold{k}")
        checkpoints_dir_base = os.path.join(CHECKPOINTS_ROOT, f"fold{k}")
        ensure_validation_alias(dataroot, "validation")  # no-op if already named 'validation'

        for pol in POL_ANGLES:
            # Unique model name per fold/polarization
            model_name = f"{ds_name}_pol{pol}"
            ckpt_dir = os.path.join(checkpoints_dir_base, f"pol{pol}")
            os.makedirs(ckpt_dir, exist_ok=True)

            # TRAIN
            train_cmd = [
                "python", TRAIN_SCRIPT,
                "--dataroot", dataroot,
                "--name", model_name,
                "--checkpoints_dir", ckpt_dir,
                "--polarization", str(pol),
            ] + TRAIN_OPTS
            run(train_cmd)

            # TEST + EVAL + MOVE RESULTS
            metrics_csv = test_and_eval(
                dataroot=dataroot,
                checkpoints_dir=ckpt_dir,
                model_name=model_name,
                pol=pol,
                split_label="validation"
            )
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                df["dataset"] = ds_name
                df["polarization"] = pol
                all_metrics.append(df)

    # Aggregate metrics across all folds / pol angles
    if all_metrics:
        master_name = os.path.join(METRICS_DIR, "master_metrics_textile_folds.csv")
        pd.concat(all_metrics, ignore_index=True).to_csv(master_name, index=False)
        print(f"\nMaster metrics written to {master_name}")
    else:
        print("\nNo metrics collected.")

if __name__ == "__main__":
    print("Base:", BASE)
    print("Checkpoints root:", CHECKPOINTS_ROOT)
    print("Results (scratch):", RESULTS_DIR_SCRATCH)
    main()
