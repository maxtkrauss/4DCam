<div align="center">
  <h1>4DCam</h1>
  <h3>Probabilistic Spectro–Polarimetric Image Reconstruction and Classification</h3>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="python version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="license">
</div>

<p align="center">
  <b>System Schematic:</b><br>
  <img src="figures/github_schem.png" alt="System schematic" width="600">
</p>

The **4DCam** framework implements a probabilistic deep-learning pipeline for reconstructing, quantifying, and classifying spectral–polarimetric information from single-shot diffuser-encoded images.  
A single grayscale measurement—encoded by a thin diffuser and polarization-resolving CMOS sensor—is decoded into a four-dimensional datacube spanning two spatial, one spectral, and one polarization dimension, together with per-voxel uncertainty estimates.  
Downstream classification modules demonstrate scene-level understanding and material discrimination from reconstructed hyperspectral–polarimetric imagery.

---

## 1. Probabilistic Reconstruction Framework

The reconstruction core is based on a **Probabilistic U-Net** architecture adapted from the pix2pix image-to-image translation framework.  
Unlike deterministic GAN-based models, the probabilistic variant models each voxel intensity as a distribution, providing **per-pixel uncertainty (σ)** alongside the reconstructed spectral intensity.

- **Input:**  
  Diffuser-encoded, polarization-resolved grayscale images — shape `(5, 660, 660)`  
  *(4 polarization states + 1 unprocessed intensity channel)*  

- **Output:**  
  Reconstructed hyperspectral–polarimetric cube — shape `(106, 120, 120)` or `(212, 120, 120)`  
  *(106 spectral bands × 4 polarization channels if expanded)*  

- **Loss formulation:**  
  The training objective minimizes the **Negative Log-Likelihood (NLL)** between predicted and true hyperspectral intensities.  
  This encourages the network to jointly optimize reconstruction accuracy and uncertainty calibration.

- **Uncertainty estimation:**  
  Each output voxel is parameterized by a mean (μ) and variance (σ).  
  These are later used to compute pixel-wise σ–MAE correlations as a quantitative measure of calibration fidelity.

- **Key scripts:**  
  - `train.py` — standard model training using the probabilistic U-Net generator  
  - `test.py` — inference and reconstruction on unseen samples  
  - `nll_train_test.py` — high-level entry script coordinating training, testing, and evaluation across datasets and polarization angles  
  - `HSI_comparison_probabalistic.py` and `_per_image.py` — compute reconstruction metrics (MAE, SSIM, spectral fidelity, etc.)  
  - `calibrate_sigma_mae.py` — analyzes σ–MAE relationships across datasets and visualizes uncertainty calibration statistics

---

## 2. Model Implementation Summary

### Architecture
- **Generator:** U-Net with latent Gaussian sampling; outputs both mean and variance per pixel.  
- **Objective:**  
  \[
  \mathcal{L} = \frac{1}{2}\sum_i \left( \frac{(y_i - \mu_i)^2}{\sigma_i^2} + \log\sigma_i^2 \right)
  \]
  capturing both reconstruction error and predictive uncertainty.  
- **Training configuration:**  
  - 20 total epochs (10 linear + 10 decay)  
  - Adam optimizer (learning rate = 2×10⁻⁴, β₁ = 0.5, β₂ = 0.999)  
  - Batch normalization and instance normalization enabled  
  - No adversarial discriminator used in probabilistic mode  

### Evaluation Outputs
- Quantitative metrics saved under `/metrics_prob_nll/` as CSV summaries.  
- Per-image reconstructions and uncertainty maps saved in `/results/`.  
- σ–MAE calibration plots generated in `/metrics_prob_nll/combined_results/`.

---

## 3. Classification Modules

Two auxiliary classification frameworks extend the reconstruction model to demonstrate material and texture discrimination using spectral, polarimetric, and joint spectro–polarimetric features.

### (a) Camo Classification

Implements a **binary convolutional neural network (CNN)** to differentiate camouflaged vs. non-camouflaged scenes under three modalities:
- **SPEC:** spectral-only features (10 wavelength bands)
- **POL:** polarization-only features (4 angles)
- **SPECPOL:** fused spectral–polarimetric representation (14-channel composite)

Each modality is trained with identical CNN architecture for fair comparison, using cross-validation folds and balanced class sampling.

**Core file:** `Classification/Camo_Classification/Camo_Encoder.py`:contentReference[oaicite:0]{index=0}  
Outputs include per-fold accuracy, AUROC, and confusion matrices stored under `results/summary_comparison.csv`.

---

### (b) Textile Classification

Implements **multi-class classification** of textile materials (e.g., cotton, felt, nylon) using features derived from reconstructed hyperspectral cubes.  
Feature extraction combines per-band spectral intensity and polarization descriptors, creating compact representations suitable for 1D convolutional encoders.

- **Feature construction:**  
  `Textile_Feature_builder.py` builds per-sample vectors capturing both broadband and band-resolved polarization statistics:contentReference[oaicite:1]{index=1}.

- **Model architecture:**  
  `Textile_ResNet.py` defines the **IdenticalEncoderNet**, a lightweight 1D attention-based encoder employing depthwise-separable convolutions, channel attention, and global pooling for spectral sequence analysis:contentReference[oaicite:2]{index=2}.

- **Training pipeline:**  
  Stratified k-fold evaluation with optional mixup augmentation, adaptive learning-rate scheduling, and confusion matrix visualization.

**Outputs:**  
Accuracy, F1-score, and visual confusion matrices saved under `features_all/` and `results/`.

---

## 4. Automated Training Pipelines

Batch experiments and cross-validation loops are automated through:

- `run_camo_folds_oof.py` — trains and evaluates the probabilistic U-Net on multi-fold camouflage datasets across polarization states, automatically generating metrics and moving result directories:contentReference[oaicite:3]{index=3}.  
- `run_textile_folds_oof.py` — performs equivalent multi-fold experiments for textile datasets, aggregating results for uncertainty and classification analysis:contentReference[oaicite:4]{index=4}.  

Both scripts sequentially invoke `train.py`, `test.py`, and the probabilistic evaluation scripts, ensuring reproducible model comparisons across folds and modalities.

---

## 5. Environment and Usage

This repository uses a modular PyTorch-based design.  
The probabilistic and classification components are self-contained and can be retrained or adapted for new imaging datasets.


