<div align="center">
  <h1>4DCam</h1>
  <h3>Probabilistic Spectro–Polarimetric Image Reconstruction and Classification</h3>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="python version">
  <img src="https://img.shields.io/badge/License-4DCam%20Research-green" alt="license">
</div>

<p align="center">
  <b>System Schematic:</b><br>
  <img src="figures/github_schem.png" alt="System schematic" width="600">
</p>

The **4DCam** framework reconstructs and interprets four-dimensional spectral–polarimetric information from single-shot diffuser-encoded measurements. A thin diffuser optically encodes wavelength information into a single grayscale exposure captured by a polarization-resolving CMOS sensor. A probabilistic neural network decodes this measurement into a hyperspectral–polarimetric datacube with per-voxel uncertainty estimates. Classification extensions further demonstrate material and texture discrimination from reconstructed data.

---

## 1. Probabilistic Reconstruction

The reconstruction backbone is a **Probabilistic U-Net** adapted from the [pix2pix GAN framework](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The model jointly optimizes reconstruction fidelity and likelihood-based uncertainty calibration.

- **Input:** diffuser-encoded, polarization-resolved image `(5, 660, 660)`  
- **Output:** hyperspectral–polarimetric cube `(106, 120, 120)` or `(212, 120, 120)`  
- **Core scripts:**  
  - `train.py` / `test.py` — training and inference  
  - `nll_train_test.py` — orchestrates probabilistic training and testing  
  - `HSI_comparison_probabalistic*.py` — reconstruction metric evaluation  
  - `calibrate_sigma_mae.py` — global σ–MAE uncertainty calibration
  - `run_camo_folds_oof.py` — probabilistic reconstruction for camo classification  
- `run_textile_folds_oof.py` — probabalistic reconstruction for textile classification

Results (images, metrics, and calibration plots) are written to the `results/` and `metrics_prob_nll/` directories.

---

## 2. Classification Modules

Two classification frameworks extend the reconstruction model for downstream analysis.

### (a) Camo Classification
Binary CNN distinguishing camouflaged vs. non-camouflaged scenes under three modalities:  
**SPEC** (spectral only), **POL** (polarization only), and **SPECPOL** (fused). Implements consistent architecture across modalities for controlled comparison.  
**Key file:** `Classification/Camo_Classification/Camo_Encoder.py`

### (b) Textile Classification
Multi-class encoder for textile material identification (e.g., cotton, felt, nylon). Features derived from reconstructed hyperspectral cubes are processed through a 1-D attention-based encoder network.  
**Key files:**  
- `Textile_Feature_builder.py` — feature construction  
- `Textile_ResNet.py` — lightweight encoder with depthwise separable convolutions

Both modules include cross-validation, confusion-matrix visualization, and summary metrics.

---

---

## Extension and Contact

For researchers seeking access to the **original training data** used in this work or the **pretrained network weights** (`.pth` files), please reach out directly to the corresponding author:

**Max T. Krauss** — maxtkrauss@gmail.com  
Department of Electrical and Computer Engineering, University of Utah  

All distributed materials and code are governed by the **4DCam Research License (Version 1.0, 2025)**, which restricts use to academic and non-commercial research purposes. Please review the license terms before requesting or redistributing any portion of the data or trained models.

If this code or dataset contributes to your research, please cite the following manuscript:

> **Krauss, M. T.**, Walker, W., Ingold, A., Dammann, J., Majumder, A., & Menon, R.  
> *"Four-dimensional video imaging via generative deep learning and a diffuser-encoded image sensor."*  
> (Manuscript in preparation, 2025)

---


