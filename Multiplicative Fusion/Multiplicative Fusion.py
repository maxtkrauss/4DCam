import os
from pathlib import Path
from typing import Tuple, Literal, Optional

import numpy as np
import tifffile


# -------------------------
# Utilities
# -------------------------
def center_crop_cube(cube: np.ndarray, target_h: int = 120, target_w: int = 120) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected cubert cube as (bands,H,W); got shape {cube.shape}")
    b, h, w = cube.shape
    if target_h > h or target_w > w:
        raise ValueError(f"Target crop {target_h}x{target_w} larger than input {h}x{w}")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return cube[:, y0:y0 + target_h, x0:x0 + target_w]


def _resize_2d(img: np.ndarray, out_hw: Tuple[int, int], mode: str = "area") -> np.ndarray:
    """
    Resize 2D image using OpenCV if available, else fallback to simple numpy.
    mode: "area" (downsample) or "linear" (upsample)
    """
    try:
        import cv2
        interp = cv2.INTER_AREA if mode == "area" else cv2.INTER_LINEAR
        h, w = out_hw
        return cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)
    except Exception:
        # Fallback (not as good as cv2, but works): nearest-neighbor
        h, w = out_hw
        y_idx = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
        x_idx = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
        return img[np.ix_(y_idx, x_idx)].astype(np.float32)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    try:
        import cv2
        # ksize=0 lets cv2 pick based on sigma
        return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
    except Exception:
        # fallback: no blur if cv2 missing
        return img

# -------------------------
# I/O + preprocessing
# -------------------------
def load_and_preprocess(
    cubert_path: str,
    thorlabs_path: str,
    thorlabs_channel: int = 0,
    cubert_bit_depth: int = 12,
    thorlabs_bit_depth: int = 12,
    output_size: int = 600,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    cubert = tifffile.imread(cubert_path).astype(np.float32)
    thorlabs = tifffile.imread(thorlabs_path).astype(np.float32)

    if verbose:
        print(f"Cubert raw shape: {cubert.shape}, dtype: {cubert.dtype}")
        print(f"Thorlabs raw shape: {thorlabs.shape}, dtype: {thorlabs.dtype}")

    # Center-crop padded cubert if needed
    if cubert.ndim == 3 and cubert.shape[1] == 128 and cubert.shape[2] == 128:
        cubert = center_crop_cube(cubert, 120, 120)
        if verbose:
            print(f"Cubert center-cropped to: {cubert.shape} (removed padding)")

    # Select Thorlabs channel if (5,H,W)
    if thorlabs.ndim == 3:
        thorlabs = thorlabs[thorlabs_channel]
        if verbose:
            print(f"Selected Thorlabs channel {thorlabs_channel}: {thorlabs.shape}")

    # Resize Thorlabs to output_size
    if thorlabs.shape != (output_size, output_size):
        thorlabs = _resize_2d(thorlabs, (output_size, output_size), mode="linear")
        if verbose:
            print(f"Thorlabs resized to: {thorlabs.shape}")

    # Normalize Thorlabs by bit-depth to [0,1] (counts-like)
    thorlabs = thorlabs / (2**thorlabs_bit_depth - 1)

    # Cubert: either already [0,1] (GAN sigmoid) or counts-like
    c_min, c_max = float(cubert.min()), float(cubert.max())
    if c_min >= -0.05 and c_max <= 1.5:
        if verbose:
            print("Cubert appears already normalized (GAN sigmoid). Skipping bit-depth normalization.")
    else:
        cubert = cubert / (2**cubert_bit_depth - 1)
        if verbose:
            print("Cubert appears count-scaled. Applied bit-depth normalization.")

    if verbose:
        print(f"Cubert normalized range: [{cubert.min():.4f}, {cubert.max():.4f}]")
        print(f"Thorlabs normalized range: [{thorlabs.min():.4f}, {thorlabs.max():.4f}]")

    return cubert, thorlabs

def super_resolve_weighted_spectral_sum(
    cubert: np.ndarray,      # (C, H_low, W_low)
    thorlabs: np.ndarray,    # (H_high, W_high)
    block_size: int = 5,
    epsilon: float = 1e-6,
    weight_smooth_sigma: float = 1.0,
    clip_output: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Super-resolution using spectral-sum normalized wavelength weights.

    Steps:
      1) Compute per-pixel wavelength weights w_i = C_i / sum_j C_j
      2) Smooth weights spatially (optional)
      3) Upsample weights to high-res
      4) Redistribute measured broadband intensity
    """

    C, H_low, W_low = cubert.shape
    H_high, W_high = thorlabs.shape

    if H_high != H_low * block_size or W_high != W_low * block_size:
        raise ValueError(
            f"Size mismatch: thorlabs {thorlabs.shape} vs cubert {cubert.shape}"
        )

    # ---- 1) spectral-sum normalization (weights)
    spec_sum = np.sum(cubert, axis=0, keepdims=True)  # (1,H_low,W_low)
    weights = cubert / (spec_sum + epsilon)           # (C,H_low,W_low)

    # ---- 2) optional spatial smoothing of weights
    if weight_smooth_sigma and weight_smooth_sigma > 0:
        for i in range(C):
            weights[i] = _gaussian_blur(weights[i], sigma=weight_smooth_sigma)

    # ---- 3) upsample weights + apply measured intensity
    out = np.zeros((C, H_high, W_high), dtype=np.float32)

    for i in range(C):
        w_up = _resize_2d(weights[i], (H_high, W_high), mode="linear")
        out[i] = thorlabs * w_up

        if verbose and (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{C} channels")

    if clip_output:
        out = np.clip(out, 0, 1)

    if verbose:
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

    return out



# -------------------------
# Core method (smooth gain)
# -------------------------
def super_resolve_smooth_multiplicative(
    cubert: np.ndarray,            # (C, H_low, W_low)
    thorlabs: np.ndarray,          # (H_high, W_high)
    block_size: int = 5,
    epsilon: float = 1e-6,
    gain_clip: Tuple[float, float] = (0.25, 4.0),
    gain_smooth_sigma: float = 1.0,
    denom_floor_percentile: float = 1.0,
    clip_output: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    C, H_low, W_low = cubert.shape
    H_high, W_high = thorlabs.shape

    if H_high != H_low * block_size or W_high != W_low * block_size:
        raise ValueError(
            f"Size mismatch: thorlabs {thorlabs.shape} vs cubert {cubert.shape} with block_size={block_size}"
        )

    # Downsample Thorlabs to low-res reference
    th_low = _resize_2d(thorlabs, (H_low, W_low), mode="area")  # robust averaging
    # Avoid tiny denominators that cause bright speckle
    floor = np.percentile(th_low, denom_floor_percentile)
    th_low = np.maximum(th_low, float(floor))

    out = np.zeros((C, H_high, W_high), dtype=np.float32)

    for i in range(C):
        cb = cubert[i]

        # Low-res gain
        gain = cb / (th_low + epsilon)

        # Clamp gain to reduce extreme amplification
        gmin, gmax = gain_clip
        gain = np.minimum(gain, gmax) # remove lower clip to allow boost from dark regions

        # Smooth gain (kills blocky artifacts / isolated hot pixels)
        gain = _gaussian_blur(gain, sigma=gain_smooth_sigma)

        # Upsample gain smoothly to high-res
        gain_up = _resize_2d(gain, (H_high, W_high), mode="linear")

        # Apply
        out[i] = thorlabs * gain_up

        if verbose and (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{C} channels")

    if clip_output:
        out = np.clip(out, 0, 1)

    if verbose:
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

    return out


def downsample_block_average(high: np.ndarray, block_size: int) -> np.ndarray:
    # (H_high,W_high) -> (H_low,W_low) by exact block averaging
    H, W = high.shape
    H_low, W_low = H // block_size, W // block_size
    x = high[:H_low * block_size, :W_low * block_size]
    x = x.reshape(H_low, block_size, W_low, block_size).mean(axis=(1, 3))
    return x.astype(np.float32)


def verify_reconstruction(output: np.ndarray, cubert_original: np.ndarray, block_size: int = 5) -> dict:
    # Note: v2 is *not* guaranteed to be perfectly block-conservative after smoothing/clamping.
    # This verification is still useful as a sanity check.
    C = output.shape[0]
    recon = np.zeros_like(cubert_original, dtype=np.float32)
    for i in range(C):
        recon[i] = downsample_block_average(output[i], block_size)

    mae = float(np.mean(np.abs(recon - cubert_original)))
    mse = float(np.mean((recon - cubert_original) ** 2))
    max_error = float(np.max(np.abs(recon - cubert_original)))

    print("\nReconstruction Verification (sanity check):")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max Error: {max_error:.6f}")

    return {"mae": mae, "mse": mse, "max_error": max_error, "reconstructed": recon}


# -------------------------
# Main API (same signature)
# -------------------------
def main(
    cubert_path: str,
    thorlabs_path: str,
    output_path: str,
    method: Literal["smooth_multiplicative", "spectral_sum_weights"] = "spectral_sum_weights",
    thorlabs_channel: int = 0,
    cubert_bit_depth: int = 12,
    thorlabs_bit_depth: int = 12,
    output_size: int = 600,
    block_size: int = 5,
    save_as_16bit: bool = True,
    # v2 knobs (used by smooth_multiplicative):
    gain_clip: Tuple[float, float] = (0.25, 4.0),
    gain_smooth_sigma: float = 1.0,
    denom_floor_percentile: float = 1.0,
    # v2 knobs (used by detail_injection_conserve):
    detail_blur_sigma: float = 2.0,
    detail_strength: float = 1.0,
    clip_output: bool = False,   # <-- IMPORTANT: do NOT clip during processing by default
    verbose: bool = True,
):
    print("=" * 60)
    print("SuperMax v2: Multiplicative Fusion Super-Resolution")
    print(f"  method = {method}")
    print("=" * 60)

    if verbose:
        print("\n[1/4] Loading images...")

    cubert, thorlabs = load_and_preprocess(
        cubert_path=cubert_path,
        thorlabs_path=thorlabs_path,
        thorlabs_channel=thorlabs_channel,
        cubert_bit_depth=cubert_bit_depth,
        thorlabs_bit_depth=thorlabs_bit_depth,
        output_size=output_size,
        verbose=verbose,
    )

    # Enforce consistent sizing: output_size must equal low*block_size
    H_low = cubert.shape[1]
    expected = H_low * block_size
    if output_size != expected:
        if verbose:
            print(f"\nAdjusting output_size from {output_size} -> {expected} to match low*block_size")
        output_size = expected
        thorlabs = _resize_2d(thorlabs, (output_size, output_size), mode="linear")

    if verbose:
        print("\n[2/4] Super-resolving...")

    if method == "smooth_multiplicative":
        output = super_resolve_smooth_multiplicative(
            cubert=cubert,
            thorlabs=thorlabs,
            block_size=block_size,
            gain_clip=gain_clip,
            gain_smooth_sigma=gain_smooth_sigma,
            denom_floor_percentile=denom_floor_percentile,
            clip_output=clip_output,   # <-- now user-controlled (default False)
            verbose=verbose,
        )
    elif method == "spectral_sum_weights":
        output = super_resolve_weighted_spectral_sum(
            cubert=cubert,
            thorlabs=thorlabs,
            block_size=block_size,
            weight_smooth_sigma=gain_smooth_sigma,  # reuse knob
            clip_output=clip_output,
            verbose=verbose,
        )


    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        print("\n[3/4] Verifying reconstruction...")
    metrics = verify_reconstruction(output, cubert, block_size)

    if verbose:
        print(f"\n[4/4] Saving to {output_path}...")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # -----------------------------
    # IMPORTANT: clip ONLY at save time for uint16
    # -----------------------------
    if save_as_16bit:
        out_min, out_max = float(output.min()), float(output.max())
        if out_min < 0.0 or out_max > 1.0:
            print(f"NOTE: output out of [0,1] before save: min={out_min:.6f}, max={out_max:.6f}")
            print("      Clipping to [0,1] ONLY for uint16 export.")
        output_to_save = np.clip(output, 0.0, 1.0)

        out16 = (output_to_save * 65535.0).round().astype(np.uint16)
        tifffile.imwrite(output_path, out16)
        if verbose:
            print("Saved as 16-bit TIFF (clipped only for export)")
    else:
        # Float32 path: preserve the true values (no clipping)
        tifffile.imwrite(output_path, output.astype(np.float32))
        if verbose:
            print("Saved as 32-bit float TIFF (no clipping)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return output, metrics

