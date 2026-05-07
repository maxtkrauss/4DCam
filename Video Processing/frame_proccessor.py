import tifffile as tiff
import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt

def load_tiff_images(file_pattern):
    """
    Load all TIFF files matching the pattern and return frames in sequential order.
    """
    tiff_files = sorted(glob(file_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    all_frames = []
    
    for tiff_file in tiff_files:
        with tiff.TiffFile(tiff_file) as tif:
            frames = tif.asarray()
            print(f"Loaded {tiff_file} with shape {frames.shape}")
            all_frames.extend(frames)
    
    return np.array(all_frames)

def demosaic_polarization(image):
    """
    Demosaic the polarization mosaic into separate channels (0, 135, 90, and 45 degrees).
    """
    pol_channels = np.empty((4, image.shape[0] // 2, image.shape[1] // 2), dtype=image.dtype)
    
    pol_channels[0] = image[0::2, 0::2]  # Top left, 0 deg
    pol_channels[1] = image[1::2, 0::2]  # Bottom left, 135 deg
    pol_channels[2] = image[1::2, 1::2]  # Bottom right, 90 deg
    pol_channels[3] = image[0::2, 1::2]  # Top right, 45 deg


    return pol_channels

def upsample_polarization(pol_channels, target_size):
    """
    Upsample each polarization channel to match the target size using zero-padding.
    """
    upsampled_channels = np.zeros((4, *target_size), dtype=pol_channels.dtype)
    
    for i in range(4):
        upsampled_channels[i] = cv2.resize(pol_channels[i], target_size, interpolation=cv2.INTER_LINEAR)
    
    return upsampled_channels

def crop_and_mirror(img, pos, box_size):
    y, x = img.shape
    cx, cy = pos
    half = box_size // 2
    x_min, x_max = max(0, cx - half), min(x, cx + half)
    y_min, y_max = max(0, cy - half), min(y, cy + half)
    
    cropped = img[y_min:y_max, x_min:x_max]
    mirrored = np.flip(cropped, axis=0)  # Flip in Y-direction
    
    return mirrored

def crop_square_2048(img2d):
    """
    Center-crop a 2D Thorlabs frame (2048x2448) to a 2048x2048 square.
    Returns (cropped, sx, sy).
    """
    H, W = img2d.shape
    S = min(H, W)  # should be 2048
    sx = (W - S) // 2
    sy = (H - S) // 2
    return img2d[sy:sy+S, sx:sx+S], sx, sy


def crop_square_stack_2048(stack5):
    """
    stack5: (5, H, W) -> (5, 2048, 2048) by center-cropping spatial dims.
    """
    assert stack5.ndim == 3 and stack5.shape[0] == 5, "Expected (5,H,W)"
    H, W = stack5.shape[1:]
    S = min(H, W)  # should be 2048
    sx = (W - S) // 2
    sy = (H - S) // 2
    return stack5[:, sy:sy+S, sx:sx+S]


def warp_stack_with_Mfull(stack5_sq, Mfull, interp=cv2.INTER_LINEAR):
    """
    Apply the SAME affine warp (Mfull) to every channel of a (5,2048,2048) stack.
    """
    assert stack5_sq.shape[1] == stack5_sq.shape[2], "Expected square stack"
    S = stack5_sq.shape[1]
    out = np.empty_like(stack5_sq, dtype=np.float32)
    for ch in range(stack5_sq.shape[0]):
        out[ch] = cv2.warpAffine(stack5_sq[ch].astype(np.float32), Mfull, (S, S), flags=interp)
    return out


def make_thorlabs_5ch_from_raw(frame2d):
    """
    frame2d: raw polarization mosaic (H,W) e.g. 2048x2448
    returns: stack5 (5,H,W) = [pol0, pol135, pol90, pol45, raw]
    (pol channels are demosaiced to half-res then resized back to full-res)
    """
    pol = demosaic_polarization(frame2d)  # (4, H/2, W/2)
    H, W = frame2d.shape
    pol_full = np.zeros((4, H, W), dtype=np.float32)
    for i in range(4):
        pol_full[i] = cv2.resize(pol[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    raw_full = frame2d.astype(np.float32)
    stack5 = np.vstack([pol_full, raw_full[np.newaxis, :, :]])  # (5,H,W)
    return stack5

def force_thorlabs_5ch(frame2d):
    """
    Given a (H,W) Thorlabs frame, force it into a (5,H,W) stack by duplicating channels.
    """
    H, W = frame2d.shape
    stack5 = np.zeros((5, H, W), dtype=np.float32)
    for i in range(5):
        stack5[i] = frame2d.astype(np.float32)
    return stack5


def format_video_frame_like_training(frame2d, Mfull):
    """
    The key function you asked for:
    raw frame -> 5ch -> center crop to 2048 square -> warp with SAME Mfull
    returns (5,2048,2048) float32
    """
    stack5 = force_thorlabs_5ch(frame2d)        # (5,2048,2448)
    stack5_sq = crop_square_stack_2048(stack5)          # (5,2048,2048)
    #aligned = warp_stack_with_Mfull(stack5_sq, Mfull)   # (5,2048,2048)
    return stack5_sq


def process_video_frames_aligned(
    tiff_files,
    Mfull,
    output_dir,
    start_index=0
):
    """
    Read frames from a list of multi-page TIFFs and save each processed frame as
    (5,2048,2048) aligned, matching the training input formatting.
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_index = start_index
    for tiff_file in tiff_files:
        with tiff.TiffFile(tiff_file) as tif:
            frames = tif.asarray()  # (N,H,W)
            print(f"Loaded {tiff_file} with shape {frames.shape}")

        for k in range(frames.shape[0]):
            frame = frames[k]
            aligned5 = format_video_frame_like_training(frame, Mfull)  # (5,2048,2048)

            out_path = os.path.join(output_dir, f"frame_{frame_index:06d}.tif")
            tiff.imwrite(out_path, aligned5.astype(np.float32))
            if frame_index % 50 == 0:
                print(f"Saved {out_path}")
            frame_index += 1

    print("Done. Total frames saved:", frame_index - start_index)


def process_tiff_frames(file_pattern, crop_region, target_size, output_base_dir, mirror = False):
    """
    Process all frames across multiple TIFF files: crop, demosaic, upsample, stack into (5, target_size[0], target_size[1]) arrays,
    and save each frame separately in a dedicated validation folder while ensuring continuity across multiple TIFF files.
    """
    video_name = os.path.basename(file_pattern).split('_')[0]  # Extract base video name
    output_dir = os.path.join(output_base_dir, video_name, "validation", "thorlabs")
    os.makedirs(output_dir, exist_ok=True)
    
    images = load_tiff_images(file_pattern)

    if mirror:
        print("foo")
        frame_index = 0  # Ensure continuous indexing across files
        for frame in images:
            #cropped = crop_and_mirror(frame, (1437, 1370), 660)  # Adjusted for Thorlabs
            pol_channels = demosaic_polarization(cropped)
            upsampled_channels = upsample_polarization(pol_channels, target_size)
            
            stacked_image = np.vstack([upsampled_channels, cropped[np.newaxis, :, :]])  # Stack with unprocessed frame
            output_path = os.path.join(output_dir, f"frame_{frame_index:04d}.tif")
            
            # Save the processed frame
            tiff.imwrite(output_path, stacked_image.astype(np.uint8))  # Ensure saved as 8-bit
            print(f"Saved {output_path}")
            frame_index += 1
    else:    
        y1, y2, x1, x2 = crop_region
        
        frame_index = 0  # Ensure continuous indexing across files
        for frame in images:
            cropped = frame[y1:y2, x1:x2]
            pol_channels = demosaic_polarization(cropped)
            upsampled_channels = upsample_polarization(pol_channels, target_size)
            
            stacked_image = np.vstack([upsampled_channels, cropped[np.newaxis, :, :]])  # Stack with unprocessed frame
            output_path = os.path.join(output_dir, f"frame_{frame_index:04d}.tif")
            
            # Save the processed frame
            tiff.imwrite(output_path, stacked_image.astype(np.uint8))  # Ensure saved as 8-bit
            print(f"Saved {output_path}")
            frame_index += 1


# Example usage
# tiff_files = [
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\35_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\3_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\5_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\10_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\15_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\20_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\25_fps_*.tif",
    # r"D:\NASA_HSI\initial_publication\color_wheel_2.0\polarimetric\30_fps_*.tif"
# ]

# /scratch/general/nfs1/u1405425/D1_videos
tiff_files = [
        r"/scratch/general/nfs1/u1405425/D1_videos/D1_video_0.tif",
        r"/scratch/general/nfs1/u1405425/D1_videos/D1_video_1.tif",
        r"/scratch/general/nfs1/u1405425/D1_videos/D1_video_2.tif",
]

crop_region = (1101, 1101+660, 1073, 1073+660)  # (y1, y2, x1, x2)
target_size = (660, 660)

# Mfull = np.array([
#     [1.00220462,  0.0263162596, -111.350606],
#     [-0.0263162596, 1.00220462, -108.438924]
# ], dtype=np.float32)

# William Mfull:
Mfull = np.array([
    [9.87867453e-01, 1.64158690e-02, -3.96270658e+01],
    [-1.64158690e-02, 9.87867453e-01, -6.44105255e+01]
], dtype=np.float32)

out_frames = r"/scratch/general/nfs1/u1528328/img_dir/office_imaging/william_1/aligned_frames_2048_mosaiced_unwarped"
process_video_frames_aligned(tiff_files, Mfull, out_frames)


# output_base_dir = r"D:\Alexander\processed_frames"

# for file_pattern in tiff_files:
#     process_tiff_frames(file_pattern, crop_region, target_size, output_base_dir, mirror = True)
