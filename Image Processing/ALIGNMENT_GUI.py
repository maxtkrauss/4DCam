import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import os
import re
from pathlib import Path
from scipy.optimize import minimize


class ImageRegistrationGUI:
    """
    Smart GUI for image registration with both automatic and manual alignment options.
    Properly handles multi-channel Thorlabs images (5, x, y) and hyperspectral Cubert images (106, x, y).
    FIXED: Preserves original image values without normalization during registration.
    UPDATED: Supports both timestamp-based (YYYYMMDD_HHMMSS) and simple numeric IDs.
    UPDATED: Added "Try Different Pair" functionality for automatic global mode.
    """
    
    def __init__(self, cb_dir, tl_dir, output_dir):
        self.cb_dir = Path(cb_dir)
        self.tl_dir = Path(tl_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.H_map = None
        self.H_map_per_pair = {}
        self.common_ids = []
        self.first_id = None
        self.current_pair_index = 0  # Track current pair for "Try Different Pair"
        
        self.overlay_alpha = 0.5
        self.registration_confirmed = False
        self.registration_direction = 'cubert_to_thorlabs'
        self.registration_mode = 'global'
        
        print("=" * 70)
        print("IMAGE REGISTRATION TOOL - Multi-Channel Support (VALUE PRESERVING)")
        print("=" * 70)
        print(f"Thorlabs folder: {self.tl_dir}")
        print(f"Cubert folder:   {self.cb_dir}")
        print(f"Output folder:   {self.output_dir}")
        print("=" * 70)
    
    @staticmethod
    def get_idx(filename):
        """
        Extract ID from filename. Supports two formats:
        1. Timestamp format: YYYYMMDD_HHMMSS (e.g., thorlabs_20260201_165214.tif)
        2. Simple numeric ID (e.g., image_001.tif, frame42.tif)
        
        Priority is given to timestamp format for more precise matching.
        """
        # First try to match timestamp format (YYYYMMDD_HHMMSS)
        m = re.search(r"(\d{8}_\d{6})", filename)
        if m:
            return m.group(1)
        
        # Fall back to simple numeric ID (first sequence of digits)
        m = re.search(r"(\d+)", filename)
        return m.group(1) if m else None
    
    def load_file_pairs(self):
        """Build dictionaries of matched file pairs."""
        tl_files = {self.get_idx(f.name): f.name for f in self.tl_dir.glob("*.tif")}
        cb_files = {self.get_idx(f.name): f.name for f in self.cb_dir.glob("*.tif")}
        
        self.common_ids = sorted(set(tl_files.keys()) & set(cb_files.keys()))
        self.tl_files = tl_files
        self.cb_files = cb_files
        
        print(f"\nFound {len(self.common_ids)} matched image pairs.")
        
        # Show some examples of matched pairs
        if self.common_ids:
            print("\nExample matched pairs:")
            for i, img_id in enumerate(self.common_ids[:3]):
                print(f"  ID '{img_id}': {self.tl_files[img_id]} <-> {self.cb_files[img_id]}")
            if len(self.common_ids) > 3:
                print(f"  ... and {len(self.common_ids) - 3} more")
        
        if not self.common_ids:
            raise ValueError("No matching image pairs found!")
        
        # Start with pair at index 15 (or last if fewer pairs)
        self.current_pair_index = min(15, len(self.common_ids) - 1)
        self.first_id = self.common_ids[self.current_pair_index]
        return len(self.common_ids)
    
    def load_and_prepare_thorlabs(self, img_id, target_size=410):
        """
        Load Thorlabs image and prepare for processing.
        Returns: (panchromatic_410, full_stack)
        """
        tl_path = self.tl_dir / self.tl_files[img_id]
        tl_stack = tifffile.imread(tl_path)
        
        # Ensure it's 3D (channels, height, width)
        if tl_stack.ndim == 2:
            tl_stack = tl_stack[np.newaxis, ...]
        
        # Use first channel for panchromatic (for homography calculation)
        tl_pan = tl_stack[1].astype(np.float32)
        
        # Normalize ONLY for display/homography calculation
        tl_min, tl_max = np.percentile(tl_pan, [1, 99])
        if tl_max > tl_min:
            tl_pan = np.clip((tl_pan - tl_min) / (tl_max - tl_min), 0, 1)
        else:
            print(f"  ⚠ Warning: Thorlabs {img_id} has no dynamic range")
            tl_pan = np.zeros_like(tl_pan)
        
        # Crop to square
        H, W = tl_pan.shape
        crop_size = min(H, W)
        x0 = (W - crop_size) // 2
        tl_pan_crop = tl_pan[:, x0:x0 + crop_size]
        
        # Downsample to target size
        tl_pan_small = cv2.resize(tl_pan_crop, (target_size, target_size), cv2.INTER_AREA)
        
        return tl_pan_small, tl_stack
    
    def load_and_prepare_cubert(self, img_id, target_size=410):
        """
        Load Cubert image and prepare for processing.
        Returns: (panchromatic_410, full_cube)
        """
        cb_path = self.cb_dir / self.cb_files[img_id]
        cb_cube = tifffile.imread(cb_path)
        
        # Ensure it's 3D or 2D
        if cb_cube.ndim == 3:
            cb_pan = cb_cube.mean(axis=0).astype(np.float32)
        else:
            cb_pan = cb_cube.astype(np.float32)
        
        # Normalize ONLY for display/homography calculation
        cb_min, cb_max = np.percentile(cb_pan, [1, 99])
        if cb_max > cb_min:
            cb_pan = np.clip((cb_pan - cb_min) / (cb_max - cb_min), 0, 1)
        else:
            print(f"  ⚠ Warning: Cubert {img_id} has no dynamic range")
            cb_pan = np.zeros_like(cb_pan)
        
        # Resize if needed (Cubert is usually already 410x410)
        if cb_pan.shape != (target_size, target_size):
            cb_pan = cv2.resize(cb_pan, (target_size, target_size), cv2.INTER_AREA)
        
        return cb_pan, cb_cube
    
    def load_pair_by_index(self, pair_index):
        """Load a specific pair by index."""
        img_id = self.common_ids[pair_index]
        print(f"\nLoading pair {pair_index + 1}/{len(self.common_ids)} (ID = {img_id})...")
        
        tl_pan, tl_stack = self.load_and_prepare_thorlabs(img_id)
        cb_pan, cb_cube = self.load_and_prepare_cubert(img_id)
        
        print(f"  Thorlabs: {tl_stack.shape} → pan {tl_pan.shape}")
        print(f"  Cubert:   {cb_cube.shape} → pan {cb_pan.shape}")
        
        return tl_pan, cb_pan, img_id
    
    def load_first_pair(self):
        """Load first pair for homography computation."""
        self.tl_pan, self.cb_pan, self.first_id = self.load_pair_by_index(self.current_pair_index)
        return self.tl_pan, self.cb_pan
    
    def compute_ncc(self, img1, img2):
        """Compute normalized cross-correlation."""
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
        return np.mean(img1_norm * img2_norm)
    
    def save_homography(self, H, filename="homography_matrix.txt"):
        """Save homography matrix to a text file."""
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            f.write("# Homography Matrix\n")
            f.write(f"# Direction: {self.registration_direction}\n")
            f.write(f"# Mode: {self.registration_mode}\n")
            f.write("# Matrix format: 3x3 (use with cv2.warpPerspective)\n")
            f.write("#\n")
            f.write("# Usage in Python:\n")
            f.write("# import numpy as np\n")
            f.write("# H = np.array([\n")
            for i, row in enumerate(H):
                if i == 0:
                    f.write(f"#     [{row[0]:.10e}, {row[1]:.10e}, {row[2]:.10e}],\n")
                elif i == 1:
                    f.write(f"#     [{row[0]:.10e}, {row[1]:.10e}, {row[2]:.10e}],\n")
                else:
                    f.write(f"#     [{row[0]:.10e}, {row[1]:.10e}, {row[2]:.10e}]\n")
            f.write("# ])\n")
            f.write("#\n")
            f.write("\n")
            
            # Write matrix in numpy format for easy copying
            for row in H:
                f.write(f"{row[0]:.10e} {row[1]:.10e} {row[2]:.10e}\n")
        
        print(f"\n📁 Homography matrix saved to: {save_path}")
        return save_path
    
    def compute_homography_translation(self, source_img=None, target_img=None):
        """
        Compute homography using translation-based NCC optimization.
        Most robust method for registration.
        """
        print("\n" + "="*60)
        print("COMPUTING HOMOGRAPHY - Translation-based NCC")
        
        if source_img is None or target_img is None:
            if self.registration_direction == 'cubert_to_thorlabs':
                source_img = self.cb_pan
                target_img = self.tl_pan
                print("Direction: Cubert → Thorlabs")
            else:
                source_img = self.tl_pan
                target_img = self.cb_pan
                print("Direction: Thorlabs → Cubert")
        
        print("="*60)
        
        def objective(params):
            dx, dy = params
            H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
            source_warped = cv2.warpPerspective(source_img, H, 
                                               (source_img.shape[1], source_img.shape[0]),
                                               flags=cv2.INTER_LINEAR)
            ncc = self.compute_ncc(target_img, source_warped)
            return -ncc
        
        # Coarse grid search
        print("Phase 1: Coarse grid search...")
        best_ncc = -np.inf
        best_dx, best_dy = 0.0, 0.0
        for dx in np.arange(-30, 31, 5):
            for dy in np.arange(-30, 31, 5):
                ncc = -objective([dx, dy])
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_dx, best_dy = dx, dy
        
        print(f"  Best: dx={best_dx:.1f}, dy={best_dy:.1f}, NCC={best_ncc:.4f}")
        
        # Fine optimization
        print("Phase 2: Fine optimization...")
        result = minimize(objective, x0=[best_dx, best_dy], method='Powell',
                         options={'maxiter': 100, 'ftol': 1e-6})
        
        final_dx, final_dy = result.x
        final_ncc = -result.fun
        
        H = np.array([[1, 0, final_dx], [0, 1, final_dy], [0, 0, 1]], dtype=np.float32)
        
        print(f"  Final: dx={final_dx:.2f}, dy={final_dy:.2f}, NCC={final_ncc:.4f}")
        print("\n✓ Homography computed!")
        print("\n" + "─"*60)
        print("HOMOGRAPHY MATRIX:")
        print("─"*60)
        for i, row in enumerate(H):
            print(f"  [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")
        print("─"*60)
        
        # Print in copy-pasteable format
        print("\nCopy-paste format for Python:")
        print("H = np.array([")
        for i, row in enumerate(H):
            if i < 2:
                print(f"    [{row[0]:.10e}, {row[1]:.10e}, {row[2]:.10e}],")
            else:
                print(f"    [{row[0]:.10e}, {row[1]:.10e}, {row[2]:.10e}]")
        print("])")
        print("="*60 + "\n")
        
        return H
    
    def compute_homography_features(self, method='orb', source_img=None, target_img=None):
        """Compute homography using feature matching."""
        print(f"\n{'='*60}")
        print(f"COMPUTING HOMOGRAPHY - {method.upper()} Features")
        
        if source_img is None or target_img is None:
            if self.registration_direction == 'cubert_to_thorlabs':
                source_img = self.cb_pan
                target_img = self.tl_pan
                print("Direction: Cubert → Thorlabs")
            else:
                source_img = self.tl_pan
                target_img = self.cb_pan
                print("Direction: Thorlabs → Cubert")
        
        print("="*60)
        
        # Normalize to uint8
        source_norm = cv2.normalize(source_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        target_norm = cv2.normalize(target_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Select detector
        if method.lower() == 'sift':
            detector = cv2.SIFT_create(nfeatures=2000)
        elif method.lower() == 'akaze':
            detector = cv2.AKAZE_create()
        else:
            detector = cv2.ORB_create(nfeatures=2000)
        
        # Detect and match
        print("Detecting features...")
        kp_target, des_target = detector.detectAndCompute(target_norm, None)
        kp_source, des_source = detector.detectAndCompute(source_norm, None)
        
        print(f"  Target: {len(kp_target)} keypoints")
        print(f"  Source: {len(kp_source)} keypoints")
        
        if len(kp_target) < 4 or len(kp_source) < 4:
            print("⚠ Insufficient keypoints, falling back to translation")
            return self.compute_homography_translation(source_img, target_img)
        
        # Match features
        print("Matching features...")
        if method.lower() in ['sift', 'akaze']:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        matches = matcher.knnMatch(des_target, des_source, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"  Good matches: {len(good_matches)}")
        
        if len(good_matches) < 4:
            print("⚠ Insufficient matches, falling back to translation")
            return self.compute_homography_translation(source_img, target_img)
        
        # Compute homography
        pts_target = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_source = np.float32([kp_source[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)
        
        if H is None:
            print("⚠ Homography failed, falling back to translation")
            return self.compute_homography_translation(source_img, target_img)
        
        print(f"  Inliers: {np.sum(mask)}/{len(good_matches)}")
        print("\n✓ Homography computed!")
        print("\n" + "─"*60)
        print("HOMOGRAPHY MATRIX:")
        print("─"*60)
        for i, row in enumerate(H):
            print(f"  [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")
        print("─"*60)
        print("="*60 + "\n")
        
        return H
    
    def get_manual_correspondences(self, n_points=4):
        """Interactive point selection."""
        if self.registration_direction == 'cubert_to_thorlabs':
            print(f"\nCLICK {n_points} POINTS on THORLABS (target)")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.tl_pan, cmap='gray')
            ax.set_title(f'Thorlabs - Click {n_points} points', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            pts_target = plt.ginput(n_points, timeout=0)
            plt.close()
            
            print(f"CLICK SAME {n_points} POINTS on CUBERT (source)")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.cb_pan, cmap='gray')
            ax.set_title(f'Cubert - Click same {n_points} points', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            pts_source = plt.ginput(n_points, timeout=0)
            plt.close()
        else:
            print(f"\nCLICK {n_points} POINTS on CUBERT (target)")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.cb_pan, cmap='gray')
            ax.set_title(f'Cubert - Click {n_points} points', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            pts_target = plt.ginput(n_points, timeout=0)
            plt.close()
            
            print(f"CLICK SAME {n_points} POINTS on THORLABS (source)")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.tl_pan, cmap='gray')
            ax.set_title(f'Thorlabs - Click same {n_points} points', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            pts_source = plt.ginput(n_points, timeout=0)
            plt.close()
        
        pts_target = np.float32(pts_target).reshape(-1, 1, 2)
        pts_source = np.float32(pts_source).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(pts_source, pts_target, method=0)
        print("\n✓ Homography computed from manual points!")
        print("\n" + "─"*60)
        print("HOMOGRAPHY MATRIX:")
        print("─"*60)
        for i, row in enumerate(H):
            print(f"  [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")
        print("─"*60 + "\n")
        return H
    
    def show_alignment_preview(self, H, img_id=None, enable_try_different=False):
        """
        Show alignment preview with multiple visualization modes.
        
        Args:
            H: Homography matrix
            img_id: Image ID being previewed
            enable_try_different: If True, adds "Try Different Pair" button (for automatic global mode)
        """
        
        # Determine source and target
        if self.registration_direction == 'cubert_to_thorlabs':
            source_img = self.cb_pan
            target_img = self.tl_pan
            source_label = 'Cubert'
            target_label = 'Thorlabs'
        else:
            source_img = self.tl_pan
            target_img = self.cb_pan
            source_label = 'Thorlabs'
            target_label = 'Cubert'
        
        # Warp source
        source_warped = cv2.warpPerspective(source_img, H, 
                                           (source_img.shape[1], source_img.shape[0]),
                                           flags=cv2.INTER_LINEAR)
        
        # Calculate NCC
        ncc = self.compute_ncc(target_img, source_warped)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Original images and registered
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(target_img, cmap='gray')
        title1 = f'{target_label} (Target)'
        if img_id:
            title1 += f' [ID: {img_id}]'
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(source_img, cmap='gray')
        title2 = f'{source_label} (Source)'
        if img_id:
            title2 += f' [ID: {img_id}]'
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(source_warped, cmap='gray')
        ax3.set_title(f'{source_label} (Registered) | NCC={ncc:.4f}',
                     fontsize=12, fontweight='bold',
                     color='green' if ncc > 0.7 else 'orange' if ncc > 0.5 else 'red')
        ax3.axis('off')
        
        # Row 2: Overlays
        ax4 = fig.add_subplot(gs[1, 0])
        overlay_color = np.zeros((*target_img.shape, 3))
        overlay_color[:, :, 0] = target_img
        overlay_color[:, :, 1] = source_warped
        ax4.imshow(overlay_color)
        ax4.set_title(f'Color Overlay (Red={target_label}, Green={source_label})', 
                     fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        # Checkerboard
        mask = np.zeros_like(target_img, dtype=bool)
        block_size = 40
        for i in range(0, target_img.shape[0], block_size):
            for j in range(0, target_img.shape[1], block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    mask[i:i+block_size, j:j+block_size] = True
        checkerboard = np.where(mask, target_img, source_warped)
        ax5.imshow(checkerboard, cmap='gray')
        ax5.set_title('Checkerboard Pattern', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        overlay_alpha = (1 - self.overlay_alpha) * target_img + self.overlay_alpha * source_warped
        im6 = ax6.imshow(overlay_alpha, cmap='gray')
        ax6.set_title(f'Alpha Blend (α={self.overlay_alpha:.2f})', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Alpha slider
        ax_slider = plt.axes([0.68, 0.02, 0.25, 0.02])
        slider = Slider(ax_slider, f'{source_label} Alpha', 0.0, 1.0,
                       valinit=self.overlay_alpha, valstep=0.05)
        
        def update_alpha(val):
            self.overlay_alpha = val
            overlay_new = (1 - val) * target_img + val * source_warped
            im6.set_data(overlay_new)
            ax6.set_title(f'Alpha Blend (α={val:.2f})', fontsize=12, fontweight='bold')
            fig.canvas.draw_idle()
        
        slider.on_changed(update_alpha)
        
        # Buttons - positions depend on whether "Try Different Pair" is enabled
        if enable_try_different:
            ax_confirm = plt.axes([0.45, 0.02, 0.12, 0.04])
            ax_redo = plt.axes([0.58, 0.02, 0.12, 0.04])
            ax_try_diff = plt.axes([0.18, 0.02, 0.15, 0.04])
            
            btn_try_diff = Button(ax_try_diff, 'Try Different Pair', color='lightyellow', hovercolor='yellow')
        else:
            ax_confirm = plt.axes([0.35, 0.02, 0.12, 0.04])
            ax_redo = plt.axes([0.48, 0.02, 0.12, 0.04])
        
        btn_confirm = Button(ax_confirm, 'Confirm & Proceed', color='lightgreen', hovercolor='green')
        btn_redo = Button(ax_redo, 'Redo Alignment', color='lightcoral', hovercolor='red')
        
        action = ['confirm']  # Default action
        
        def on_confirm(event):
            action[0] = 'confirm'
            plt.close(fig)
        
        def on_redo(event):
            action[0] = 'redo'
            plt.close(fig)
        
        def on_try_different(event):
            action[0] = 'try_different'
            plt.close(fig)
        
        btn_confirm.on_clicked(on_confirm)
        btn_redo.on_clicked(on_redo)
        
        if enable_try_different:
            btn_try_diff.on_clicked(on_try_different)
        
        title = f'Registration Preview - Verify Alignment'
        if img_id:
            title += f' - Pair {self.current_pair_index + 1}/{len(self.common_ids)} (ID: {img_id})'
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        plt.show()
        
        return action[0]
    
    def select_direction(self):
        """Select registration direction."""
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.85, 'Select Registration Direction', ha='center', va='center',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.5, 0.70, 'Choose which images to register:', ha='center', va='center',
               fontsize=11, transform=ax.transAxes)
        
        cb_to_tl_text = ("CUBERT → THORLABS\n"
                        "Register Cubert to Thorlabs\n"
                        "Output: Aligned Cubert (410×410)")
        
        tl_to_cb_text = ("THORLABS → CUBERT\n"
                        "Register Thorlabs to Cubert\n"
                        "Output: Aligned Thorlabs (all channels, 2048×2048)")
        
        ax.text(0.25, 0.45, cb_to_tl_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
               transform=ax.transAxes)
        
        ax.text(0.75, 0.45, tl_to_cb_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
               transform=ax.transAxes)
        
        ax_cb_tl = plt.axes([0.12, 0.15, 0.30, 0.08])
        ax_tl_cb = plt.axes([0.58, 0.15, 0.30, 0.08])
        
        btn_cb_tl = Button(ax_cb_tl, 'CUBERT → THORLABS', color='lightblue', hovercolor='blue')
        btn_tl_cb = Button(ax_tl_cb, 'THORLABS → CUBERT', color='lightcoral', hovercolor='red')
        
        self.direction_selected = None
        
        def on_cb_tl(event):
            self.direction_selected = 'cubert_to_thorlabs'
            plt.close(fig)
        
        def on_tl_cb(event):
            self.direction_selected = 'thorlabs_to_cubert'
            plt.close(fig)
        
        btn_cb_tl.on_clicked(on_cb_tl)
        btn_tl_cb.on_clicked(on_tl_cb)
        
        plt.show()
        return self.direction_selected
    
    def select_mode(self):
        """Select registration mode."""
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.85, 'Select Registration Mode', ha='center', va='center',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        global_text = ("GLOBAL\n"
                      "One homography for all images\n"
                      "Faster, best for fixed cameras")
        
        per_pair_text = ("PER-PAIR\n"
                        "Unique homography per image\n"
                        "Slower, best for varying perspectives")
        
        ax.text(0.25, 0.45, global_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
               transform=ax.transAxes)
        
        ax.text(0.75, 0.45, per_pair_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
               transform=ax.transAxes)
        
        ax_global = plt.axes([0.12, 0.15, 0.30, 0.08])
        ax_per_pair = plt.axes([0.58, 0.15, 0.30, 0.08])
        
        btn_global = Button(ax_global, 'GLOBAL', color='lightgreen', hovercolor='green')
        btn_per_pair = Button(ax_per_pair, 'PER-PAIR', color='lightyellow', hovercolor='yellow')
        
        self.mode_selected = None
        
        def on_global(event):
            self.mode_selected = 'global'
            plt.close(fig)
        
        def on_per_pair(event):
            self.mode_selected = 'per_pair'
            plt.close(fig)
        
        btn_global.on_clicked(on_global)
        btn_per_pair.on_clicked(on_per_pair)
        
        plt.show()
        return self.mode_selected
    
    def select_method(self):
        """Select alignment method."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.85, 'Select Alignment Method', ha='center', va='center',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        auto_text = ("AUTOMATIC\n"
                    "Translation-based NCC\n"
                    "Most robust")
        
        manual_text = ("MANUAL\n"
                      "Click corresponding points\n"
                      "Full control")
        
        ax.text(0.25, 0.45, auto_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
               transform=ax.transAxes)
        
        ax.text(0.75, 0.45, manual_text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
               transform=ax.transAxes)
        
        ax_auto = plt.axes([0.15, 0.15, 0.25, 0.08])
        ax_manual = plt.axes([0.6, 0.15, 0.25, 0.08])
        
        btn_auto = Button(ax_auto, 'AUTOMATIC', color='lightgreen', hovercolor='green')
        btn_manual = Button(ax_manual, 'MANUAL', color='lightblue', hovercolor='blue')
        
        self.method_selected = None
        
        def on_auto(event):
            self.method_selected = 'auto'
            plt.close(fig)
        
        def on_manual(event):
            self.method_selected = 'manual'
            plt.close(fig)
        
        btn_auto.on_clicked(on_auto)
        btn_manual.on_clicked(on_manual)
        
        plt.show()
        return self.method_selected
    
    def register_cubert_to_thorlabs(self, img_id, H):
        """
        Register Cubert image to Thorlabs reference.
        FIXED: Preserves original values without normalization.
        """
        cb_path = self.cb_dir / self.cb_files[img_id]
        cb_cube = tifffile.imread(cb_path)
        
        if cb_cube.ndim == 2:
            # Single channel - warp directly without normalization
            cb_registered = cv2.warpPerspective(
                cb_cube.astype(np.float32), 
                H, 
                (410, 410), 
                flags=cv2.INTER_LINEAR
            )
            # Convert back to original dtype
            cb_registered = cb_registered.astype(cb_cube.dtype)
        else:
            # Multi-channel - warp each channel preserving original values
            num_channels = cb_cube.shape[0]
            cb_registered_stack = []
            
            for ch in range(num_channels):
                cb_ch = cb_cube[ch].astype(np.float32)
                cb_ch_reg = cv2.warpPerspective(cb_ch, H, (410, 410), flags=cv2.INTER_LINEAR)
                cb_registered_stack.append(cb_ch_reg)
            
            cb_registered = np.stack(cb_registered_stack, axis=0)
            # Convert back to original dtype
            cb_registered = cb_registered.astype(cb_cube.dtype)
        
        return cb_registered
    
    def register_thorlabs_to_cubert(self, img_id, H):
        """
        Register Thorlabs image to Cubert reference - ALL CHANNELS.
        FIXED: Preserves original values without normalization.
        """
        tl_path = self.tl_dir / self.tl_files[img_id]
        tl_stack = tifffile.imread(tl_path)
        
        # Ensure it's 3D
        if tl_stack.ndim == 2:
            tl_stack = tl_stack[np.newaxis, ...]
        
        num_channels = tl_stack.shape[0]
        OUTPUT_SIZE = 2048
        scale_factor = OUTPUT_SIZE / 410.0
        
        # Scale homography to 2048x2048
        H_scaled = H.copy()
        H_scaled[0, 2] *= scale_factor
        H_scaled[1, 2] *= scale_factor
        
        print(f"  Processing {num_channels} channel(s)...", end=" ")
        
        tl_registered_stack = []
        
        for ch in range(num_channels):
            tl_ch = tl_stack[ch].astype(np.float32)
            
            # Crop to square (no normalization)
            H_img, W_img = tl_ch.shape
            crop_size = min(H_img, W_img)
            x0 = (W_img - crop_size) // 2
            tl_ch_crop = tl_ch[:, x0:x0 + crop_size]
            
            # Upsample to 2048x2048
            tl_ch_up = cv2.resize(tl_ch_crop, (OUTPUT_SIZE, OUTPUT_SIZE), cv2.INTER_LINEAR)
            
            # Apply homography (preserving values)
            tl_ch_reg = cv2.warpPerspective(tl_ch_up, H_scaled, (OUTPUT_SIZE, OUTPUT_SIZE),
                                           flags=cv2.INTER_LINEAR)
            
            tl_registered_stack.append(tl_ch_reg)
        
        # Stack all channels
        tl_registered = np.stack(tl_registered_stack, axis=0)
        
        # Convert back to original dtype
        tl_registered = tl_registered.astype(tl_stack.dtype)
        
        return tl_registered
    
    def batch_process(self):
        """Process all image pairs."""
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING - {len(self.common_ids)} image pairs")
        print(f"Direction: {self.registration_direction}")
        print(f"Mode: {self.registration_mode}")
        print(f"{'='*70}\n")
        
        for idx, img_id in enumerate(self.common_ids):
            print(f"[{idx+1}/{len(self.common_ids)}] Processing ID: {img_id}...", end=' ')
            
            # Get homography
            if self.registration_mode == 'per_pair':
                H = self.H_map_per_pair.get(img_id)
                if H is None:
                    print("⚠ Skipped (no homography)")
                    continue
            else:
                H = self.H_map
            
            # Register and save
            if self.registration_direction == 'cubert_to_thorlabs':
                registered = self.register_cubert_to_thorlabs(img_id, H)
                out_name = self.cb_files[img_id]
            else:
                registered = self.register_thorlabs_to_cubert(img_id, H)
                out_name = self.tl_files[img_id]
            
            out_path = self.output_dir / out_name
            tifffile.imwrite(out_path, registered)
            
            print(f"✓ {registered.shape}")
        
        print(f"\n{'='*70}")
        print(f"✓ ALL {len(self.common_ids)} IMAGES REGISTERED!")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def run(self, n_points=4, auto_method='translation'):
        """Main workflow."""
        # Load file pairs
        self.load_file_pairs()
        
        # Load first pair
        self.load_first_pair()
        
        # Select direction
        self.registration_direction = self.select_direction()
        if not self.registration_direction:
            print("No direction selected. Exiting.")
            return
        
        print(f"\n→ Direction: {self.registration_direction}")
        
        # Select mode
        self.registration_mode = self.select_mode()
        if not self.registration_mode:
            print("No mode selected. Exiting.")
            return
        
        print(f"→ Mode: {self.registration_mode}")
        
        # Select method
        alignment_method = self.select_method()
        if not alignment_method:
            print("No method selected. Exiting.")
            return
        
        print(f"→ Method: {alignment_method}")
        
        # Compute homography
        if self.registration_mode == 'global':
            # GLOBAL MODE with "Try Different Pair" for automatic
            initial_pair_index = self.current_pair_index  # Track starting point
            
            while True:
                # Compute homography for current pair
                if alignment_method == 'auto':
                    H = self.compute_homography_translation()
                    # Show preview with "Try Different Pair" button
                    action = self.show_alignment_preview(H, self.first_id, enable_try_different=True)
                else:
                    H = self.get_manual_correspondences(n_points)
                    # Manual mode - no "Try Different Pair" button
                    action = self.show_alignment_preview(H, self.first_id, enable_try_different=False)
                
                if action == 'confirm':
                    self.H_map = H
                    self.registration_confirmed = True
                    print("✓ Registration confirmed!")
                    
                    # Save homography matrix to file
                    self.save_homography(H, f"homography_{self.registration_direction}.txt")
                    break
                    
                elif action == 'redo':
                    print("↻ Redoing alignment on same pair...")
                    # Stay on current pair, loop continues
                    
                elif action == 'try_different':
                    # Move to next pair (wrap around)
                    self.current_pair_index = (self.current_pair_index + 1) % len(self.common_ids)
                    
                    # Check if we've looped all the way around
                    if self.current_pair_index == initial_pair_index:
                        print("\n↻ Looped through all pairs, back to starting pair")
                    
                    # Load the new pair
                    self.tl_pan, self.cb_pan, self.first_id = self.load_pair_by_index(self.current_pair_index)
                    print(f"→ Trying pair {self.current_pair_index + 1}/{len(self.common_ids)} (ID: {self.first_id})")
        
        else:
            # PER-PAIR MODE
            print(f"\n{'='*70}")
            print("PER-PAIR MODE: Computing homography for each image")
            print(f"{'='*70}\n")
            
            for idx, img_id in enumerate(self.common_ids):
                print(f"\n[{idx+1}/{len(self.common_ids)}] Pair ID: {img_id}")
                
                # Load this pair
                tl_pan, _ = self.load_and_prepare_thorlabs(img_id)
                cb_pan, _ = self.load_and_prepare_cubert(img_id)
                
                # Determine source/target
                if self.registration_direction == 'cubert_to_thorlabs':
                    source_img, target_img = cb_pan, tl_pan
                else:
                    source_img, target_img = tl_pan, cb_pan
                
                # Compute homography
                while True:
                    if alignment_method == 'auto':
                        H = self.compute_homography_translation(source_img, target_img)
                    else:
                        # Store current images for manual selection
                        self.tl_pan = tl_pan
                        self.cb_pan = cb_pan
                        H = self.get_manual_correspondences(n_points)
                    
                    # Preview
                    source_warped = cv2.warpPerspective(source_img, H, 
                                                        (source_img.shape[1], source_img.shape[0]),
                                                        flags=cv2.INTER_LINEAR)
                    ncc = self.compute_ncc(target_img, source_warped)
                    
                    # Auto-confirm if good
                    if ncc >= 0.7:
                        print(f"✓ Auto-confirmed (NCC={ncc:.4f})")
                        confirmed = True
                    else:
                        # Store for preview
                        self.tl_pan = tl_pan
                        self.cb_pan = cb_pan
                        action = self.show_alignment_preview(H, img_id, enable_try_different=False)
                        confirmed = (action == 'confirm')
                    
                    if confirmed:
                        self.H_map_per_pair[img_id] = H
                        break
                    else:
                        print("↻ Redoing...")
            
            print(f"\n✓ All {len(self.H_map_per_pair)} pairs confirmed!")
            
            # Save all homographies to file in per-pair mode
            save_path = self.output_dir / f"homographies_per_pair_{self.registration_direction}.txt"
            with open(save_path, 'w') as f:
                f.write("# Per-Pair Homography Matrices\n")
                f.write(f"# Direction: {self.registration_direction}\n")
                f.write("#\n")
                for img_id, H in self.H_map_per_pair.items():
                    f.write(f"\n# Image ID: {img_id}\n")
                    for row in H:
                        f.write(f"{row[0]:.10e} {row[1]:.10e} {row[2]:.10e}\n")
                    f.write("\n")
            print(f"📁 Per-pair homographies saved to: {save_path}")
        
        # Batch process
        self.batch_process()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Registration with Multi-Channel Support (Value Preserving)')
    parser.add_argument('--cubert', '-c', type=str, default=r"D:\RESCHARTS\alignment\cubert",
                       help='Cubert images folder')
    parser.add_argument('--thorlabs', '-t', type=str, default=r"D:\RESCHARTS\alignment\thorlabs",
                       help='Thorlabs images folder')
    parser.add_argument('--output', '-o', type=str, default=r"D:\RESCHARTS\alignment\output",
                       help='Output folder')
    parser.add_argument('--method', '-m', type=str, default='translation',
                       choices=['translation', 'orb', 'sift', 'akaze'],
                       help='Automatic alignment method')
    parser.add_argument('--points', '-p', type=int, default=88,
                       help='Number of manual points')
    
    args = parser.parse_args()
    
    app = ImageRegistrationGUI(args.cubert, args.thorlabs, args.output)
    app.run(n_points=args.points, auto_method=args.method)