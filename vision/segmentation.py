import cv2
import numpy as np

FRAME_W = 1280
FRAME_H = 720

class BackgroundSegmenter:
    """Production-grade segmentation for invisibility and background blur effects."""
    
    def __init__(self):
        self.bg_frame = None           # Captured clean background (1280x720)
        self.last_pose_mask = None     # Temporal stabilization accumulator
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _force_size(self, img):
        """Force any image/mask to standard 1280x720."""
        if img is None:
            return None
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]
        if w != FRAME_W or h != FRAME_H:
            return cv2.resize(img, (FRAME_W, FRAME_H))
        return img

    def _normalize_mask(self, mask):
        """Ensure mask is float32 in [0,1] range."""
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        return mask

    def _refine_mask(self, raw_mask_01):
        """
        Production mask refinement pipeline:
        1. Binarize at threshold
        2. Morphological close (fill holes)
        3. Morphological open (remove noise)
        4. Temporal EMA stabilization
        5. Gaussian feathering
        Returns: feathered float32 alpha mask [0,1]
        """
        # Binarize
        _, binary = cv2.threshold(raw_mask_01, 0.5, 1.0, cv2.THRESH_BINARY)
        binary_u8 = (binary * 255).astype(np.uint8)
        
        # Morphological refinement
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, self._morph_kernel, iterations=3)
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, self._small_kernel, iterations=2)
        binary_u8 = cv2.dilate(binary_u8, self._small_kernel, iterations=1)
        
        refined = binary_u8.astype(np.float32)
        
        # Temporal stabilization (EMA)
        if self.last_pose_mask is None or self.last_pose_mask.shape != refined.shape:
            self.last_pose_mask = refined.copy()
        else:
            cv2.accumulateWeighted(refined, self.last_pose_mask, 0.35)
        
        stable = self.last_pose_mask.copy()
        stable_u8 = np.clip(stable, 0, 255).astype(np.uint8)
        
        # Gaussian feathering for seamless edges
        feathered = cv2.GaussianBlur(stable_u8, (31, 31), 0)
        return feathered.astype(np.float32) / 255.0

    def apply_invisibility(self, frame, pose_segmentation=None):
        """
        Full invisibility: replace person with captured background.
        Uses MediaPipe pose_segmentation mask for person detection.
        """
        if self.bg_frame is None:
            return frame
        
        frame = self._force_size(frame)
        bg = self._force_size(self.bg_frame)
        
        if pose_segmentation is not None:
            pose_segmentation = self._force_size(pose_segmentation)
            norm = self._normalize_mask(pose_segmentation)
            alpha = self._refine_mask(norm)
        else:
            return frame
        
        # alpha=1 where person is → replace with background
        alpha_3d = np.stack([alpha] * 3, axis=-1)
        
        result = (alpha_3d * bg.astype(np.float32) + 
                  (1.0 - alpha_3d) * frame.astype(np.float32))
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_ultra_blur(self, frame, pose_segmentation=None, blur_level=3):
        """
        DSLR-style portrait blur: person stays sharp, background is heavily blurred.
        Multi-pass downscale blur for quality + performance. Adjustable intensity (1-5).
        """
        frame = self._force_size(frame)
        
        if pose_segmentation is not None:
            pose_segmentation = self._force_size(pose_segmentation)
            norm = self._normalize_mask(pose_segmentation)
            
            # Refine mask
            _, binary = cv2.threshold(norm, 0.5, 1.0, cv2.THRESH_BINARY)
            binary_u8 = (binary * 255).astype(np.uint8)
            binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, self._morph_kernel, iterations=3)
            binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, self._small_kernel, iterations=2)
            binary_u8 = cv2.dilate(binary_u8, self._small_kernel, iterations=1)
            
            # Feather edges
            alpha = cv2.GaussianBlur(binary_u8, (31, 31), 0).astype(np.float32) / 255.0
        else:
            return frame
        
        # Blur settings based on level (1-5)
        # 1: Light, 2: Medium, 3: Strong, 4: Ultra, 5: Privacy (Extreme)
        blur_level = max(1, min(5, int(blur_level)))
        
        scale_map = {1: 0.5, 2: 0.5, 3: 0.25, 4: 0.25, 5: 0.125}
        passes_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        kernel_map = {1: 11, 2: 15, 3: 21, 4: 31, 5: 31}
        
        scale = scale_map[blur_level]
        passes = passes_map[blur_level]
        k_size = kernel_map[blur_level]
        
        h, w = FRAME_H, FRAME_W
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        for _ in range(passes):
            small = cv2.GaussianBlur(small, (k_size, k_size), 0)
            
        if blur_level >= 3:
            small = cv2.bilateralFilter(small, 9, 75, 75)
            small = cv2.GaussianBlur(small, (k_size, k_size), 0)
            
        bg_blur = cv2.resize(small, (FRAME_W, FRAME_H))
        
        # Blend: person sharp, background blurred
        alpha_3d = np.stack([alpha] * 3, axis=-1)
        result = (alpha_3d * frame.astype(np.float32) + 
                  (1.0 - alpha_3d) * bg_blur.astype(np.float32))
        
        return np.clip(result, 0, 255).astype(np.uint8)
