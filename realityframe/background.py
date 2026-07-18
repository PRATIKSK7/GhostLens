import cv2
import numpy as np
import time

class BackgroundCapture:
    def __init__(self):
        self.bg_frame = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
    
    def capture(self, cap, num_frames=60, use_cv2_window=True):
        """Capture clean background - user must stay out of frame"""
        print("\n" + "="*50)
        print("📸 BACKGROUND CAPTURE")
        print("="*50)
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                display = frame.copy()
                cv2.putText(display,
                    f"Stay out of frame! Capturing in {i}...",
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
                if use_cv2_window:
                    cv2.imshow("PhantomFrame - Setup", display)
                    cv2.waitKey(1000)
                else:
                    time.sleep(1)
        
        # Warm up camera
        print("⏳ Warming up camera...")
        for _ in range(30):
            cap.read()
        
        # Capture frames for median background
        print("⏳ Capturing background frames...")
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame.astype(np.float32))
            if use_cv2_window:
                cv2.waitKey(30)
            else:
                time.sleep(0.03)
        
        if use_cv2_window:
            cv2.destroyWindow("PhantomFrame - Setup")
        
        if not frames:
            raise RuntimeError("❌ Could not capture background!")
        
        self.bg_frame = np.median(frames, axis=0).astype(np.uint8)
        print("✅ Background captured successfully!")
        return self.bg_frame
    
    def get_fg_mask(self, frame):
        """Get foreground mask using MOG2"""
        mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask
    
    def apply_invisibility(self, frame, opacity=1.0):
        """Blend frame with background for invisibility effect"""
        if self.bg_frame is None:
            return frame
        
        fg_mask = self.get_fg_mask(frame)
        fg_mask_blur = cv2.GaussianBlur(fg_mask, (21, 21), 0)
        alpha = (fg_mask_blur.astype(np.float32) / 255.0) * opacity
        
        result = frame.copy().astype(np.float32)
        bg = self.bg_frame.astype(np.float32)
        
        for c in range(3):
            result[:,:,c] = (
                (1 - alpha) * result[:,:,c] + 
                alpha * bg[:,:,c]
            )
        
        return result.astype(np.uint8)
