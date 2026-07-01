import mediapipe as mp
import cv2
import numpy as np

class FaceAnchor:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        self.enabled = True
    
    def apply(self, original_frame, processed_frame):
        """Keep face region visible over any processed output"""
        if not self.enabled:
            return processed_frame
        
        rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
        output = processed_frame.copy()
        h, w = original_frame.shape[:2]
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                pad = 0.2
                x = max(0, int((bbox.xmin - pad) * w))
                y = max(0, int((bbox.ymin - pad * 2) * h))
                fw = min(w-x, int((bbox.width + pad*2) * w))
                fh = min(h-y, int((bbox.height + pad*3) * h))
                
                # Elliptical feathered mask
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (x + fw//2, y + fh//2)
                axes = (fw//2, fh//2)
                if axes[0] > 0 and axes[1] > 0:
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                    mask = cv2.GaussianBlur(mask, (61, 61), 0)
                    
                    alpha = mask.astype(np.float32) / 255.0
                    for c in range(3):
                        output[:,:,c] = (
                            alpha * original_frame[:,:,c] +
                            (1 - alpha) * output[:,:,c]
                        ).astype(np.uint8)
        
        return output
    
    def toggle(self):
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"👤 Face Anchor: {state}")
        return self.enabled
    
    def close(self):
        self.detector.close()
