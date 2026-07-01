import cv2
import numpy as np
import time

try:
    import mediapipe as mp
except ImportError:
    mp = None

FRAME_W = 1280
FRAME_H = 720

def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes. bb = (x, y, w, h)"""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = w1 * h1
    bb2_area = w2 * h2
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

class FaceTracker:
    """Face detection for privacy AI. Uses Spatial IOU + Histogram Embeddings for Primary Face Lock."""
    
    def __init__(self):
        self.enabled = False
        self.mp_face = mp.solutions.face_detection if mp else None
        self.detector = None
        if self.mp_face:
            self.detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
        
        # Primary User State
        self.primary_bbox = None
        self.primary_histogram = None
        self.status = "Searching" # Searching, Locked, Reacquiring
        self.lost_time = 0
        
        # Frame skipping
        self._frame_count = 0
        self._cached_bbox = None
        self._cached_face_count = 0
        self._cached_detections = []
        self._cached_status = "Searching"
        self._cached_blurred_count = 0
        
        # Toggles
        self.blur_others_only = True
        self.face_lock_enabled = True

    def _extract_histogram(self, frame, bbox):
        x, y, w, h = bbox
        # Add slight padding to capture full face context
        pad = int(min(w, h) * 0.1)
        sx = max(0, x - pad)
        sy = max(0, y - pad)
        ex = min(frame.shape[1], x + w + pad)
        ey = min(frame.shape[0], y + h + pad)
        
        roi = frame[sy:ey, sx:ex]
        if roi.size == 0:
            return None
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def apply(self, original_frame, processed_frame, ai_res=(960, 540)):
        """
        Returns: (processed_frame, primary_bbox, face_count, status_str, blurred_count)
        """
        if not self.enabled or not self.detector:
            return processed_frame, None, 0, "Inactive", 0
            
        # Reset lock if disabled
        if not self.face_lock_enabled:
            self.primary_bbox = None
            self.primary_histogram = None
            self.status = "Searching"
        
        # Process every frame for smooth tracking (removed frame skipping)
        try:
            rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, ai_res)
            results = self.detector.process(rgb_small)
        except Exception:
            return processed_frame, self._cached_bbox, self._cached_face_count, self._cached_status, 0
        
        h, w = original_frame.shape[:2]
        face_count = 0
        self._cached_detections = []
        
        if results.detections:
            face_count = len(results.detections)
            # Parse all detections
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                fw = min(w - x, int(bbox.width * w))
                fh = min(h - y, int(bbox.height * h))
                self._cached_detections.append((x, y, fw, fh))
            
            # --- FACE LOCK LOGIC ---
            if self.face_lock_enabled:
                best_iou = 0
                best_iou_bbox = None
                
                # 1. Spatial Tracking (IOU)
                if self.primary_bbox is not None:
                    for det in self._cached_detections:
                        iou = get_iou(self.primary_bbox, det)
                        if iou > best_iou:
                            best_iou = iou
                            best_iou_bbox = det
                            
                if best_iou > 0.3:
                    # Spatial Lock Maintained!
                    self.primary_bbox = best_iou_bbox
                    self.status = "Locked"
                    self.lost_time = 0
                    # Update histogram slightly to adapt to lighting changes
                    if self._frame_count % 30 == 0:
                        hist = self._extract_histogram(original_frame, self.primary_bbox)
                        if hist is not None:
                            self.primary_histogram = hist
                else:
                    # Spatial tracking lost. Enter Reacquisition Phase
                    if self.status == "Locked":
                        self.status = "Reacquiring"
                        self.lost_time = time.time()
                        
                    if self.status == "Reacquiring" and self.primary_histogram is not None:
                        # Histogram check
                        best_sim = float('inf')
                        best_hist_bbox = None
                        
                        for det in self._cached_detections:
                            hist = self._extract_histogram(original_frame, det)
                            if hist is not None:
                                # Bhattacharyya distance (lower is better)
                                sim = cv2.compareHist(self.primary_histogram, hist, cv2.HISTCMP_BHATTACHARYYA)
                                if sim < best_sim:
                                    best_sim = sim
                                    best_hist_bbox = det
                        
                        if best_sim < 0.65: # Threshold for match
                            self.primary_bbox = best_hist_bbox
                            self.status = "Locked"
                            self.lost_time = 0
                        else:
                            # Still reacquiring
                            if time.time() - self.lost_time > 5.0:
                                # Timeout after 5 seconds, reset
                                self.status = "Searching"
                                self.primary_bbox = None
                                self.primary_histogram = None
                                
                # 3. Initial Acquisition
                if self.status == "Searching":
                    # Lock onto the largest face
                    largest_area = 0
                    best_bbox = None
                    for det in self._cached_detections:
                        area = det[2] * det[3]
                        if area > largest_area:
                            largest_area = area
                            best_bbox = det
                            
                    if largest_area > 14400: # Must be reasonably close
                        self.primary_bbox = best_bbox
                        hist = self._extract_histogram(original_frame, self.primary_bbox)
                        if hist is not None:
                            self.primary_histogram = hist
                            self.status = "Locked"
            else:
                # Legacy behavior (Largest Area always)
                largest_area = 0
                for det in self._cached_detections:
                    area = det[2] * det[3]
                    if area > largest_area:
                        largest_area = area
                        self.primary_bbox = det
                self.status = "Largest Area"
        else:
            # No faces detected
            if self.status == "Locked":
                self.status = "Reacquiring"
                self.lost_time = time.time()
            elif self.status == "Reacquiring":
                if time.time() - self.lost_time > 5.0:
                    self.status = "Searching"
                    self.primary_bbox = None
                    self.primary_histogram = None

        out_frame = self._apply_blur(processed_frame)
        
        self._cached_bbox = self.primary_bbox
        self._cached_face_count = face_count
        self._cached_status = self.status
        return out_frame, self.primary_bbox, face_count, self.status, self._cached_blurred_count

    def _apply_blur(self, frame):
        """Blur based on UI toggles and primary tracking"""
        output = frame.copy()
        h, w = frame.shape[:2]
        self._cached_blurred_count = 0
        
        for (x, y, fw, fh) in self._cached_detections:
            is_primary = (self.primary_bbox is not None and (x, y, fw, fh) == self.primary_bbox)
            
            should_blur = False
            if self.blur_others_only:
                if not is_primary:
                    should_blur = True
                # Note: if reacquiring, we DON'T blur the primary. If primary is None, blur all? 
                # If searching/reacquiring and we don't know who is primary, we shouldn't blur until locked?
                # Actually, blur everything that isn't the primary. If no primary, blur everything.
            else:
                # Blur everything
                should_blur = True
                
            if should_blur:
                pad = 25
                sx = max(0, x - pad)
                sy = max(0, y - pad)
                ex = min(w, x + fw + pad)
                ey = min(h, y + fh + pad)
                
                if ex > sx and ey > sy:
                    roi = output[sy:ey, sx:ex]
                    if roi.size > 0:
                        output[sy:ey, sx:ex] = cv2.GaussianBlur(roi, (99, 99), 0)
                        self._cached_blurred_count += 1
        
        return output

    def close(self):
        try:
            if self.detector:
                self.detector.close()
        except Exception:
            pass
