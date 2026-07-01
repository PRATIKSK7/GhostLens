import cv2
import numpy as np
from typing import Tuple, Optional, List

def is_valid_point(p):
    return (
        isinstance(p, (tuple, list))
        and len(p) == 2
        and all(isinstance(v, (int, float)) for v in p)
    )

def safe_point(p):
    if not is_valid_point(p):
        return None
    return (int(p[0]), int(p[1]))

try:
    import mediapipe as mp
except ImportError:
    mp = None

FRAME_W = 1280
FRAME_H = 720

class GestureEngine:
    """
    Production gesture detector with Kalman-filtered cursor and temporal stabilization.
    
    Gestures:
        OPEN_PALM   → Background Blur
        PINCH       → Full Invisibility
        PRIVACY_RECT_CREATE → Privacy Rectangle mode
        POINTING    → Drawing mode
        THUMBS_UP   → Screenshot
        CLOSED_FIST → Polygon smooth / clear
        NONE        → No hand detected
    """
    
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands if mp else None
        self.mp_selfie = mp.solutions.selfie_segmentation if mp else None
        
        self.hands = None
        self.selfie = None
        
        # Kalman Filter for cursor smoothing
        self.kalman = self._init_kalman()
        self.kalman_left = self._init_kalman()
        self.kalman_right = self._init_kalman()
        self.kalman_initialized = False
        self.kalman_rect_initialized = False
        
        # Gesture temporal stability
        self._gesture_buffer = []
        self._stable_gesture = "NONE"
        self._stability_frames = 3
        
        # Segmentation frame skip (process every 2nd frame)
        self._seg_count = 0
        self._cached_mask = None
        
        # Hand detection frame skip
        self._hand_count = 0
        self.gesture_skip_frames = 2
        self._cached_gesture = "NONE"
        self._cached_confidence = 0.0
        self._cached_lm = None
        self._cached_pointer_raw = None
        
        # Create MediaPipe instances ONCE
        if self.mp_hands:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Multi-hand for privacy rectangle
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
                model_complexity=0  # Lite model for speed
            )
        if self.mp_selfie:
            self.selfie = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def _init_kalman(self):
        """Initialize Kalman filter for 2D cursor position."""
        kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x,y,vx,vy), 2 measurement vars (x,y)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        # Higher process noise = trust raw coordinates more (faster, less lag)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        # Lower measurement noise = more trust in raw coordinates
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        return kf

    def _kalman_predict(self, raw_x, raw_y):
        """Apply Kalman filter to smooth cursor position."""
        measurement = np.array([[np.float32(raw_x)], [np.float32(raw_y)]])
        
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([[raw_x], [raw_y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[raw_x], [raw_y], [0], [0]], np.float32)
            self.kalman_initialized = True
            return raw_x, raw_y
        
        self.kalman.predict()
        corrected = self.kalman.correct(measurement)
        return int(corrected[0][0]), int(corrected[1][0])

    def _kalman_predict_rect(self, raw_lx, raw_ly, raw_rx, raw_ry):
        meas_l = np.array([[np.float32(raw_lx)], [np.float32(raw_ly)]])
        meas_r = np.array([[np.float32(raw_rx)], [np.float32(raw_rx)]])
        # Wait, copy-paste error in string above? Yes, measuring rx twice. Let me fix that.
        meas_r = np.array([[np.float32(raw_rx)], [np.float32(raw_ry)]])
        
        if not self.kalman_rect_initialized:
            self.kalman_left.statePre = np.array([[raw_lx], [raw_ly], [0], [0]], np.float32)
            self.kalman_left.statePost = np.array([[raw_lx], [raw_ly], [0], [0]], np.float32)
            self.kalman_right.statePre = np.array([[raw_rx], [raw_ry], [0], [0]], np.float32)
            self.kalman_right.statePost = np.array([[raw_rx], [raw_ry], [0], [0]], np.float32)
            self.kalman_rect_initialized = True
            return (raw_lx, raw_ly), (raw_rx, raw_ry)
        
        self.kalman_left.predict()
        corr_l = self.kalman_left.correct(meas_l)
        self.kalman_right.predict()
        corr_r = self.kalman_right.correct(meas_r)
        
        return (int(corr_l[0][0]), int(corr_l[1][0])), (int(corr_r[0][0]), int(corr_r[1][0]))

    def _stabilize_gesture(self, raw_gesture):
        """Require N consecutive identical detections before switching gesture."""
        self._gesture_buffer.append(raw_gesture)
        if len(self._gesture_buffer) > self._stability_frames:
            self._gesture_buffer.pop(0)
        
        if len(self._gesture_buffer) == self._stability_frames:
            if all(g == self._gesture_buffer[0] for g in self._gesture_buffer):
                self._stable_gesture = self._gesture_buffer[0]
        
        return self._stable_gesture

    def detect(self, frame: np.ndarray, ai_res: Tuple[int, int] = (960, 540)):
        raw_gesture = "NONE"
        confidence = 0.0
        pointer = None
        seg_mask = None
        
        if not self.hands or not self.selfie:
            return frame, "NONE", 0.0, None, None

        h, w = frame.shape[:2]
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_ai = cv2.resize(rgb_full, ai_res)
        
        # --- Selfie Segmentation (skip every other frame) ---
        self._seg_count += 1
        if self._seg_count % 2 == 1 or self._cached_mask is None:
            try:
                seg_result = self.selfie.process(rgb_ai)
                if seg_result.segmentation_mask is not None:
                    self._cached_mask = cv2.resize(
                        seg_result.segmentation_mask, (FRAME_W, FRAME_H)
                    )
            except Exception:
                pass
        seg_mask = self._cached_mask
        
        # --- Hand Detection (Skip Frames for Optimization) ---
        self._hand_count += 1
        
        if self._hand_count % self.gesture_skip_frames != 0 and self._cached_lm is not None:
            raw_gesture = self._cached_gesture
            confidence = self._cached_confidence
            lm = self._cached_lm
            
            if raw_gesture in ("POINTING", "PINCH") and self.kalman_initialized:
                predicted = self.kalman.predict()
                sx, sy = int(predicted[0][0]), int(predicted[1][0])
                pointer = (max(0, min(w-1, sx)), max(0, min(h-1, sy)))
            elif raw_gesture == "PRIVACY_RECT_CREATE" and self.kalman_rect_initialized:
                pl = self.kalman_left.predict()
                pr = self.kalman_right.predict()
                s_pl = safe_point((pl[0][0], pl[1][0]))
                s_pr = safe_point((pr[0][0], pr[1][0]))
                if s_pl and s_pr:
                    pt1 = (max(0, min(w-1, s_pl[0])), max(0, min(h-1, s_pl[1])))
                    pt2 = (max(0, min(w-1, s_pr[0])), max(0, min(h-1, s_pr[1])))
                    pointer = (pt1, pt2)
                else:
                    pointer = None
            else:
                self.kalman_initialized = False
                self.kalman_rect_initialized = False
        else:
            try:
                hand_results = self.hands.process(rgb_ai)
            except Exception:
                gesture = self._stabilize_gesture("NONE")
                return frame, gesture, 0.0, seg_mask, None
            
            if hand_results.multi_hand_landmarks:
                num_hands = len(hand_results.multi_hand_landmarks)
                
                # Check for dual-hand PRIVACY_RECT_CREATE
                if num_hands == 2:
                    lm0 = hand_results.multi_hand_landmarks[0].landmark
                    lm1 = hand_results.multi_hand_landmarks[1].landmark
                    label0 = hand_results.multi_handedness[0].classification[0].label
                    label1 = hand_results.multi_handedness[1].classification[0].label
                    conf0 = hand_results.multi_handedness[0].classification[0].score
                    conf1 = hand_results.multi_handedness[1].classification[0].score
                    
                    fingers0 = self._get_extended_fingers(lm0, label0)
                    fingers1 = self._get_extended_fingers(lm1, label1)
                    
                    # Both hands showing ONLY index finger
                    if (not fingers0[0] and fingers0[1] and sum(fingers0[2:]) == 0) and \
                       (not fingers1[0] and fingers1[1] and sum(fingers1[2:]) == 0):
                        raw_gesture = "PRIVACY_RECT_CREATE"
                        confidence = (conf0 + conf1) / 2.0
                        
                        raw_lx = int(lm0[8].x * w)
                        raw_ly = int(lm0[8].y * h)
                        raw_rx = int(lm1[8].x * w)
                        raw_ry = int(lm1[8].y * h)
                        
                        pt1, pt2 = self._kalman_predict_rect(raw_lx, raw_ly, raw_rx, raw_ry)
                        
                        s_pt1 = safe_point(pt1)
                        s_pt2 = safe_point(pt2)
                        if s_pt1 and s_pt2:
                            pointer = (
                                (max(0, min(w-1, s_pt1[0])), max(0, min(h-1, s_pt1[1]))),
                                (max(0, min(w-1, s_pt2[0])), max(0, min(h-1, s_pt2[1])))
                            )
                        else:
                            pointer = None
                        
                        self._cached_gesture = raw_gesture
                        self._cached_confidence = confidence
                        self._cached_lm = [lm0, lm1]
                        self._cached_pointer_raw = pointer
                        self.kalman_initialized = False # Reset single-hand kalman
                
                # Single hand logic (or fallback if 2 hands but not PRIVACY_RECT_CREATE)
                if raw_gesture != "PRIVACY_RECT_CREATE":
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    handedness = hand_results.multi_handedness[0]
                    hand_label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    
                    lm = hand_landmarks.landmark
                    fingers = self._get_extended_fingers(lm, hand_label)
                    
                    thumb_tip = lm[4]
                    index_tip = lm[8]
                    pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5
                    
                    raw_x = int(index_tip.x * w)
                    raw_y = int(index_tip.y * h)
                    
                    if pinch_dist < 0.05:
                        raw_gesture = "PINCH"
                    elif not fingers[0] and fingers[1] and sum(fingers[2:]) == 0:
                        raw_gesture = "POINTING"
                    elif fingers[0] and sum(fingers[1:]) == 0:
                        raw_gesture = "THUMBS_UP"
                    elif sum(fingers) == 5:
                        raw_gesture = "OPEN_PALM"
                    elif sum(fingers) == 0:
                        raw_gesture = "CLOSED_FIST"
                    
                    self._cached_gesture = raw_gesture
                    self._cached_confidence = confidence
                    self._cached_lm = [lm]
                    self._cached_pointer_raw = (raw_x, raw_y)
                    
                    if raw_gesture in ("POINTING", "PINCH"):
                        sx, sy = self._kalman_predict(raw_x, raw_y)
                        pointer = (max(0, min(w-1, sx)), max(0, min(h-1, sy)))
                    else:
                        self.kalman_initialized = False
                    self.kalman_rect_initialized = False
            else:
                self._cached_gesture = "NONE"
                self._cached_confidence = 0.0
                self._cached_lm = None
                self.kalman_initialized = False
                self.kalman_rect_initialized = False
                
        if self._cached_lm is not None:
            for lm in self._cached_lm:
                palm_x = int(lm[9].x * w)
                palm_y = int(lm[9].y * h)
                cv2.circle(frame, (palm_x, palm_y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (palm_x, palm_y), 12, (0, 255, 255), 2)
                for tip_id in [4, 8, 12, 16, 20]:
                    tx = int(lm[tip_id].x * w)
                    ty = int(lm[tip_id].y * h)
                    cv2.circle(frame, (tx, ty), 5, (255, 100, 255), -1)
        
        gesture = self._stabilize_gesture(raw_gesture)
        return frame, gesture, confidence, seg_mask, pointer

    def _get_extended_fingers(self, landmarks, hand_label: str) -> List[int]:
        fingers = []
        if hand_label == "Right":
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip_j in zip(tips, pips):
            fingers.append(1 if landmarks[tip].y < landmarks[pip_j].y else 0)
        return fingers

    def close(self):
        try:
            if self.hands:
                self.hands.close()
            if self.selfie:
                self.selfie.close()
        except Exception:
            pass
