import mediapipe as mp
import cv2
import numpy as np

class GestureDetector:
    def __init__(self):
        # CORRECT WAY for mediapipe 0.10.x
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            model_complexity=0  # 0=lite, 1=full (use 0 for speed)
        )
        
        # Gesture state
        self.current_gesture = "NONE"
        self.gesture_cooldown = 0
        
    def detect(self, frame):
        """
        Returns: gesture name string
        'PINCH'     → Partial invisibility
        'L_HAND'    → Full invisibility  
        'NONE'      → Normal mode
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        gesture = "NONE"
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get handedness (Left/Right)
                handedness = results.multi_handedness[idx]
                hand_label = handedness.classification[0].label
                
                lm = hand_landmarks.landmark
                
                # PINCH DETECTION
                thumb_tip = lm[4]
                index_tip = lm[8]
                dist = ((thumb_tip.x - index_tip.x)**2 + 
                       (thumb_tip.y - index_tip.y)**2) ** 0.5
                if dist < 0.06:
                    gesture = "PINCH"
                
                # L-HAND DETECTION (left hand open palm)
                if hand_label == "Left":
                    fingers_up = self._count_fingers(lm)
                    if fingers_up >= 4:
                        gesture = "L_HAND"
        
        return gesture, frame
    
    def _count_fingers(self, landmarks):
        """Count how many fingers are extended"""
        tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        bases = [6, 10, 14, 18]  # Corresponding base joints
        count = 0
        for tip, base in zip(tips, bases):
            if landmarks[tip].y < landmarks[base].y:
                count += 1
        return count
    
    def close(self):
        self.hands.close()
