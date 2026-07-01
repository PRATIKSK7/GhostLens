import cv2
import numpy as np
import sys
import platform

from gesture_detector import GestureDetector
from background import BackgroundCapture
from face_anchor import FaceAnchor
from video_recorder import VideoRecorder
from utils import HUDRenderer

class PhantomFrame:
    def __init__(self, init_camera=True):
        print("\n👻 PhantomFrame Starting...\n")
        
        # Camera
        self.cap = self._init_camera() if init_camera else None
        
        # Modules
        print("⏳ Loading modules...")
        self.gesture_detector = GestureDetector()
        self.bg_capture = BackgroundCapture()
        self.face_anchor = FaceAnchor()
        self.recorder = VideoRecorder()
        self.hud = HUDRenderer()
        
        # State
        self.mode = "NORMAL"  # NORMAL, PARTIAL, INVISIBLE
        self.focus_mode = False
        self.face_anchor_on = True
        self.depth_mode = False
        self.focus_points = []
        self.opacity = 0.85
        
        # Depth segmenter (lazy load)
        self.depth_seg = None
        
        print("✅ All modules loaded!")
    
    def _init_camera(self):
        backend = (cv2.CAP_AVFOUNDATION 
                  if platform.system() == "Darwin" 
                  else cv2.CAP_ANY)
        cap = cv2.VideoCapture(0, backend)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            print("Go to: System Settings > Privacy > Camera")
            print("Enable camera for Terminal/VS Code")
            sys.exit(1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("✅ Camera initialized!")
        return cap
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.focus_points) < 4:
                self.focus_points.append((x, y))
                print(f"📍 Point {len(self.focus_points)}: ({x},{y})")
    
    def _handle_key(self, key):
        if key == ord('q'):
            return False
        elif key == ord('f'):
            self.focus_mode = not self.focus_mode
            self.focus_points = []
            print(f"🔲 Focus Mode: {'ON' if self.focus_mode else 'OFF'}")
        elif key == ord('r'):
            self.focus_points = []
            print("🔄 Region Reset")
        elif key == ord('a'):
            self.face_anchor_on = self.face_anchor.toggle()
        elif key == ord('v'):
            h = 720
            w = 1280
            if self.cap:
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.recorder.toggle(w, h)
        elif key == ord('d'):
            self._toggle_depth_mode()
        elif key == ord('1'):
            self.mode = "NORMAL"
            print("👁️ Mode: NORMAL")
        elif key == ord('2'):
            self.mode = "PARTIAL"
            print("👻 Mode: PARTIAL INVISIBILITY")
        elif key == ord('3'):
            self.mode = "INVISIBLE"
            print("🫥 Mode: FULL INVISIBILITY")
        return True
    
    def _toggle_depth_mode(self):
        if not self.depth_mode:
            try:
                from depth_segmenter import DepthSegmenter
                if self.depth_seg is None:
                    self.depth_seg = DepthSegmenter()
                self.depth_mode = True
                print("🧠 Depth Mode: MiDaS AI")
            except ImportError:
                print("❌ Install torch: pip install torch torchvision")
        else:
            self.depth_mode = False
            print("📷 Depth Mode: MOG2")
    
    def _apply_invisibility(self, frame):
        if self.mode == "NORMAL":
            return frame
        
        opacity = 0.5 if self.mode == "PARTIAL" else self.opacity
        
        # Apply depth segmentation if active and loaded
        if self.mode == "INVISIBLE" and self.depth_mode and self.depth_seg is not None and self.depth_seg.loaded and self.bg_capture.bg_frame is not None:
            mask = self.depth_seg.get_foreground_mask(frame)
            result = self.depth_seg.apply_invisibility(frame, self.bg_capture.bg_frame, mask)
        else:
            result = self.bg_capture.apply_invisibility(frame, opacity)
        
        if self.face_anchor_on:
            result = self.face_anchor.apply(frame, result)
        
        return result
    
    def _apply_focus_mode(self, frame):
        if not self.focus_mode:
            return frame
        if len(self.focus_points) < 4:
            # Show instructions
            cv2.putText(frame,
                f"Click {4 - len(self.focus_points)} more points",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,255), 2)
            # Draw clicked points
            for pt in self.focus_points:
                cv2.circle(frame, pt, 8, (0,255,0), -1)
            return frame
        
        # Apply focus mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        valid_points = []
        for p in self.focus_points:
            if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                valid_points.append([int(p[0]), int(p[1])])
                
        if len(valid_points) < 2:
            return frame
            
        print("POINTS:", valid_points)
        pts = np.array(valid_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        bg = self.bg_capture.bg_frame
        if bg is None:
            return frame
        
        result = np.where(
            mask[:,:,None] == 255, frame, bg
        )
        
        # Draw focus border
        cv2.polylines(result, [pts], True, (0,255,255), 2)
        
        return result

    def process_frame(self, frame):
        """Processes a single frame. Useful for Streamlit integration."""
        # Gesture detection
        gesture, frame = self.gesture_detector.detect(frame)
        if gesture == "PINCH":
            self.mode = "PARTIAL"
        elif gesture == "L_HAND":
            self.mode = "INVISIBLE"
        
        # Apply effects
        result = frame.copy()
        
        if self.focus_mode:
            result = self._apply_focus_mode(result)
        else:
            result = self._apply_invisibility(result)
        
        # HUD
        state = {
            'mode': self.mode,
            'face_anchor': self.face_anchor_on,
            'recording': self.recorder.is_recording,
            'depth_mode': self.depth_mode,
            'focus_mode': self.focus_mode,
        }
        result = self.hud.draw(result, state)
        result = self.recorder.draw_indicator(result)
        
        # Record
        self.recorder.write(result)
        
        return result
    
    def run(self):
        # Capture background first
        self.bg_capture.capture(self.cap)
        
        # Setup window
        cv2.namedWindow("PhantomFrame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PhantomFrame", 1280, 720)
        cv2.setMouseCallback("PhantomFrame", self._mouse_callback)
        
        print("\n🎮 CONTROLS:")
        print("  1/2/3 → Normal / Partial / Full Invisibility")
        print("  F     → Focus Box Mode (click 4 points)")
        print("  R     → Reset region selection")
        print("  A     → Toggle Face Anchor")
        print("  V     → Start/Stop Recording")
        print("  D     → Toggle MiDaS AI Depth Mode")
        print("  Q     → Quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror
            
            result = self.process_frame(frame)
            
            cv2.imshow("PhantomFrame", result)
            
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
        
        self.cleanup()
    
    def cleanup(self):
        print("\n🧹 Cleaning up...")
        if self.recorder.is_recording:
            self.recorder.stop()
        self.gesture_detector.close()
        self.face_anchor.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ PhantomFrame closed.")

if __name__ == "__main__":
    app = PhantomFrame()
    app.run()
