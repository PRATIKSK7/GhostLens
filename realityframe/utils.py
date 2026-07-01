import cv2
import time

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

def safe_rectangle(frame, pt1, pt2, color, thickness=-1, lineType=cv2.LINE_8):
    try:
        p1 = safe_point(pt1)
        p2 = safe_point(pt2)
        
        if p1 is None or p2 is None:
            return frame

        print("PT1 =", p1)
        print("PT2 =", p2)

        cv2.rectangle(frame, p1, p2, color, thickness, lineType)
    except Exception as e:
        print("Rectangle Error:", e)
    return frame

class HUDRenderer:
    def __init__(self):
        self.fps_history = []
        self.last_time = time.time()
    
    def get_fps(self):
        now = time.time()
        fps = 1.0 / max(now - self.last_time, 0.001)
        self.last_time = now
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        return sum(self.fps_history) / len(self.fps_history)
    
    def draw(self, frame, state: dict):
        """Draw all HUD elements on frame"""
        h, w = frame.shape[:2]
        fps = self.get_fps()
        
        # Semi-transparent top bar
        overlay = frame.copy()
        safe_rectangle(overlay, (0,0), (w, 50), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # FPS - top right
        cv2.putText(frame, f"FPS: {fps:.1f}",
            (w-130, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0,255,0), 2)
        
        # Mode - top left  
        mode_text = state.get('mode', 'NORMAL')
        mode_colors = {
            'NORMAL':    (255,255,255),
            'PARTIAL':   (0,255,255),
            'INVISIBLE': (0,0,255),
            'FOCUS':     (255,165,0),
            'AR':        (0,255,0),
        }
        color = mode_colors.get(mode_text, (255,255,255))
        cv2.putText(frame, f"MODE: {mode_text}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2)
        
        # Bottom status bar
        safe_rectangle(overlay, (0,h-40), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        status_items = [
            f"Face:{' ON' if state.get('face_anchor') else 'OFF'}",
            f"Rec:{' ON' if state.get('recording') else 'OFF'}",
            f"Depth:{' AI' if state.get('depth_mode') else 'MOG2'}",
            f"Focus:{' ON' if state.get('focus_mode') else 'OFF'}",
        ]
        status = "  |  ".join(status_items)
        cv2.putText(frame, status,
            (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (200,200,200), 1)
        
        # Controls hint - bottom right
        controls = "Q:Quit F:Focus A:Face V:Rec D:Depth 1/2/3:Mode"
        cv2.putText(frame, controls,
            (10, h-55), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (150,150,150), 1)
        
        return frame
