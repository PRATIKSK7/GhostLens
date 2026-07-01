import cv2
import time
import numpy as np
import psutil

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

class AROverlay:
    """Lightweight HUD overlay. FPS tracking and system stats (cached)."""
    
    def __init__(self):
        self.fps_history = []
        self.last_time = time.time()
        
        # Cache psutil — expensive, update every 2 seconds
        self._last_sys_time = 0.0
        self._cached_cpu = 0.0
        self._cached_mem = 0.0
        
    def get_fps(self):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        fps = 1.0 / dt
        self.last_time = now
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        return sum(self.fps_history) / len(self.fps_history)
    
    def _get_sys_stats(self):
        now = time.time()
        if now - self._last_sys_time > 2.0:
            self._last_sys_time = now
            try:
                self._cached_cpu = psutil.cpu_percent(interval=0)
                self._cached_mem = psutil.virtual_memory().percent
            except Exception:
                pass
        return self._cached_cpu, self._cached_mem
        
    def apply(self, frame, mode="NORMAL_MODE", gesture="NONE", hand_conf=0.0, face_count=0, **kwargs):
        """Draw premium HUD on frame. Returns (frame, fps)."""
        face_follow = kwargs.get('face_follow', False)
        h, w = frame.shape[:2]
        fps = self.get_fps()
        latency = 1000.0 / max(fps, 1.0)
        
        overlay = frame.copy()
        
        def draw_hud_box(x, y, bw, bh):
            # Semi-transparent dark background
            safe_rectangle(overlay, (x, y), (x + bw, y + bh), (15, 20, 25), -1)
            # Cyber accent line on the left
            safe_rectangle(overlay, (x, y), (x + 3, y + bh), (255, 229, 0), -1)
            
        # 1. Top Left: Mode
        mode_short = mode.replace("_", " ").title()
        if face_follow and mode == "NORMAL_MODE":
            mode_short = "Face Follow Active"
        
        draw_hud_box(20, 20, 320, 60)
        cv2.putText(overlay, "CURRENT MODE", (35, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, mode_short, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2. Top Right: Gesture
        draw_hud_box(w - 280, 20, 260, 60)
        cv2.putText(overlay, "GESTURE DETECTED", (w - 265, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        g_color = (0, 255, 255) if gesture != "NONE" else (100, 100, 100)
        cv2.putText(overlay, gesture.replace("_", " "), (w - 265, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2, cv2.LINE_AA)
        
        # 3. Bottom Left: Telemetry
        draw_hud_box(20, h - 80, 400, 60)
        cv2.putText(overlay, "LIVE TELEMETRY", (35, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        stats_text = f"FPS: {fps:.0f}   |   LATENCY: {latency:.0f}ms   |   FACES: {face_count}"
        cv2.putText(overlay, stats_text, (35, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame, fps
