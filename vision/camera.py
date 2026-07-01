import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple

# Standard frame size for ALL operations
FRAME_W = 1280
FRAME_H = 720

class CameraController:
    """Thread-safe, lock-free camera capture module for ultra-low latency."""
    
    def __init__(self, camera_id: int = 0, width: int = FRAME_W, height: int = FRAME_H, fps: int = 30) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        self.current_frame = None
        self.running = False
        self.thread = None
        self._error = None
        self._frame_count = 0
        from collections import deque
        self.frame_buffer = deque(maxlen=1)

    def start(self) -> bool:
        """
        Start camera capture thread. Safe to call multiple times.
        Returns:
            bool: True if camera started successfully, False otherwise.
        """
        if self.running and self.thread is not None and self.thread.is_alive():
            return True
        
        # Clean up any stale resources
        if self.thread is not None or self.cap is not None:
            self.stop()
        
        # Prefer AVFOUNDATION for macOS
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_id)
        except Exception:
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap or not self.cap.isOpened():
            self._error = f"Could not open camera {self.camera_id}"
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer latency
        
        self._error = None
        self.running = True
        self._frame_count = 0
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Watchdog thread
        self._last_frame_time = time.time()
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()
        
        # Warmup — wait for first frame
        for _ in range(100):
            if self.current_frame is not None:
                break
            time.sleep(0.05)
            
        return self.current_frame is not None

    def _watchdog(self):
        """Monitors the camera thread and restarts it if it freezes."""
        while self.running:
            time.sleep(1.0)
            if self.current_frame is not None:
                if time.time() - self._last_frame_time > 5.0:  # 5 seconds freeze
                    print("[Watchdog] Camera frozen. Restarting...")
                    self.stop()
                    time.sleep(1.0)
                    self.start()
                    break

    def _update(self):
        """Background thread: reads frames continuously."""
        consecutive_failures = 0
        max_failures = 200
        
        try:
            while self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    # Mirror + force standard size
                    frame = cv2.flip(frame, 1)
                    if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                    
                    self.frame_buffer.append(frame)
                    self.current_frame = frame  # for watchdog
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self._error = "Camera read failures exceeded limit"
                        self.running = False
                        break
                    time.sleep(0.002)
        except Exception as e:
            self._error = str(e)
            self.running = False

    def capture_background(self, num_frames: int = 40) -> Optional[np.ndarray]:
        """
        Capture clean background using median of multiple frames.
        Args:
            num_frames: Number of frames to sample.
        Returns:
            np.ndarray: The median background frame, or None if failed.
        """
        if not self.running:
            return None
            
        # Let auto-exposure settle
        time.sleep(1.5)
        
        frames = []
        for _ in range(num_frames):
            frame = self.read_frame()
            if frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.033)
                
        if len(frames) < 5:
            return None
            
        bg = np.median(frames, axis=0).astype(np.uint8)
        # Force standard size
        if bg.shape[1] != FRAME_W or bg.shape[0] != FRAME_H:
            bg = cv2.resize(bg, (FRAME_W, FRAME_H))
        return bg

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Returns the most recent frame (1280x720, mirrored) lock-free.
        Returns:
            np.ndarray: The latest frame, or None if not available.
        """
        if self.frame_buffer:
            return self.frame_buffer[0]
        return None

    @property
    def is_running(self):
        return self.running and self.thread is not None and self.thread.is_alive()

    @property
    def last_error(self):
        return self._error

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
