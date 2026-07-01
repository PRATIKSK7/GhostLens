import cv2
import datetime
import os

class VideoRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.writer = None
        self.is_recording = False
        self.filename = ""
        self.frame_count = 0
    
    def start(self, width=1280, height=720, fps=20):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{self.output_dir}/phantomframe_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename, fourcc, fps, (width, height)
        )
        self.is_recording = True
        self.frame_count = 0
        print(f"🔴 Recording: {self.filename}")
    
    def write(self, frame):
        if self.is_recording and self.writer:
            h, w = frame.shape[:2]
            self.writer.write(frame)
            self.frame_count += 1
    
    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        print(f"✅ Saved: {self.filename} ({self.frame_count} frames)")
        return self.filename
    
    def toggle(self, width=1280, height=720):
        if self.is_recording:
            return self.stop()
        else:
            self.start(width, height)
            return None
    
    def draw_indicator(self, frame):
        import time
        if self.is_recording:
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (30, 30), 10, (0,0,255), -1)
                cv2.putText(frame, f"REC {self.frame_count}f",
                    (48, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 2)
        return frame
