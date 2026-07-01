import cv2
import numpy as np

class FocusMode:
    def __init__(self):
        self.selected_points = []
        self.is_selecting = False
        self.is_active = False

    def handle_click(self, event, x, y, flags, param):
        """
        Mouse callback function to collect 4 points for the focus region.
        """
        if self.is_selecting and event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 4:
                self.selected_points.append((x, y))
            # If we reached 4 points, stop selecting automatically
            if len(self.selected_points) == 4:
                self.is_selecting = False

    def reset_selection(self):
        """
        Clears current points and enters selection mode.
        """
        self.selected_points = []
        self.is_selecting = True
        
    def toggle_active(self):
        """
        Toggles focus mode on and off.
        """
        self.is_active = not self.is_active

    def apply_focus_mask(self, frame, bg_frame):
        """
        Keeps the polygon region visible, replaces the rest with the background.
        """
        if not self.is_active or len(self.selected_points) < 4 or bg_frame is None:
            return frame
            
        # Create an empty mask for the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        valid_points = []
        for p in self.selected_points:
            if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                valid_points.append([int(p[0]), int(p[1])])
                
        if len(valid_points) >= 4:
            print("POINTS:", valid_points)
            pts = np.array(valid_points, dtype=np.int32)
            # Fill the selected polygon region with white (255)
            cv2.fillPoly(mask, [pts], 255)
        
        # Composite: Where mask is 255 (inside poly), show frame. Elsewhere, show background.
        result = np.where(mask[:,:,None] == 255, frame, bg_frame)
        return result

    def draw_ui(self, frame):
        """
        Draws the selection points and polygon lines on the frame.
        """
        if self.is_selecting:
            # Draw individual points
            for pt in self.selected_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            
            # Draw lines connecting the points
            if len(self.selected_points) > 1:
                valid_points = []
                for p in self.selected_points:
                    if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                        valid_points.append([int(p[0]), int(p[1])])
                
                if len(valid_points) > 1:
                    print("POINTS:", valid_points)
                    pts = np.array(valid_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # Only close the polygon if we have all 4 points
                    is_closed = len(valid_points) == 4
                    cv2.polylines(frame, [pts], is_closed, (0, 255, 0), 2)
                
        # Draw the final polygon if it's active and selected
        if self.is_active and len(self.selected_points) == 4:
             valid_points = []
             for p in self.selected_points:
                 if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                     valid_points.append([int(p[0]), int(p[1])])
                     
             if len(valid_points) == 4:
                 print("POINTS:", valid_points)
                 pts = np.array(valid_points, np.int32)
                 pts = pts.reshape((-1, 1, 2))
                 cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
