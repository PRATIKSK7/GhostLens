import cv2
import numpy as np
import os

class AROverlay:
    def __init__(self, overlay_image_path="assets/creeper.png"):
        """
        Initializes the AR Overlay module.
        Loads the overlay image and sets up ArUco detection.
        """
        self.overlay_image = None
        if os.path.exists(overlay_image_path):
            # Load the overlay image with alpha channel if possible
            self.overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
        else:
            print(f"Warning: Could not find overlay image at {overlay_image_path}")

        # Setup ArUco dictionary and parameters
        # We'll use a common dictionary like DICT_6X6_250
        # In newer OpenCV versions (>4.7), the API has changed slightly.
        # We will try the new API, fallback to old if necessary.
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            # Fallback for older OpenCV versions
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

    def apply_overlay(self, frame):
        """
        Detects ArUco marker in the frame and overlays the texture.
        """
        if self.overlay_image is None:
            return frame

        # Detect markers
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(corners) > 0:
            # We'll just use the first detected marker
            marker_corners = corners[0][0] # 4 corners: top-left, top-right, bottom-right, bottom-left
            
            h, w = self.overlay_image.shape[:2]
            # Define corners of the overlay image
            src_points = [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ]
            valid_src = []
            for p in src_points:
                if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                    valid_src.append([float(p[0]), float(p[1])])
            pts_src = np.array(valid_src, dtype=np.float32)
            
            # Destination corners on the frame (where the marker is)
            valid_dst = []
            for p in marker_corners:
                if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2 and isinstance(p[0], (int, float, np.number)) and isinstance(p[1], (int, float, np.number)):
                    valid_dst.append([float(p[0]), float(p[1])])
                    
            if len(valid_dst) < 4:
                return frame
            pts_dst = np.array(valid_dst, dtype=np.float32)
            
            # Calculate perspective transform matrix
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            
            # Warp the overlay image
            warped_overlay = cv2.warpPerspective(self.overlay_image, matrix, (frame.shape[1], frame.shape[0]))
            
            # Blend overlay onto the frame
            # If the image has an alpha channel
            if self.overlay_image.shape[2] == 4:
                # We need to warp the alpha channel separately to use as a mask
                alpha_channel = self.overlay_image[:, :, 3]
                warped_alpha = cv2.warpPerspective(alpha_channel, matrix, (frame.shape[1], frame.shape[0]))
                
                # Normalize alpha to 0-1
                mask = warped_alpha / 255.0
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                mask_inv = 1.0 - mask
                
                # Extract BGR channels from warped overlay
                warped_bgr = warped_overlay[:, :, :3]
                
                # Blend (we must make sure frame and warped_bgr are float for multiplication)
                frame_float = frame.astype(float)
                warped_bgr_float = warped_bgr.astype(float)
                
                blended = (mask * warped_bgr_float) + (mask_inv * frame_float)
                frame = blended.astype(np.uint8)
            else:
                # If no alpha, just create a simple binary mask from warped image
                gray_warped = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                
                # Black-out the area of overlay in frame
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                # Take only region of overlay from overlay image
                overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)
                
                frame = cv2.add(frame_bg, overlay_fg)
                
        return frame
