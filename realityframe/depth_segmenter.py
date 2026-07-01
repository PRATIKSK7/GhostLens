import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DepthSegmenter:
    def __init__(self):
        self.is_active = False
        self.model = None
        self.transform = None
        self.device = None
        self.loaded = False

    def toggle(self):
        self.is_active = not self.is_active
        # Lazy load MiDaS model when first activated
        if self.is_active and not self.loaded:
            self.load_model()

    def load_model(self):
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not installed. Cannot load MiDaS model.")
            print("Run: pip install torch torchvision timm")
            self.is_active = False
            return

        print("⏳ Loading MiDaS model...")
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", 
                "MiDaS_small",  # Use small for real-time speed
                trust_repo=True
            )
            self.model.eval()
            
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", 
                "transforms",
                trust_repo=True
            )
            self.transform = midas_transforms.small_transform
            
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()  # Apple Silicon
                else "cpu"
            )
            self.model.to(self.device)
            self.loaded = True
            print("✅ MiDaS loaded on:", self.device)
        except Exception as e:
            print(f"❌ Failed to load MiDaS model: {e}")
            self.is_active = False

    def get_foreground_mask(self, frame, depth_threshold=0.4):
        """
        Returns binary mask where True = foreground (person)
        depth_threshold: 0.0-1.0, lower = only very close objects
        """
        if not self.loaded:
            # Fallback to a zero mask if not loaded
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        
        with torch.no_grad():
            depth = self.model(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        depth_np = depth.cpu().numpy()
        
        # Normalize depth map
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max == depth_min:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
            
        depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        
        # High depth value = close to camera = foreground
        fg_mask = (depth_norm > depth_threshold).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Smooth edges
        fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)
        
        return fg_mask
    
    def apply_invisibility(self, frame, bg_frame, mask):
        """Apply invisibility using depth-based mask"""
        alpha = mask.astype(np.float32) / 255.0
        result = frame.copy()
        for c in range(3):
            result[:,:,c] = (
                alpha * frame[:,:,c] + 
                (1 - alpha) * bg_frame[:,:,c]
            ).astype(np.uint8)
        return result
