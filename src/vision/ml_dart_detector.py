"""
ML-based dart detection using YOLOv8.

Uses a trained YOLOv8 model to detect dart tips directly on the dartboard image.
Model trained on DartsVision dataset (99% precision, 98% recall).
Supports 5 detection classes:
  - dart: Dart tip (what we score)
  - cal_top, cal_right, cal_bottom, cal_left: Calibration points (for perspective transform)

Author: Mario Neuhauser
Source: https://github.com/mohamedamineyoukaoui-ops/DartsVision-
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. ML dart detection disabled.")


class MLDartDetector:
    """
    ML-based dart detector using YOLOv8.
    
    Uses a model trained on DartsVision dataset.
    Detects dart tips on dartboard images with ~99% precision.
    
    More robust than classical computer vision methods for:
    - Varying lighting conditions
    - Different dart types and colors
    - Angled camera views
    - Partial occlusions
    """
    
    # Detection classes from DartsVision model
    DART_CLASS = 0
    CAL_TOP_CLASS = 1
    CAL_RIGHT_CLASS = 2
    CAL_BOTTOM_CLASS = 3
    CAL_LEFT_CLASS = 4
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize ML dart detector.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file).
                       If None, tries to load trained model from models/ folder.
                       Falls back to YOLOv8n base model if custom model not found.
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_source = None
        
        if not YOLO_AVAILABLE:
            logger.error("YOLOv8 not available. Install with: pip install ultralytics torch")
            return
        
        # Load model
        try:
            # Try custom DartsVision model first
            if model_path is None:
                # Look for trained DartsVision model
                default_paths = [
                    Path(__file__).parent.parent.parent / "models" / "deepdarts_finetuned.pt",  # Finetuned model (priority)
                    Path(__file__).parent.parent.parent / "bullsight_training" / "finetuned_model" / "weights" / "best.pt",
                    Path(__file__).parent.parent.parent / "bullsight_training" / "dart_detector_v1" / "weights" / "best.pt",
                    Path(__file__).parent.parent.parent / "models" / "deepdarts_trained.pt",
                    Path(__file__).parent.parent.parent / "models" / "deepdarts_s_best.pt",
                    Path(__file__).parent.parent.parent / "models" / "deepdarts_best.pt",
                    Path("bullsight_training/finetuned_model/weights/best.pt"),
                    Path("models/deepdarts_finetuned.pt"),
                ]
                for path in default_paths:
                    if path.exists():
                        model_path = str(path)
                        break
            
            if model_path and Path(model_path).exists():
                # Determine model type
                if 'finetuned' in model_path:
                    model_source = "Custom finetuned (trained with your darts)"
                else:
                    model_source = "DartsVision trained (synthetic data)"
                
                logger.info(f"Loading trained dart model from {model_path}")
                self.model = YOLO(model_path)
                self.model_source = model_source
            else:
                # Fall back to YOLOv8 nano base model
                logger.warning("No trained dart model found - using YOLOv8n base model")
                self.model = YOLO('yolov8n.pt')
                self.model_source = "YOLOv8n base (generic COCO model)"
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def is_available(self) -> bool:
        """Check if ML detection is available."""
        return self.model is not None
    
    def get_model_info(self) -> str:
        """Get information about loaded model."""
        return self.model_source if self.model_source else "Unknown"
    
    def detect(self, image: np.ndarray, 
               reference_image: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
        """
        Detect dart tip position in image.
        
        Args:
            image: Current image with dart
            reference_image: Optional reference image without dart (not used in ML approach)
        
        Returns:
            (x, y) pixel coordinates of dart tip, or None if no dart detected
        """
        if not self.is_available():
            logger.error("ML detector not available")
            return None
        
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            # Get detections
            if len(results) == 0 or len(results[0].boxes) == 0:
                logger.debug("No objects detected")
                return None
            
            # Get detections
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            # Filter for dart class (class 0) and confidence threshold
            valid_idx = (classes == self.DART_CLASS) & (confidences >= self.confidence_threshold)
            if not valid_idx.any():
                logger.debug(f"No darts detected above confidence threshold {self.confidence_threshold}")
                return None
            
            # Get best dart detection
            valid_confs = confidences[valid_idx]
            best_valid_idx = valid_idx.copy()
            best_valid_idx[valid_idx] = False  # Reset all
            best_valid_idx[np.where(valid_idx)[0][valid_confs.argmax()]] = True
            
            best_box = boxes.xyxy[best_valid_idx].cpu().numpy()[0]
            best_conf = confidences[best_valid_idx][0]
            
            # Calculate dart tip position (center of bounding box)
            x = int((best_box[0] + best_box[2]) / 2)
            y = int((best_box[1] + best_box[3]) / 2)
            
            logger.info(f"Detected dart at ({x}, {y}) with confidence {best_conf:.2f}")
            return (x, y)
            
        except Exception as e:
            logger.error(f"Error during ML dart detection: {e}")
            return None
    
    def detect_multiple(self, image: np.ndarray, 
                       max_darts: int = 3) -> List[Tuple[int, int, float, str]]:
        """
        Detect multiple darts in image.
        
        Args:
            image: Image with darts
            max_darts: Maximum number of darts to detect
        
        Returns:
            List of (x, y, confidence, class_name) tuples for each detected dart
            (filters to only "dart" class, ignoring calibration points)
        """
        if not self.is_available():
            return []
        
        try:
            results = self.model(image, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return []
            
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            detections = []
            
            # Filter for dart class only
            for i, (box, conf, cls_id) in enumerate(zip(boxes.xyxy.cpu().numpy(), confidences, classes)):
                if int(cls_id) == self.DART_CLASS and conf >= self.confidence_threshold:
                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    class_name = "dart"
                    detections.append((x, y, float(conf), class_name))
            
            # Sort by confidence and limit
            detections.sort(key=lambda d: d[2], reverse=True)
            
            logger.debug(f"Detected {len(detections)} darts in image")
            
            return detections[:max_darts]
            
        except Exception as e:
            logger.error(f"Error during multiple dart detection: {e}")
            return []
    
    def visualize(self, image: np.ndarray, 
                  dart_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image
            dart_position: Optional dart position to mark
        
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        if dart_position is not None:
            x, y = dart_position
            # Draw crosshair at detection
            cv2.drawMarker(vis_image, (x, y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 30, 3)
            # Draw circle
            cv2.circle(vis_image, (x, y), 10, (0, 255, 0), 2)
        
        return vis_image
    
    # Stereo Detection Methods
    
    def detect_stereo(self, image_left: np.ndarray, image_right: np.ndarray,
                     stereo_calib: Optional['StereoCalibrationData'] = None,
                     max_darts: int = 3) -> List[dict]:
        """
        Detect darts in stereo image pair and compute 3D positions.
        
        Args:
            image_left: Left camera image
            image_right: Right camera image
            stereo_calib: Optional stereo calibration data for 3D triangulation
            max_darts: Maximum number of darts to detect
        
        Returns:
            List of detection dictionaries with:
            {
                'left_2d': (x, y),          # Left image coordinates
                'right_2d': (x, y),         # Right image coordinates
                'confidence_left': float,    # Left detection confidence
                'confidence_right': float,   # Right detection confidence
                'position_3d': (X, Y, Z),   # 3D position (if stereo_calib provided)
                'epipolar_error': float     # Matching error in pixels
            }
        """
        # Detect in both images
        dets_left = self.detect_multiple(image_left, max_darts=max_darts * 2)
        dets_right = self.detect_multiple(image_right, max_darts=max_darts * 2)
        
        if not dets_left or not dets_right:
            logger.debug("No detections in one or both images")
            return []
        
        # Match detections using epipolar constraint
        matches = self.match_stereo_detections(
            dets_left, dets_right, stereo_calib
        )
        
        # If stereo calibration available, triangulate to 3D
        if stereo_calib and matches:
            for match in matches:
                point_3d = self.triangulate_point(
                    match['left_2d'], match['right_2d'], stereo_calib
                )
                match['position_3d'] = point_3d
        
        return matches[:max_darts]
    
    def match_stereo_detections(self, 
                                dets_left: List[Tuple[int, int, float, str]],
                                dets_right: List[Tuple[int, int, float, str]],
                                stereo_calib: Optional['StereoCalibrationData'] = None
                                ) -> List[dict]:
        """
        Match detections between left and right images using epipolar constraint.
        
        Args:
            dets_left: Left image detections [(x, y, conf, class), ...]
            dets_right: Right image detections [(x, y, conf, class), ...]
            stereo_calib: Optional stereo calibration for epipolar constraint
        
        Returns:
            List of matched detection dictionaries
        """
        matches = []
        used_right = set()
        
        # If no calibration, use simple horizontal distance matching
        if stereo_calib is None:
            logger.debug("No stereo calibration - using simple horizontal matching")
            for x_l, y_l, conf_l, _ in dets_left:
                best_match = None
                best_score = float('inf')
                
                for i, (x_r, y_r, conf_r, _) in enumerate(dets_right):
                    if i in used_right:
                        continue
                    
                    # Simple constraint: same y, x_left > x_right (left camera sees object more to left)
                    y_dist = abs(y_l - y_r)
                    x_dist = abs(x_l - x_r)
                    
                    if y_dist < 50:  # Within 50 pixels vertically
                        score = y_dist + x_dist * 0.1
                        if score < best_score:
                            best_score = score
                            best_match = i
                
                if best_match is not None and best_score < 100:
                    x_r, y_r, conf_r, _ = dets_right[best_match]
                    matches.append({
                        'left_2d': (x_l, y_l),
                        'right_2d': (x_r, y_r),
                        'confidence_left': conf_l,
                        'confidence_right': conf_r,
                        'epipolar_error': best_score,
                        'position_3d': None
                    })
                    used_right.add(best_match)
        
        else:
            # Use epipolar constraint with fundamental matrix
            F = stereo_calib.F
            
            for x_l, y_l, conf_l, _ in dets_left:
                pt_left = np.array([x_l, y_l, 1.0])
                
                # Compute epipolar line in right image: l = F * pt_left
                epiline = F @ pt_left
                a, b, c = epiline
                
                best_match = None
                best_error = float('inf')
                
                for i, (x_r, y_r, conf_r, _) in enumerate(dets_right):
                    if i in used_right:
                        continue
                    
                    # Distance from point to epipolar line: |ax + by + c| / sqrt(a^2 + b^2)
                    pt_right = np.array([x_r, y_r, 1.0])
                    distance = abs(a * x_r + b * y_r + c) / np.sqrt(a**2 + b**2)
                    
                    # Also check that detection confidences are similar
                    conf_diff = abs(conf_l - conf_r)
                    
                    # Combined score: epipolar distance + confidence penalty
                    score = distance + conf_diff * 10
                    
                    if distance < 5.0 and score < best_error:  # Within 5px of epipolar line
                        best_error = score
                        best_match = i
                
                if best_match is not None:
                    x_r, y_r, conf_r, _ = dets_right[best_match]
                    matches.append({
                        'left_2d': (x_l, y_l),
                        'right_2d': (x_r, y_r),
                        'confidence_left': conf_l,
                        'confidence_right': conf_r,
                        'epipolar_error': best_error,
                        'position_3d': None
                    })
                    used_right.add(best_match)
        
        logger.debug(f"Matched {len(matches)} dart pairs from {len(dets_left)} left and {len(dets_right)} right")
        return matches
    
    def triangulate_point(self, pt_left: Tuple[int, int], pt_right: Tuple[int, int],
                         stereo_calib: 'StereoCalibrationData') -> Tuple[float, float, float]:
        """
        Triangulate 3D point from stereo correspondences.
        
        Args:
            pt_left: (x, y) in left image
            pt_right: (x, y) in right image
            stereo_calib: Stereo calibration data
        
        Returns:
            (X, Y, Z) 3D coordinates in left camera coordinate system
        """
        # Convert to numpy arrays
        pts_left = np.array([[pt_left[0], pt_left[1]]], dtype=np.float32)
        pts_right = np.array([[pt_right[0], pt_right[1]]], dtype=np.float32)
        
        # Triangulate using OpenCV
        pts_4d = cv2.triangulatePoints(
            stereo_calib.P1, stereo_calib.P2,
            pts_left.T, pts_right.T
        )
        
        # Convert from homogeneous to 3D
        pts_3d = pts_4d[:3] / pts_4d[3]
        
        X, Y, Z = pts_3d[0, 0], pts_3d[1, 0], pts_3d[2, 0]
        
        logger.debug(f"Triangulated point: ({X:.1f}, {Y:.1f}, {Z:.1f})")
        
        return (float(X), float(Y), float(Z))


def create_training_dataset(image_dir: str, output_dir: str):
    """
    Helper function to prepare training dataset for YOLO.
    
    Creates YOLO-format dataset from collected dart images.
    
    Args:
        image_dir: Directory containing dart images
        output_dir: Output directory for YOLO dataset
    """
    # TODO: Implement dataset preparation
    # - Convert images to YOLO format
    # - Create train/val split
    # - Generate YAML config
    logger.info("Dataset preparation tool - To be implemented")
    logger.info("Use tools like LabelImg or Roboflow to annotate dart tips")
    pass


def train_model(data_yaml: str, epochs: int = 100, model_size: str = 'n'):
    """
    Train or fine-tune YOLOv8 model for dart detection.
    
    Args:
        data_yaml: Path to YOLO dataset YAML config
        epochs: Number of training epochs
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO not available for training")
        return
    
    try:
        # Load base model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            device='cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
            patience=50,
            save=True,
            project='bullsight_training',
            name='dart_detector'
        )
        
        logger.info("Training complete!")
        logger.info(f"Model saved to: {results.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = MLDartDetector()
    
    if detector.is_available():
        logger.info("ML Dart Detector initialized successfully")
        logger.info("To use: detector.detect(image)")
    else:
        logger.error("ML Dart Detector not available")
        logger.info("Install dependencies: pip install -r requirements-ml.txt")
