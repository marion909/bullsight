"""
Dart detection engine using OpenCV difference-based approach.

This module provides the core computer vision functionality for detecting
dart impacts on a dartboard through reference image comparison.

Author: Mario Neuhauser
License: MIT
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DartCoordinate:
    """
    Represents a detected dart coordinate.
    
    Attributes:
        x (int): X pixel coordinate of dart tip
        y (int): Y pixel coordinate of dart tip
        confidence (float): Detection confidence score (0.0-1.0)
        contour_area (float): Area of detected contour in pixels
    """
    x: int
    y: int
    confidence: float
    contour_area: float
    
    def __repr__(self) -> str:
        return f"DartCoordinate(x={self.x}, y={self.y}, conf={self.confidence:.2f})"


class DartDetector:
    """
    Computer vision engine for detecting dart impacts.
    
    Uses difference-based detection comparing reference and current images
    to identify new dart positions on the dartboard.
    
    Attributes:
        reference_image (np.ndarray): Clean dartboard image without darts
        min_contour_area (int): Minimum contour area to consider as dart
        max_contour_area (int): Maximum contour area to consider as dart
        blur_kernel_size (int): Gaussian blur kernel size for noise reduction
        threshold_value (int): Binary threshold for difference detection
    """
    
    def __init__(
        self,
        min_contour_area: int = 100,
        max_contour_area: int = 5000,
        blur_kernel_size: int = 5,
        threshold_value: int = 30
    ):
        """
        Initialize dart detector with configurable parameters.
        
        Args:
            min_contour_area: Minimum pixels to consider as dart (default: 100)
            max_contour_area: Maximum pixels to consider as dart (default: 5000)
            blur_kernel_size: Gaussian blur kernel size (default: 5)
            threshold_value: Binary threshold for difference (default: 30)
        """
        self.reference_image: Optional[np.ndarray] = None
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.blur_kernel_size = blur_kernel_size
        self.threshold_value = threshold_value
        
        logger.info(f"DartDetector initialized with contour range: "
                   f"{min_contour_area}-{max_contour_area} pixels")
    
    def set_reference_image(self, image: np.ndarray) -> None:
        """
        Set the reference image (dartboard without darts).
        
        Args:
            image: OpenCV image array (BGR format)
            
        Raises:
            ValueError: If image is invalid or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Reference image cannot be None or empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Reference image must be BGR color image")
        
        self.reference_image = image.copy()
        logger.info(f"Reference image set: {image.shape}")
    
    def load_reference_from_file(self, filepath: Path) -> None:
        """
        Load reference image from file.
        
        Args:
            filepath: Path to reference image file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read as image
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Reference image not found: {filepath}")
        
        image = cv2.imread(str(filepath))
        if image is None:
            raise ValueError(f"Could not read image file: {filepath}")
        
        self.set_reference_image(image)
        logger.info(f"Reference image loaded from: {filepath}")
    
    def save_reference_to_file(self, filepath: Path) -> None:
        """
        Save current reference image to file.
        
        Args:
            filepath: Path where to save the image
            
        Raises:
            RuntimeError: If no reference image is set
        """
        if self.reference_image is None:
            raise RuntimeError("No reference image to save")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), self.reference_image)
        logger.info(f"Reference image saved to: {filepath}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection (grayscale + blur).
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )
        return blurred
    
    def compute_difference(
        self,
        current_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute difference between reference and current image.
        
        Args:
            current_image: Current BGR image to compare
            
        Returns:
            Tuple of (difference image, thresholded binary image)
            
        Raises:
            RuntimeError: If no reference image is set
        """
        if self.reference_image is None:
            raise RuntimeError("Reference image must be set before detection")
        
        # Preprocess both images
        ref_processed = self.preprocess_image(self.reference_image)
        cur_processed = self.preprocess_image(current_image)
        
        # Compute absolute difference
        diff = cv2.absdiff(ref_processed, cur_processed)
        
        # Apply binary threshold
        _, thresh = cv2.threshold(
            diff,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY
        )
        
        return diff, thresh
    
    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in binary difference image.
        
        Args:
            binary_image: Thresholded binary image
            
        Returns:
            List of contours as numpy arrays
        """
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours by area to identify dart candidates.
        
        Args:
            contours: List of all detected contours
            
        Returns:
            List of filtered contours within size range
        """
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                filtered.append(contour)
        
        logger.debug(f"Filtered {len(filtered)}/{len(contours)} contours "
                    f"by area ({self.min_contour_area}-{self.max_contour_area})")
        return filtered
    
    def extract_dart_tip(self, contour: np.ndarray) -> Tuple[int, int]:
        """
        Extract dart tip coordinate from contour.
        
        Strategy: Find the topmost point (lowest y-value) which typically
        corresponds to the dart tip pointing upward into the board.
        
        Args:
            contour: OpenCV contour array
            
        Returns:
            Tuple of (x, y) coordinates of dart tip
        """
        # Find topmost point (minimum y)
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        return topmost[0], topmost[1]
    
    def calculate_confidence(
        self,
        contour: np.ndarray,
        diff_image: np.ndarray
    ) -> float:
        """
        Calculate detection confidence based on contour properties.
        
        Args:
            contour: Detected contour
            diff_image: Difference image
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        area = cv2.contourArea(contour)
        
        # Create mask for contour
        mask = np.zeros(diff_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Calculate mean intensity in contour region
        mean_intensity = cv2.mean(diff_image, mask=mask)[0]
        
        # Normalize confidence (higher intensity = higher confidence)
        intensity_score = min(mean_intensity / 255.0, 1.0)
        
        # Combine with area score (prefer mid-range areas)
        ideal_area = (self.min_contour_area + self.max_contour_area) / 2
        area_score = 1.0 - abs(area - ideal_area) / ideal_area
        area_score = max(0.0, min(area_score, 1.0))
        
        # Weighted combination
        confidence = 0.7 * intensity_score + 0.3 * area_score
        return confidence
    
    def detect_dart(self, current_image: np.ndarray) -> Optional[DartCoordinate]:
        """
        Detect dart in current image compared to reference.
        
        Args:
            current_image: Current BGR image with dart
            
        Returns:
            DartCoordinate if dart detected, None otherwise
            
        Raises:
            RuntimeError: If no reference image is set
        """
        # Compute difference
        diff, thresh = self.compute_difference(current_image)
        
        # Find and filter contours
        contours = self.find_contours(thresh)
        filtered_contours = self.filter_contours(contours)
        
        if not filtered_contours:
            logger.info("No dart detected (no valid contours found)")
            return None
        
        # Select largest contour as most likely dart
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        
        # Extract dart tip
        x, y = self.extract_dart_tip(largest_contour)
        
        # Calculate confidence
        confidence = self.calculate_confidence(largest_contour, diff)
        area = cv2.contourArea(largest_contour)
        
        dart = DartCoordinate(
            x=int(x),
            y=int(y),
            confidence=confidence,
            contour_area=float(area)
        )
        
        logger.info(f"Dart detected: {dart}")
        return dart
    
    def detect_multiple_darts(
        self,
        current_image: np.ndarray,
        max_darts: int = 3
    ) -> List[DartCoordinate]:
        """
        Detect multiple darts in current image.
        
        Args:
            current_image: Current BGR image with darts
            max_darts: Maximum number of darts to detect (default: 3)
            
        Returns:
            List of detected DartCoordinate objects, sorted by confidence
        """
        diff, thresh = self.compute_difference(current_image)
        contours = self.find_contours(thresh)
        filtered_contours = self.filter_contours(contours)
        
        if not filtered_contours:
            return []
        
        # Sort by area (descending)
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        darts = []
        for contour in filtered_contours[:max_darts]:
            x, y = self.extract_dart_tip(contour)
            confidence = self.calculate_confidence(contour, diff)
            area = cv2.contourArea(contour)
            
            darts.append(DartCoordinate(
                x=int(x),
                y=int(y),
                confidence=confidence,
                contour_area=float(area)
            ))
        
        # Sort by confidence
        darts.sort(key=lambda d: d.confidence, reverse=True)
        
        logger.info(f"Detected {len(darts)} darts")
        return darts
    
    def visualize_detection(
        self,
        current_image: np.ndarray,
        dart: Optional[DartCoordinate] = None,
        show_diff: bool = False
    ) -> np.ndarray:
        """
        Create visualization of detection results.
        
        Args:
            current_image: Current BGR image
            dart: Detected dart coordinate (if any)
            show_diff: Whether to show difference image overlay
            
        Returns:
            Annotated image with detection visualization
        """
        vis_image = current_image.copy()
        
        if show_diff and self.reference_image is not None:
            diff, _ = self.compute_difference(current_image)
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            vis_image = cv2.addWeighted(vis_image, 0.6, diff_colored, 0.4, 0)
        
        if dart is not None:
            # Draw crosshair at dart tip
            cv2.drawMarker(
                vis_image,
                (dart.x, dart.y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                markerSize=30,
                thickness=3
            )
            
            # Add text annotation
            text = f"({dart.x}, {dart.y}) conf={dart.confidence:.2f}"
            cv2.putText(
                vis_image,
                text,
                (dart.x + 20, dart.y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return vis_image
