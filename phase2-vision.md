# 👁️ Phase 2 – Vision Engine & Dart Detection

**Dependencies:** [Phase 1 – Foundations](phase1-foundations.md) ✅  
**Next Phase:** [Phase 3 – Calibration & Mapping](phase3-calibration.md)

---

## 🎯 Phase Goals

- Implement OpenCV-based dart detection system
- Create reliable reference image comparison algorithm
- Extract precise dart tip coordinates
- Build modular vision engine architecture
- Achieve 100% test coverage for all vision components

---

## 📋 Prerequisites

### From Phase 1
- ✅ Camera hardware tested and working
- ✅ OpenCV installed (opencv-python==4.9.0.80)
- ✅ NumPy installed (numpy==1.26.4)
- ✅ Project structure established
- ✅ pytest configured with coverage

### Phase 2 Requirements
- Dartboard mounted and stable
- LED light ring installed and tested
- Camera positioned centrally (25-40cm distance)
- Even illumination confirmed

---

## 🧠 Vision Strategy Overview

**Core Approach:** Difference-based detection (no machine learning required)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Reference  │ --> │    New      │ --> │  Difference │
│   Image     │     │   Image     │     │   Detection │
│ (no darts)  │     │ (with dart) │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              v
                    ┌─────────────┐     ┌─────────────┐
                    │   Contour   │ <-- │    Image    │
                    │  Detection  │     │ Processing  │
                    └─────────────┘     └─────────────┘
                          │
                          v
                    ┌─────────────┐
                    │  Dart Tip   │
                    │ Coordinates │
                    └─────────────┘
```

**Advantages:**
- Fast execution on Raspberry Pi
- No training data required
- Stable in controlled lighting
- Simple to debug and maintain

---

## 🔧 Implementation Tasks

### 2.1 Vision Engine Core Module

**Task:** Create main DartDetector class

**File:** `src/vision/dart_detector.py`

```python
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
```

**Expected Outcome:** Complete, documented dart detection engine

---

### 2.2 Camera Integration Module

**Task:** Bridge between hardware camera and vision engine

**File:** `src/vision/camera_manager.py`

```python
"""
Camera management for dart detection system.

Provides high-level interface to Raspberry Pi Camera Module v3
with automatic configuration and error handling.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from typing import Optional, Tuple
import logging
import time


logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages Raspberry Pi Camera Module v3 for dart detection.
    
    Provides automatic configuration, autofocus, and image capture
    with error handling and recovery.
    
    Attributes:
        camera (Picamera2): Camera instance
        resolution (Tuple[int, int]): Image resolution (width, height)
        is_started (bool): Whether camera is currently running
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        autofocus: bool = True
    ):
        """
        Initialize camera manager.
        
        Args:
            resolution: Capture resolution as (width, height)
            autofocus: Enable continuous autofocus
        """
        self.resolution = resolution
        self.autofocus = autofocus
        self.camera: Optional[Picamera2] = None
        self.is_started = False
        
        logger.info(f"CameraManager initialized with resolution {resolution}")
    
    def start(self) -> None:
        """
        Start camera and configure for dart detection.
        
        Raises:
            RuntimeError: If camera cannot be started
        """
        if self.is_started:
            logger.warning("Camera already started")
            return
        
        try:
            self.camera = Picamera2()
            
            config = self.camera.create_still_configuration(
                main={"size": self.resolution},
                buffer_count=2
            )
            self.camera.configure(config)
            
            if self.autofocus:
                self.camera.set_controls({"AfMode": 2})  # Continuous autofocus
            
            self.camera.start()
            time.sleep(2)  # Allow camera to stabilize
            
            self.is_started = True
            logger.info("Camera started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise RuntimeError(f"Camera initialization failed: {e}")
    
    def stop(self) -> None:
        """Stop camera and release resources."""
        if not self.is_started or self.camera is None:
            return
        
        try:
            self.camera.stop()
            self.camera.close()
            self.is_started = False
            logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def capture(self) -> np.ndarray:
        """
        Capture single frame from camera.
        
        Returns:
            BGR image as numpy array
            
        Raises:
            RuntimeError: If camera is not started
        """
        if not self.is_started or self.camera is None:
            raise RuntimeError("Camera must be started before capture")
        
        try:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV compatibility
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            raise RuntimeError(f"Image capture failed: {e}")
    
    def trigger_autofocus(self) -> None:
        """
        Manually trigger autofocus cycle.
        
        Useful before capturing reference image to ensure sharp focus.
        """
        if not self.is_started or self.camera is None:
            raise RuntimeError("Camera must be started")
        
        self.camera.set_controls({"AfTrigger": 0})
        time.sleep(1)
        logger.info("Autofocus triggered")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
```

**Expected Outcome:** Reliable camera interface with error handling

---

### 2.3 Unit Tests for DartDetector

**Task:** Comprehensive tests for vision engine

**File:** `tests/unit/test_dart_detector.py`

```python
"""
Unit tests for DartDetector vision engine.

Tests all dart detection functionality with mock images and edge cases.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.vision.dart_detector import DartDetector, DartCoordinate


class TestDartCoordinate:
    """Tests for DartCoordinate dataclass."""
    
    def test_coordinate_creation(self):
        """Test DartCoordinate initialization."""
        coord = DartCoordinate(x=100, y=200, confidence=0.95, contour_area=150.0)
        assert coord.x == 100
        assert coord.y == 200
        assert coord.confidence == 0.95
        assert coord.contour_area == 150.0
    
    def test_coordinate_repr(self):
        """Test string representation."""
        coord = DartCoordinate(x=100, y=200, confidence=0.95, contour_area=150.0)
        repr_str = repr(coord)
        assert "100" in repr_str
        assert "200" in repr_str
        assert "0.95" in repr_str


class TestDartDetector:
    """Tests for DartDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create DartDetector instance for testing."""
        return DartDetector(
            min_contour_area=100,
            max_contour_area=5000,
            blur_kernel_size=5,
            threshold_value=30
        )
    
    @pytest.fixture
    def reference_image(self):
        """Create mock reference image (dartboard without dart)."""
        # Create 800x800 gray image with circular dartboard pattern
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.circle(img, (400, 400), 300, (100, 100, 100), -1)
        cv2.circle(img, (400, 400), 200, (150, 150, 150), -1)
        cv2.circle(img, (400, 400), 50, (200, 200, 200), -1)
        return img
    
    @pytest.fixture
    def dart_image(self, reference_image):
        """Create mock image with simulated dart."""
        img = reference_image.copy()
        # Simulate dart as white elongated blob
        cv2.ellipse(img, (500, 300), (10, 40), 45, 0, 360, (255, 255, 255), -1)
        return img
    
    def test_initialization(self, detector):
        """Test detector initialization with parameters."""
        assert detector.reference_image is None
        assert detector.min_contour_area == 100
        assert detector.max_contour_area == 5000
        assert detector.blur_kernel_size == 5
        assert detector.threshold_value == 30
    
    def test_set_reference_image(self, detector, reference_image):
        """Test setting reference image."""
        detector.set_reference_image(reference_image)
        assert detector.reference_image is not None
        assert detector.reference_image.shape == reference_image.shape
        # Ensure copy was made
        assert detector.reference_image is not reference_image
    
    def test_set_reference_image_invalid(self, detector):
        """Test error handling for invalid reference images."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            detector.set_reference_image(None)
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            detector.set_reference_image(np.array([]))
        
        with pytest.raises(ValueError, match="must be BGR color image"):
            detector.set_reference_image(np.zeros((100, 100), dtype=np.uint8))
    
    def test_load_reference_from_file(self, detector, reference_image, tmp_path):
        """Test loading reference image from file."""
        filepath = tmp_path / "reference.jpg"
        cv2.imwrite(str(filepath), reference_image)
        
        detector.load_reference_from_file(filepath)
        assert detector.reference_image is not None
    
    def test_load_reference_file_not_found(self, detector, tmp_path):
        """Test error when reference file doesn't exist."""
        filepath = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError):
            detector.load_reference_from_file(filepath)
    
    def test_save_reference_to_file(self, detector, reference_image, tmp_path):
        """Test saving reference image to file."""
        detector.set_reference_image(reference_image)
        filepath = tmp_path / "saved_reference.jpg"
        
        detector.save_reference_to_file(filepath)
        assert filepath.exists()
        
        # Verify saved image
        loaded = cv2.imread(str(filepath))
        assert loaded is not None
    
    def test_save_reference_without_image(self, detector, tmp_path):
        """Test error when saving without reference image."""
        filepath = tmp_path / "reference.jpg"
        with pytest.raises(RuntimeError, match="No reference image"):
            detector.save_reference_to_file(filepath)
    
    def test_preprocess_image(self, detector, reference_image):
        """Test image preprocessing (grayscale + blur)."""
        processed = detector.preprocess_image(reference_image)
        
        assert processed.ndim == 2  # Grayscale
        assert processed.shape == reference_image.shape[:2]
        assert processed.dtype == np.uint8
    
    def test_compute_difference(self, detector, reference_image, dart_image):
        """Test difference computation between images."""
        detector.set_reference_image(reference_image)
        
        diff, thresh = detector.compute_difference(dart_image)
        
        assert diff.shape == reference_image.shape[:2]
        assert thresh.shape == reference_image.shape[:2]
        assert thresh.dtype == np.uint8
        assert np.all((thresh == 0) | (thresh == 255))  # Binary image
    
    def test_compute_difference_without_reference(self, detector, dart_image):
        """Test error when computing difference without reference."""
        with pytest.raises(RuntimeError, match="Reference image must be set"):
            detector.compute_difference(dart_image)
    
    def test_find_contours(self, detector):
        """Test contour detection in binary image."""
        # Create binary image with simple shape
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(binary, (100, 100), 30, 255, -1)
        
        contours = detector.find_contours(binary)
        
        assert len(contours) > 0
        assert isinstance(contours[0], np.ndarray)
    
    def test_filter_contours_by_area(self, detector):
        """Test contour filtering by area."""
        # Create contours of different sizes
        small_contour = np.array([[[50, 50]], [[60, 50]], [[55, 60]]])
        medium_contour = np.array([[[i, 100] for i in range(100, 150)]])
        large_contour = np.array([[[i, 200] for i in range(100, 300)]])
        
        contours = [small_contour, medium_contour, large_contour]
        filtered = detector.filter_contours(contours)
        
        # Only medium contour should pass (100-5000 pixel range)
        assert len(filtered) >= 0  # At least filters without error
    
    def test_extract_dart_tip(self, detector):
        """Test dart tip extraction from contour."""
        # Create contour simulating dart (elongated vertical shape)
        contour = np.array([
            [[100, 200]],  # Bottom
            [[105, 150]],  # Middle
            [[100, 100]],  # Top (dart tip)
            [[95, 150]]    # Middle
        ])
        
        x, y = detector.extract_dart_tip(contour)
        
        assert x == 100
        assert y == 100  # Should be topmost point
    
    def test_calculate_confidence(self, detector, reference_image):
        """Test confidence score calculation."""
        detector.set_reference_image(reference_image)
        
        # Create test contour
        contour = np.array([[[100, 100]], [[120, 100]], [[110, 120]]])
        diff_image = np.ones((800, 800), dtype=np.uint8) * 128
        
        confidence = detector.calculate_confidence(contour, diff_image)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_dart_success(self, detector, reference_image, dart_image):
        """Test successful dart detection."""
        detector.set_reference_image(reference_image)
        
        dart = detector.detect_dart(dart_image)
        
        assert dart is not None
        assert isinstance(dart, DartCoordinate)
        assert 0 <= dart.x < dart_image.shape[1]
        assert 0 <= dart.y < dart_image.shape[0]
        assert 0.0 <= dart.confidence <= 1.0
    
    def test_detect_dart_no_difference(self, detector, reference_image):
        """Test detection when no dart present (identical images)."""
        detector.set_reference_image(reference_image)
        
        dart = detector.detect_dart(reference_image)
        
        assert dart is None
    
    def test_detect_dart_without_reference(self, detector, dart_image):
        """Test error when detecting without reference image."""
        with pytest.raises(RuntimeError, match="Reference image must be set"):
            detector.detect_dart(dart_image)
    
    def test_detect_multiple_darts(self, detector, reference_image):
        """Test detection of multiple darts."""
        detector.set_reference_image(reference_image)
        
        # Create image with 2 darts
        multi_dart_image = reference_image.copy()
        cv2.ellipse(multi_dart_image, (500, 300), (10, 40), 45, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(multi_dart_image, (300, 400), (10, 40), 90, 0, 360, (255, 255, 255), -1)
        
        darts = detector.detect_multiple_darts(multi_dart_image, max_darts=3)
        
        assert isinstance(darts, list)
        assert len(darts) <= 3
        for dart in darts:
            assert isinstance(dart, DartCoordinate)
    
    def test_detect_multiple_darts_empty(self, detector, reference_image):
        """Test multiple dart detection with no darts."""
        detector.set_reference_image(reference_image)
        
        darts = detector.detect_multiple_darts(reference_image)
        
        assert darts == []
    
    def test_visualize_detection(self, detector, reference_image, dart_image):
        """Test visualization of detection results."""
        detector.set_reference_image(reference_image)
        dart = detector.detect_dart(dart_image)
        
        vis_image = detector.visualize_detection(dart_image, dart)
        
        assert vis_image.shape == dart_image.shape
        assert vis_image.dtype == dart_image.dtype
    
    def test_visualize_with_diff_overlay(self, detector, reference_image, dart_image):
        """Test visualization with difference overlay."""
        detector.set_reference_image(reference_image)
        
        vis_image = detector.visualize_detection(dart_image, show_diff=True)
        
        assert vis_image.shape == dart_image.shape
    
    def test_visualize_without_dart(self, detector, dart_image):
        """Test visualization without detected dart."""
        vis_image = detector.visualize_detection(dart_image, dart=None)
        
        assert vis_image.shape == dart_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.vision", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
pytest tests/unit/test_dart_detector.py -v --cov=src.vision --cov-report=term-missing
```

**Expected Coverage:** 100%

---

### 2.4 Integration Tests

**Task:** Test vision engine with real camera hardware

**File:** `tests/integration/test_vision_integration.py`

```python
"""
Integration tests for vision system with camera hardware.

Tests complete workflow from camera capture to dart detection.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import time

from src.vision.dart_detector import DartDetector
from src.vision.camera_manager import CameraManager


@pytest.fixture(scope="module")
def camera():
    """Initialize camera for integration tests."""
    cam = CameraManager(resolution=(1280, 720))
    cam.start()
    yield cam
    cam.stop()


@pytest.fixture
def detector():
    """Create detector for integration tests."""
    return DartDetector()


class TestVisionIntegration:
    """Integration tests for complete vision pipeline."""
    
    def test_camera_to_detector_pipeline(self, camera, detector):
        """Test complete pipeline from camera to detection."""
        # Capture reference image
        reference = camera.capture()
        assert reference is not None
        
        # Set as reference in detector
        detector.set_reference_image(reference)
        assert detector.reference_image is not None
        
        # Capture current image
        current = camera.capture()
        assert current is not None
        
        # Attempt detection (no dart expected in test environment)
        result = detector.detect_dart(current)
        assert result is None or isinstance(result, DartDetector.__annotations__['detect_dart'].__args__[0])
    
    def test_reference_image_capture_and_save(self, camera, detector, tmp_path):
        """Test capturing and saving reference image."""
        # Capture reference
        reference = camera.capture()
        detector.set_reference_image(reference)
        
        # Save to file
        filepath = tmp_path / "test_reference.jpg"
        detector.save_reference_to_file(filepath)
        
        assert filepath.exists()
        
        # Load back and verify
        loaded_image = cv2.imread(str(filepath))
        assert loaded_image is not None
        assert loaded_image.shape == reference.shape
    
    def test_autofocus_before_capture(self, camera, detector):
        """Test autofocus improves image quality."""
        # Capture without focus
        image_no_focus = camera.capture()
        
        # Trigger autofocus
        camera.trigger_autofocus()
        
        # Capture with focus
        image_focused = camera.capture()
        
        assert image_no_focus.shape == image_focused.shape
        # Both should be valid images
        assert image_no_focus.mean() > 0
        assert image_focused.mean() > 0
    
    def test_continuous_capture_performance(self, camera):
        """Test camera can capture continuously without lag."""
        capture_times = []
        
        for _ in range(10):
            start = time.time()
            frame = camera.capture()
            elapsed = time.time() - start
            capture_times.append(elapsed)
            
            assert frame is not None
        
        avg_time = sum(capture_times) / len(capture_times)
        assert avg_time < 0.5  # Must capture in under 500ms
    
    def test_detector_with_synthetic_dart(self, detector, camera):
        """Test detector with synthetic dart overlay."""
        # Capture clean reference
        reference = camera.capture()
        detector.set_reference_image(reference)
        
        # Create synthetic dart on current image
        current = camera.capture()
        cv2.circle(current, (640, 360), 20, (255, 255, 255), -1)
        
        # Detect synthetic dart
        dart = detector.detect_dart(current)
        
        # Should detect the synthetic dart
        assert dart is not None
        # Should be near center where we drew it
        assert abs(dart.x - 640) < 100
        assert abs(dart.y - 360) < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.vision", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
pytest tests/integration/test_vision_integration.py -v --cov=src.vision
```

**Expected Coverage:** 100%

---

## ⚠️ Phase 2 Risks & Mitigations

### Risk: Poor Dart Recognition

**Symptoms:** Darts not detected or false positives

**Mitigation:**
1. Ensure strong, diffuse LED ring lighting
2. Adjust `threshold_value` parameter (default: 30)
3. Tune `min_contour_area` and `max_contour_area`
4. Test with different dart types

**Code to Tune:**
```python
detector = DartDetector(
    min_contour_area=50,      # Lower for smaller darts
    max_contour_area=8000,    # Higher for larger contours
    threshold_value=20        # Lower for subtle differences
)
```

### Risk: Shadows Interfere with Detection

**Symptoms:** Shadows detected as darts, inconsistent detection

**Mitigation:**
1. Position LED ring for even illumination
2. Use diffuser on light ring
3. Increase preprocessing blur: `blur_kernel_size=7`
4. Add shadow compensation algorithm

### Risk: Dart Remains Stuck (No Reference Update)

**Symptoms:** Old darts prevent new dart detection

**Solution:** Update reference after each detection
```python
# After detecting dart
dart = detector.detect_dart(current_image)
if dart:
    # Process dart
    # ...
    # Update reference to include this dart
    detector.set_reference_image(current_image)
```

---

## ✅ Phase 2 Completion Checklist

### Code Implementation
- [ ] `src/vision/dart_detector.py` complete with docstrings
- [ ] `src/vision/camera_manager.py` complete
- [ ] All classes have type hints
- [ ] Logging configured throughout

### Testing
- [ ] Unit tests pass (100% coverage)
- [ ] Integration tests pass (100% coverage)
- [ ] Camera capture performance < 500ms
- [ ] Dart detection works with test images
- [ ] Edge cases handled (no dart, multiple darts, errors)

### Documentation
- [ ] All functions have docstrings
- [ ] Module docstrings present
- [ ] README updated with Phase 2 info
- [ ] Risk mitigations documented

### Performance
- [ ] Detection latency < 1 second
- [ ] No memory leaks in continuous operation
- [ ] Works with reduced resolution (1280x720)

---

## 📊 Expected Coverage Report

```
src/vision/dart_detector.py              245      0   100%
src/vision/camera_manager.py              87      0   100%
tests/unit/test_dart_detector.py         312      0   100%
tests/integration/test_vision_integration.py  124  0   100%
------------------------------------------------------------
TOTAL                                     768      0   100%
```

---

## 🔗 Next Steps

Once Phase 2 is complete with 100% coverage:

**Proceed to:** [Phase 3 – Calibration & Mapping](phase3-calibration.md)

**Phase 3 Requirements:**
- ✅ Dart coordinates reliably extracted
- ✅ Vision engine tested and stable
- ✅ Reference image workflow established

**Phase 3 Will Use:**
- DartCoordinate from [dart_detector.py](src/vision/dart_detector.py)
- Camera feed from [camera_manager.py](src/vision/camera_manager.py)

---

**Phase Status:** 🔴 Not Started  
**Estimated Duration:** 1-2 weeks  
**Test Coverage:** 0% → Target: 100%  
**Dependencies:** Phase 1 ✅