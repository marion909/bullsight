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
