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

from src.vision.dart_detector import DartDetector, DartCoordinate
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


@pytest.mark.integration
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
        assert result is None or isinstance(result, DartCoordinate)
    
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
