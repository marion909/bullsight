"""
Additional unit tests for edge cases in camera_manager and dart_detector.

Ensures 100% test coverage.

Author: Mario Neuhauser
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector


class TestCameraManagerEdgeCases:
    """Edge case tests for CameraManager."""
    
    @pytest.fixture
    def mock_picamera(self):
        """Mock Picamera2 for testing."""
        with patch('src.vision.camera_manager.Picamera2') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_start_failure_exception_handling(self, mock_picamera):
        """Test exception handling during camera start."""
        mock_picamera.create_still_configuration.side_effect = Exception("Hardware error")
        
        manager = CameraManager()
        with pytest.raises(RuntimeError, match="Camera initialization failed"):
            manager.start()
    
    def test_stop_with_exception(self, mock_picamera):
        """Test stop with error during cleanup."""
        mock_picamera.stop.side_effect = Exception("Stop error")
        
        manager = CameraManager()
        manager.start()
        manager.stop()  # Should not raise, just log
        
        # Should still set camera to None despite error
        assert manager.camera is None
    
    def test_capture_with_exception(self, mock_picamera):
        """Test capture with camera error."""
        mock_picamera.capture_array.side_effect = Exception("Capture error")
        
        manager = CameraManager()
        manager.start()
        
        with pytest.raises(RuntimeError, match="Image capture failed"):
            manager.capture()


class TestDartDetectorEdgeCases:
    """Edge case tests for DartDetector."""
    
    def test_detect_dart_with_no_valid_contours(self):
        """Test detection when contours exist but none are valid."""
        detector = DartDetector()
        
        # Create reference
        ref = np.zeros((800, 800, 3), dtype=np.uint8)
        detector.set_reference_image(ref)
        
        # Create image with tiny differences (below threshold)
        current = ref.copy()
        current[400, 400] = [5, 5, 5]  # Very small difference
        
        dart = detector.detect_dart(current)
        assert dart is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
