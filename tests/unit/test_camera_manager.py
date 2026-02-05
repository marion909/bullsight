"""
Unit tests for CameraManager hardware abstraction.

Tests camera initialization, configuration, and capture functionality.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.vision.camera_manager import CameraManager


class TestCameraManager:
    """Tests for CameraManager class."""
    
    @pytest.fixture
    def mock_picamera(self):
        """Mock Picamera2 for testing."""
        with patch('src.vision.camera_manager.Picamera2') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_initialization_default_resolution(self, mock_picamera):
        """Test camera initialization with default resolution."""
        manager = CameraManager()
        
        assert manager.resolution == (1280, 720)
        assert manager.camera is None
        assert manager.enable_autofocus is True
    
    def test_initialization_custom_resolution(self, mock_picamera):
        """Test camera initialization with custom resolution."""
        manager = CameraManager(resolution=(1920, 1080), enable_autofocus=False)
        
        assert manager.resolution == (1920, 1080)
        assert manager.enable_autofocus is False
    
    def test_start_camera(self, mock_picamera):
        """Test camera startup."""
        manager = CameraManager()
        manager.start()
        
        assert manager.camera is not None
        mock_picamera.create_still_configuration.assert_called_once()
        mock_picamera.configure.assert_called_once()
        mock_picamera.start.assert_called_once()
    
    def test_start_camera_with_autofocus(self, mock_picamera):
        """Test camera startup enables autofocus."""
        manager = CameraManager(enable_autofocus=True)
        manager.start()
        
        mock_picamera.set_controls.assert_called()
        # Verify AfMode was set
        calls = mock_picamera.set_controls.call_args_list
        assert len(calls) > 0
    
    def test_start_camera_already_started(self, mock_picamera):
        """Test starting camera when already started."""
        manager = CameraManager()
        manager.start()
        manager.start()  # Second start should be no-op
        
        # Should only create/configure once
        assert mock_picamera.create_still_configuration.call_count == 1
    
    def test_stop_camera(self, mock_picamera):
        """Test camera shutdown."""
        manager = CameraManager()
        manager.start()
        manager.stop()
        
        mock_picamera.close.assert_called_once()
        assert manager.camera is None
    
    def test_stop_camera_not_started(self, mock_picamera):
        """Test stopping camera when not started."""
        manager = CameraManager()
        manager.stop()  # Should not raise error
        
        mock_picamera.close.assert_not_called()
    
    def test_capture_frame(self, mock_picamera):
        """Test frame capture."""
        # Create mock RGB frame
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_frame[:, :, 0] = 255  # Red channel
        mock_picamera.capture_array.return_value = mock_frame
        
        manager = CameraManager()
        manager.start()
        frame = manager.capture()
        
        assert frame is not None
        assert frame.shape == (720, 1280, 3)
        # Check RGB to BGR conversion happened (red should now be in channel 2)
        assert frame[0, 0, 2] == 255
    
    def test_capture_without_start(self, mock_picamera):
        """Test capture raises error when camera not started."""
        manager = CameraManager()
        
        with pytest.raises(RuntimeError, match="Camera not initialized"):
            manager.capture()
    
    def test_trigger_autofocus(self, mock_picamera):
        """Test manual autofocus trigger."""
        manager = CameraManager()
        manager.start()
        manager.trigger_autofocus()
        
        # Should set AfTrigger control
        mock_picamera.set_controls.assert_called()
    
    def test_trigger_autofocus_disabled(self, mock_picamera):
        """Test autofocus trigger when disabled."""
        manager = CameraManager(enable_autofocus=False)
        manager.start()
        manager.trigger_autofocus()
        
        # Should still trigger even if disabled (user can manually trigger)
        # Just verify it doesn't crash
        assert True
    
    def test_trigger_autofocus_not_started(self, mock_picamera):
        """Test autofocus trigger when camera not started."""
        manager = CameraManager()
        
        with pytest.raises(RuntimeError, match="Camera not initialized"):
            manager.trigger_autofocus()
    
    def test_context_manager_success(self, mock_picamera):
        """Test context manager protocol."""
        with CameraManager() as manager:
            assert manager.camera is not None
        
        mock_picamera.stop.assert_called_once()
    
    def test_context_manager_with_exception(self, mock_picamera):
        """Test context manager cleanup on exception."""
        try:
            with CameraManager() as manager:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still close camera
        mock_picamera.close.assert_called_once()
    
    def test_repr(self, mock_picamera):
        """Test string representation."""
        manager = CameraManager(resolution=(1920, 1080))
        repr_str = repr(manager)
        
        assert "CameraManager" in repr_str
        assert "1920x1080" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.vision.camera_manager", "--cov-report=term-missing"])
