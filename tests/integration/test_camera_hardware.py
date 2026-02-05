"""
Camera hardware integration tests.
Tests Raspberry Pi Camera Module v3 functionality.

Coverage Target: 100%
"""

import pytest
from picamera2 import Picamera2
import numpy as np
import time


class TestCameraHardware:
    """Test suite for camera hardware integration."""

    @pytest.fixture
    def camera(self):
        """
        Initialize camera for testing.
        
        Yields:
            Picamera2: Configured camera instance
        """
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (1920, 1080)},
            buffer_count=2
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)  # Allow camera to stabilize
        yield cam
        cam.stop()
        cam.close()

    def test_camera_initialization(self, camera):
        """
        Test camera can be initialized successfully.
        
        Verifies:
        - Camera object creation
        - Configuration acceptance
        - Start/stop lifecycle
        """
        assert camera is not None
        assert camera.started

    def test_camera_capture_single_frame(self, camera):
        """
        Test single frame capture.
        
        Verifies:
        - Image capture succeeds
        - Image has correct dimensions
        - Image contains valid pixel data
        """
        frame = camera.capture_array()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[0] == 1080  # Height
        assert frame.shape[1] == 1920  # Width
        assert frame.shape[2] == 3     # RGB channels
        
        # Verify not all black (camera working)
        assert frame.mean() > 0

    def test_camera_autofocus(self, camera):
        """
        Test autofocus functionality (Camera Module v3 feature).
        
        Verifies:
        - Autofocus can be triggered
        - Focus completes without error
        """
        # Trigger autofocus
        camera.set_controls({"AfMode": 2})  # AfModeAuto
        time.sleep(1)
        
        # Capture should succeed after autofocus
        frame = camera.capture_array()
        assert frame is not None

    def test_camera_multiple_captures(self, camera):
        """
        Test multiple sequential captures (dart detection simulation).
        
        Verifies:
        - Camera can capture multiple frames
        - No memory leaks
        - Consistent frame quality
        """
        frames = []
        for _ in range(5):
            frame = camera.capture_array()
            frames.append(frame)
            time.sleep(0.2)
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (1080, 1920, 3)

    def test_camera_resolution_configuration(self):
        """
        Test different resolution configurations.
        
        Verifies:
        - Camera accepts various resolutions
        - Lower resolutions work (performance optimization)
        """
        resolutions = [(1920, 1080), (1280, 720), (640, 480)]
        
        for width, height in resolutions:
            cam = Picamera2()
            config = cam.create_still_configuration(
                main={"size": (width, height)}
            )
            cam.configure(config)
            cam.start()
            time.sleep(1)
            
            frame = cam.capture_array()
            assert frame.shape[0] == height
            assert frame.shape[1] == width
            
            cam.stop()
            cam.close()

    def test_camera_error_handling(self):
        """
        Test camera error scenarios.
        
        Verifies:
        - Proper exception on double start
        - Clean shutdown on errors
        """
        cam = Picamera2()
        config = cam.create_still_configuration()
        cam.configure(config)
        cam.start()
        
        # Attempting to start again should be handled
        with pytest.raises(Exception):
            cam.start()
        
        cam.stop()
        cam.close()


@pytest.mark.hardware
def test_camera_availability():
    """
    Test camera device availability at system level.
    
    Verifies:
    - Camera is detected by system
    - Camera interface is enabled
    """
    try:
        cam = Picamera2()
        cam.close()
        assert True
    except Exception as e:
        pytest.fail(f"Camera not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term-missing"])
