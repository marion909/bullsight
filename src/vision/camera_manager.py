"""
Camera management for dart detection system.

Provides high-level interface to Raspberry Pi Camera Module v3
or USB/Webcams with automatic configuration and error handling.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
import platform
try:
    from picamera2 import Picamera2
except ImportError:
    # Allow import for testing on non-Raspberry Pi systems
    Picamera2 = None
from typing import Optional, Tuple, Union
import logging
import time


logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages camera for dart detection.
    
    Supports:
    - Raspberry Pi Camera Module v3 (via picamera2)
    - USB/Webcams (via OpenCV VideoCapture)
    
    Automatically selects appropriate backend based on platform.
    
    Attributes:
        camera: Camera instance (Picamera2 or cv2.VideoCapture)
        resolution (Tuple[int, int]): Image resolution (width, height)
        is_started (bool): Whether camera is currently running
        use_picamera (bool): Whether using picamera2 backend
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        enable_autofocus: bool = True,
        camera_index: int = 0,
        demo_mode: bool = False
    ):
        """
        Initialize camera manager.
        
        Args:
            resolution: Capture resolution as (width, height)
            enable_autofocus: Enable continuous autofocus (picamera2 only)
            camera_index: Camera device index for USB/Webcam (default: 0)
            demo_mode: Use synthetic frames instead of real camera (for testing)
        """
        self.resolution = resolution
        self.enable_autofocus = enable_autofocus
        self.camera_index = camera_index
        self.demo_mode = demo_mode
        self.camera: Optional[Union[Picamera2, cv2.VideoCapture]] = None
        self.is_started = False
        self._demo_frame: Optional[np.ndarray] = None
        
        # Determine which backend to use
        self.use_picamera = Picamera2 is not None and platform.system() == "Linux" and not demo_mode
        
        if demo_mode:
            backend = "Demo Mode (synthetic frames)"
        else:
            backend = "picamera2" if self.use_picamera else "OpenCV VideoCapture"
        logger.info(f"CameraManager initialized with {backend}, resolution {resolution}")
    
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
            if self.demo_mode:
                self._start_demo_mode()
            elif self.use_picamera:
                self._start_picamera()
            else:
                self._start_opencv_camera()
            
            self.is_started = True
            logger.info("Camera started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise RuntimeError(f"Camera initialization failed: {e}")
    
    def _start_picamera(self) -> None:
        """Start Raspberry Pi Camera Module v3."""
        self.camera = Picamera2()
        
        config = self.camera.create_still_configuration(
            main={"size": self.resolution},
            buffer_count=2
        )
        self.camera.configure(config)
        
        if self.enable_autofocus:
            self.camera.set_controls({"AfMode": 2})  # Continuous autofocus
        
        self.camera.start()
        time.sleep(2)  # Allow camera to stabilize
    
    def _start_opencv_camera(self) -> None:
        """Start USB/Webcam via OpenCV."""
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Enable autofocus if supported
        if self.enable_autofocus:
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Warm up camera
        for _ in range(5):
            self.camera.read()
        
        logger.info(f"OpenCV camera {self.camera_index} initialized")
    
    def _start_demo_mode(self) -> None:
        """Initialize demo mode with synthetic frame."""
        self._demo_frame = self._generate_demo_frame()
        logger.info("Demo mode initialized with synthetic dartboard")
    
    def _generate_demo_frame(self) -> np.ndarray:
        """Generate a synthetic dartboard frame for demo mode."""
        width, height = self.resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate dartboard center and size
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 3
        
        # Draw dartboard circles (from outer to inner)
        # Outer double ring (red/green)
        cv2.circle(frame, (center_x, center_y), max_radius, (0, 100, 0), -1)  # Dark green
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.95), (0, 0, 100), -1)  # Dark red
        
        # Triple ring (red/green)
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.65), (0, 100, 0), -1)
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.60), (0, 0, 100), -1)
        
        # Inner single area
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.35), (200, 200, 200), -1)  # Light gray
        
        # Bull's eye
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.10), (0, 255, 0), -1)  # Green bull
        cv2.circle(frame, (center_x, center_y), int(max_radius * 0.04), (0, 0, 255), -1)  # Red bullseye
        
        # Add text overlay
        cv2.putText(frame, "DEMO MODE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "No camera detected", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return frame
    
    def stop(self) -> None:
        """Stop camera and release resources."""
        if not self.is_started or self.camera is None:
            return
        
        try:
            if self.use_picamera:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
        finally:
            self.camera = None
            self.is_started = False
            logger.info("Camera stopped")
    
    def capture(self) -> np.ndarray:
        """
        Capture single frame from camera.
        
        Returns:
            BGR image as numpy array
            
        Raises:
            RuntimeError: If camera is not started
        """
        if not self.is_started:
            raise RuntimeError("Camera not initialized")
        
        try:
            if self.demo_mode:
                # Return copy of demo frame to prevent modifications
                return self._demo_frame.copy()
            elif self.use_picamera:
                if self.camera is None:
                    raise RuntimeError("Camera not initialized")
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame_bgr
            else:
                if self.camera is None:
                    raise RuntimeError("Camera not initialized")
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    raise RuntimeError("Failed to read frame from camera")
                return frame
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            raise RuntimeError(f"Image capture failed: {e}")
    
    def trigger_autofocus(self) -> None:
        """
        Manually trigger autofocus cycle.
        
        Useful before capturing reference image to ensure sharp focus.
        Note: Only fully supported on picamera2 backend.
        """
        if not self.is_started or self.camera is None:
            raise RuntimeError("Camera not initialized")
        
        if self.use_picamera:
            self.camera.set_controls({"AfTrigger": 0})
            time.sleep(1)
            logger.info("Autofocus triggered (picamera2)")
        else:
            # OpenCV VideoCapture: Re-enable autofocus
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            time.sleep(0.5)
            logger.info("Autofocus triggered (OpenCV)")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "started" if self.is_started else "stopped"
        return f"CameraManager({self.resolution[0]}x{self.resolution[1]}, {status})"
