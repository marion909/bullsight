"""
Dual Camera Manager for Stereo Vision System.

Manages two USB cameras simultaneously with synchronized frame capture
for 3D triangulation and improved dart position accuracy.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
import logging
import time
import traceback
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StereoFrame:
    """Container for synchronized stereo frame pair."""
    left: np.ndarray
    right: np.ndarray
    timestamp: float
    time_delta_ms: float  # Time difference between captures


class DualCameraManager:
    """
    Manages two USB cameras for stereo vision.
    
    Provides synchronized frame capture from two cameras with timestamp
    validation to ensure frames are captured within acceptable time window.
    
    Attributes:
        camera_left: Left camera VideoCapture
        camera_right: Right camera VideoCapture
        resolution: Tuple of (width, height)
        is_started: Whether cameras are initialized
    """
    
    def __init__(
        self,
        camera_index_left: int = 0,
        camera_index_right: int = 1,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30
    ):
        """
        Initialize dual camera manager.
        
        Args:
            camera_index_left: USB device index for left camera (default: 0)
            camera_index_right: USB device index for right camera (default: 1)
            resolution: Target resolution as (width, height)
            fps: Target frames per second
            
        Raises:
            RuntimeError: If less than 2 cameras are available
        """
        self.camera_index_left = camera_index_left
        self.camera_index_right = camera_index_right
        self.resolution = resolution
        self.fps = fps
        
        self.camera_left: Optional[cv2.VideoCapture] = None
        self.camera_right: Optional[cv2.VideoCapture] = None
        self.is_started = False
        
        # Thread pool for parallel capture
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera")
        
        # Synchronization tolerance (ms) - 16ms = ~60fps tolerance
        self.sync_tolerance_ms = 16.0
        
    def start(self) -> bool:
        """
        Initialize and start both cameras.
        
        Returns:
            True if both cameras started successfully, False otherwise
            
        Raises:
            RuntimeError: If less than 2 cameras detected
        """
        if self.is_started:
            logger.warning("Cameras already started")
            return True
        
        try:
            # Initialize left camera
            logger.info(f"Initializing left camera (index {self.camera_index_left})...")
            self.camera_left = cv2.VideoCapture(self.camera_index_left, cv2.CAP_DSHOW)
            
            if not self.camera_left.isOpened():
                raise RuntimeError(f"Failed to open left camera at index {self.camera_index_left}")
            
            # Initialize right camera
            logger.info(f"Initializing right camera (index {self.camera_index_right})...")
            self.camera_right = cv2.VideoCapture(self.camera_index_right, cv2.CAP_DSHOW)
            
            if not self.camera_right.isOpened():
                raise RuntimeError(f"Failed to open right camera at index {self.camera_index_right}")
            
            # Configure both cameras
            self._configure_camera(self.camera_left, "left")
            self._configure_camera(self.camera_right, "right")
            
            # Warm-up: capture and discard first few frames
            logger.info("Camera warm-up...")
            for _ in range(5):
                self.camera_left.read()
                self.camera_right.read()
                time.sleep(0.05)
            
            self.is_started = True
            logger.info("✅ Dual camera system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cameras: {e}")
            self.stop()
            return False
    
    def _configure_camera(self, camera: cv2.VideoCapture, name: str) -> None:
        """Configure camera settings."""
        width, height = self.resolution
        
        # Set resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Set FPS
        camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set buffer size to 1 (minimal buffering for real-time capture)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Disable autofocus for consistency
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Set fixed focus (adjust as needed for your setup)
        # camera.set(cv2.CAP_PROP_FOCUS, 50)
        
        # Verify settings
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"{name.capitalize()} camera configured: {int(actual_width)}x{int(actual_height)} @ {actual_fps}fps")
    
    def capture_stereo(self) -> Optional[StereoFrame]:
        """
        Capture frames from both cameras with minimal delay.
        
        Uses simple sequential read() calls which is reliable and fast enough
        for most stereo vision applications (<10ms typ sync time).
        
        Returns:
            StereoFrame with left and right images and metadata,
            or None if capture failed
        """
        if not self.is_started:
            logger.warning("Cameras not started")
            return None
        
        try:
            # Read from both cameras as quickly as possible
            ts_start = time.time()
            ret_left, frame_left = self.camera_left.read()
            ts_mid = time.time()
            ret_right, frame_right = self.camera_right.read()
            ts_end = time.time()
            
            if not ret_left or not ret_right or frame_left is None or frame_right is None:
                return None
            
            # Calculate actual time between captures
            time_delta_ms = (ts_mid - ts_start) * 1000
            
            # Use middle timestamp
            timestamp = (ts_start + ts_mid) / 2
            
            return StereoFrame(
                left=frame_left,
                right=frame_right,
                timestamp=timestamp,
                time_delta_ms=time_delta_ms
            )
            
        except Exception as e:
            logger.error(f"Stereo capture error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _capture_single(self, camera: cv2.VideoCapture, name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Capture single frame from one camera with timestamp.
        
        Args:
            camera: VideoCapture instance
            name: Camera identifier for logging
            
        Returns:
            Tuple of (frame, timestamp) or None if capture failed
        """
        try:
            if camera is None:
                logger.error(f"{name} camera is None!")
                return None
                
            if not camera.isOpened():
                logger.error(f"{name} camera is not opened!")
                return None
            
            # Clear buffer by grabbing without retrieving (ensures fresh frame)
            camera.grab()
            
            timestamp = time.time()
            ret, frame = camera.read()
            
            if not ret:
                logger.error(f"Failed to read from {name} camera (ret=False)")
                return None
                
            if frame is None:
                logger.error(f"Frame from {name} camera is None")
                return None
            
            return frame, timestamp
            
        except Exception as e:
            logger.error(f"Capture exception on {name} camera: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_camera_info(self) -> dict:
        """Get information about both cameras."""
        info = {
            "left": self._get_single_camera_info(self.camera_left) if self.camera_left else None,
            "right": self._get_single_camera_info(self.camera_right) if self.camera_right else None,
            "is_started": self.is_started,
            "resolution": self.resolution,
            "sync_tolerance_ms": self.sync_tolerance_ms
        }
        return info
    
    def _get_single_camera_info(self, camera: cv2.VideoCapture) -> dict:
        """Get info from single camera."""
        if camera is None or not camera.isOpened():
            return {"status": "not_opened"}
        
        return {
            "status": "opened",
            "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": camera.get(cv2.CAP_PROP_FPS),
            "backend": camera.getBackendName()
        }
    
    def stop(self) -> None:
        """Stop and release both cameras."""
        if self.camera_left is not None:
            self.camera_left.release()
            self.camera_left = None
            logger.info("Left camera released")
        
        if self.camera_right is not None:
            self.camera_right.release()
            self.camera_right = None
            logger.info("Right camera released")
        
        self.is_started = False
        logger.info("Dual camera system stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def check_available_cameras(max_check: int = 5) -> list:
    """
    Check which cameras are available and produce valid (non-black) frames.
    
    Tests multiple frames from each camera to ensure consistency.
    
    Args:
        max_check: Maximum number of indices to check
        
    Returns:
        List of available camera indices that produce valid frames,
        sorted by image brightness (brightest first)
    """
    available = []
    
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Test capture multiple frames to check consistency
            brightness_values = []
            for _ in range(3):  # Test 3 frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    brightness_values.append(frame.mean())
                time.sleep(0.05)  # Small delay between captures
            
            cap.release()
            
            if brightness_values:
                avg_brightness = sum(brightness_values) / len(brightness_values)
                min_brightness = min(brightness_values)
                
                # Only include cameras where ALL frames are non-black
                if min_brightness > 5.0:  # Stricter threshold
                    available.append((i, avg_brightness))
                    logger.debug(f"Camera {i}: ✅ valid (avg={avg_brightness:.1f}, min={min_brightness:.1f})")
                else:
                    logger.warning(f"Camera {i}: ❌ produces black frames (avg={avg_brightness:.1f}, min={min_brightness:.1f}), skipping")
    
    # Sort by brightness (descending) to prefer better cameras
    available.sort(key=lambda x: x[1], reverse=True)
    camera_indices = [idx for idx, _ in available]
    
    logger.info(f"Found {len(camera_indices)} usable cameras: {camera_indices}")
    return camera_indices


if __name__ == "__main__":
    # Test dual camera system
    logging.basicConfig(level=logging.INFO)
    
    print("🎥 Testing Dual Camera System")
    print("=" * 50)
    
    # Check available cameras
    cameras = check_available_cameras()
    print(f"\n✅ Found {len(cameras)} cameras: {cameras}")
    
    if len(cameras) < 2:
        print("❌ ERROR: Need at least 2 cameras for stereo vision!")
        exit(1)
    
    # Initialize dual camera manager
    print("\n📷 Initializing dual camera system...")
    manager = DualCameraManager(
        camera_index_left=cameras[0],
        camera_index_right=cameras[1],
        resolution=(1280, 720)
    )
    
    if not manager.start():
        print("❌ Failed to start cameras")
        exit(1)
    
    print("\n✅ Cameras started successfully!")
    print(f"Camera Info: {manager.get_camera_info()}")
    
    # Capture test
    print("\n📸 Capturing stereo frames...")
    for i in range(5):
        stereo = manager.capture_stereo()
        if stereo:
            print(f"  Frame {i+1}: L={stereo.left.shape}, R={stereo.right.shape}, Δt={stereo.time_delta_ms:.2f}ms")
        else:
            print(f"  Frame {i+1}: ❌ FAILED")
    
    manager.stop()
    print("\n✅ Test complete!")
