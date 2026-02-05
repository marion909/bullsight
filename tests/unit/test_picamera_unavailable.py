"""
Test for unreachable Picamera2 None check.

Author: Mario Neuhauser  
"""

import pytest
from unittest.mock import patch

from src.vision.camera_manager import CameraManager


def test_start_without_picamera2():
    """Test starting camera when picamera2 module is not available."""
    # Temporarily set Picamera2 to None (simulating non-Raspberry Pi system)
    with patch('src.vision.camera_manager.Picamera2', None):
        manager = CameraManager()
        
        with pytest.raises(RuntimeError, match="picamera2 not available"):
            manager.start()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
