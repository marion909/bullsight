"""
Test for unreachable code in dart_detector.

Author: Mario Neuhauser
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.vision.dart_detector import DartDetector


def test_load_reference_invalid_image_file(tmp_path):
    """Test loading invalid image file that exists but can't be read."""
    detector = DartDetector()
    
    # Create a non-image file
    invalid_file = tmp_path / "invalid.jpg"
    invalid_file.write_text("not an image")
    
    # Mock cv2.imread to return None (indicating read failure)
    with patch('cv2.imread', return_value=None):
        with pytest.raises(ValueError, match="Could not read image file"):
            detector.load_reference_from_file(invalid_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
