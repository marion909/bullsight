"""
Dartboard coordinate mapping system.

Transforms pixel coordinates from camera to dartboard segments and scoring zones
using polar coordinate mathematics and configurable calibration.

Author: Mario Neuhauser
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging


logger = logging.getLogger(__name__)


# Standard dartboard segment order (clockwise from top)
DARTBOARD_SEGMENTS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]


@dataclass
class DartboardField:
    """
    Represents a specific field on the dartboard.
    
    Attributes:
        segment: Segment number (1-20, or 25 for bull)
        zone: Scoring zone ('bull_eye', 'bull', 'single', 'double', 'triple', 'miss')
        score: Points value for this field
        multiplier: Score multiplier (1, 2, or 3)
    """
    segment: int
    zone: str
    score: int
    multiplier: int
    
    def __repr__(self) -> str:
        if self.zone == "bull_eye":
            return "Bull's Eye (50)"
        elif self.zone == "bull":
            return "Bull (25)"
        elif self.zone == "miss":
            return "Miss (0)"
        else:
            return f"{self.zone.title()} {self.segment} ({self.score})"


@dataclass
class CalibrationData:
    """
    Calibration data for dartboard mapping.
    
    Attributes:
        center_x: X coordinate of dartboard center
        center_y: Y coordinate of dartboard center
        bull_eye_radius: Radius of bull's eye in pixels
        bull_radius: Radius of outer bull ring
        triple_inner_radius: Inner radius of triple ring
        triple_outer_radius: Outer radius of triple ring
        double_inner_radius: Inner radius of double ring
        double_outer_radius: Outer radius of double ring
    """
    center_x: int
    center_y: int
    bull_eye_radius: float
    bull_radius: float
    triple_inner_radius: float
    triple_outer_radius: float
    double_inner_radius: float
    double_outer_radius: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bull_eye_radius": self.bull_eye_radius,
            "bull_radius": self.bull_radius,
            "triple_inner_radius": self.triple_inner_radius,
            "triple_outer_radius": self.triple_outer_radius,
            "double_inner_radius": self.double_inner_radius,
            "double_outer_radius": self.double_outer_radius
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class BoardMapper:
    """
    Maps pixel coordinates to dartboard fields.
    
    Performs geometric transformation from camera pixel space to
    dartboard segment and zone identification using polar coordinates.
    
    Attributes:
        calibration: Board calibration parameters
        segments: Dartboard segment sequence
    """
    
    def __init__(self, calibration: Optional[CalibrationData] = None):
        """
        Initialize board mapper.
        
        Args:
            calibration: Optional pre-loaded calibration data
        """
        self.calibration = calibration
        self.segments = DARTBOARD_SEGMENTS
        logger.info("BoardMapper initialized")
    
    def set_calibration(self, calibration: CalibrationData) -> None:
        """
        Set calibration data for mapping.
        
        Args:
            calibration: Calibration parameters
        """
        self.calibration = calibration
        logger.info(f"Calibration set: center=({calibration.center_x}, {calibration.center_y})")
    
    def load_calibration(self, filepath: Path) -> None:
        """
        Load calibration from JSON file.
        
        Args:
            filepath: Path to calibration JSON file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.calibration = CalibrationData.from_dict(data)
            logger.info(f"Calibration loaded from {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid calibration JSON: {e}")
    
    def save_calibration(self, filepath: Path) -> None:
        """
        Save calibration to JSON file.
        
        Args:
            filepath: Path where to save calibration
            
        Raises:
            RuntimeError: If no calibration data is set
        """
        if self.calibration is None:
            raise RuntimeError("No calibration data to save")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def pixel_to_polar(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to polar coordinates.
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            Tuple of (radius in pixels, angle in degrees 0-360)
            
        Raises:
            RuntimeError: If calibration not set
        """
        if self.calibration is None:
            raise RuntimeError("Calibration must be set before mapping")
        
        # Calculate relative position from center
        dx = x - self.calibration.center_x
        dy = y - self.calibration.center_y
        
        # Calculate radius
        radius = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle (atan2 returns -π to π)
        angle_rad = np.arctan2(dy, dx)
        
        # Convert to degrees and normalize to 0-360
        # dx>0, dy=0 → 0° (right, segment 6)
        # dx=0, dy<0 → 270° (up, segment 20)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        return radius, angle_deg
    
    def angle_to_segment(self, angle: float) -> int:
        """
        Map angle to dartboard segment number.
        
        Args:
            angle: Angle in degrees (0-360), where 0° is right (3 o'clock)
            
        Returns:
            Segment number (1-20)
        """
        # Standard dartboard has segment 20 at top (270° or 12 o'clock)
        # Segments go clockwise from there
        # Rotate angle so that 270° (top) maps to segment 20
        # Since dartboard starts with 20 at top, we need to rotate our angle
        # so that 270° becomes index 0
        
        adjusted_angle = (angle - 270 + 9) % 360  # Rotate and offset by 9° to center segments
        segment_index = int(adjusted_angle / 18)
        
        # Clamp to valid range
        segment_index = max(0, min(19, segment_index))
        
        return self.segments[segment_index]
    
    def radius_to_zone(self, radius: float) -> str:
        """
        Map radius to scoring zone.
        
        Args:
            radius: Distance from center in pixels
            
        Returns:
            Zone name ('bull_eye', 'bull', 'single', 'double', 'triple', 'miss')
            
        Raises:
            RuntimeError: If calibration not set
        """
        if self.calibration is None:
            raise RuntimeError("Calibration must be set")
        
        cal = self.calibration
        
        if radius <= cal.bull_eye_radius:
            return "bull_eye"
        elif radius <= cal.bull_radius:
            return "bull"
        elif radius <= cal.triple_inner_radius:
            return "single"
        elif radius <= cal.triple_outer_radius:
            return "triple"
        elif radius <= cal.double_inner_radius:
            return "single"
        elif radius <= cal.double_outer_radius:
            return "double"
        else:
            return "miss"
    
    def calculate_score(self, segment: int, zone: str) -> Tuple[int, int]:
        """
        Calculate score and multiplier for field.
        
        Args:
            segment: Segment number (1-20 or 25 for bull)
            zone: Zone name
            
        Returns:
            Tuple of (score, multiplier)
        """
        if zone == "bull_eye":
            return 50, 1
        elif zone == "bull":
            return 25, 1
        elif zone == "miss":
            return 0, 0
        elif zone == "double":
            return segment * 2, 2
        elif zone == "triple":
            return segment * 3, 3
        else:  # single
            return segment, 1
    
    def map_coordinate(self, x: int, y: int) -> DartboardField:
        """
        Map pixel coordinate to dartboard field.
        
        Complete transformation pipeline from pixels to game field.
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            DartboardField with segment, zone, and score
            
        Raises:
            RuntimeError: If calibration not set
        """
        # Convert to polar
        radius, angle = self.pixel_to_polar(x, y)
        
        # Map to zone
        zone = self.radius_to_zone(radius)
        
        # Map to segment (unless bull/miss)
        if zone in ["bull_eye", "bull"]:
            segment = 25
        elif zone == "miss":
            segment = 0
        else:
            segment = self.angle_to_segment(angle)
        
        # Calculate score
        score, multiplier = self.calculate_score(segment, zone)
        
        field = DartboardField(
            segment=segment,
            zone=zone,
            score=score,
            multiplier=multiplier
        )
        
        logger.debug(f"Mapped ({x}, {y}) → {field}")
        return field
    
    def validate_calibration(self) -> Dict[str, bool]:
        """
        Validate calibration data consistency.
        
        Returns:
            Dictionary of validation checks and results
        """
        if self.calibration is None:
            return {"calibration_set": False}
        
        cal = self.calibration
        
        checks = {
            "calibration_set": True,
            "bull_eye_smaller_than_bull": cal.bull_eye_radius < cal.bull_radius,
            "bull_smaller_than_triple_inner": cal.bull_radius < cal.triple_inner_radius,
            "triple_inner_smaller_than_outer": cal.triple_inner_radius < cal.triple_outer_radius,
            "triple_outer_smaller_than_double_inner": cal.triple_outer_radius < cal.double_inner_radius,
            "double_inner_smaller_than_outer": cal.double_inner_radius < cal.double_outer_radius,
            "center_positive": cal.center_x > 0 and cal.center_y > 0,
            "all_radii_positive": all([
                cal.bull_eye_radius > 0,
                cal.bull_radius > 0,
                cal.triple_inner_radius > 0,
                cal.triple_outer_radius > 0,
                cal.double_inner_radius > 0,
                cal.double_outer_radius > 0
            ])
        }
        
        all_valid = all(checks.values())
        checks["all_checks_passed"] = all_valid
        
        return checks
    
    def get_segment_bounds(self, segment: int) -> Tuple[float, float]:
        """
        Get angle boundaries for a segment.
        
        Args:
            segment: Segment number (1-20)
            
        Returns:
            Tuple of (start_angle, end_angle) in degrees (0° = right, 270° = top)
            
        Raises:
            ValueError: If segment not found
        """
        try:
            segment_index = self.segments.index(segment)
        except ValueError:
            raise ValueError(f"Invalid segment number: {segment}")
        
        # Segment 20 (index 0) starts at 261° (270° - 9°)
        # Each segment is 18 degrees wide
        start_angle = float((270 + segment_index * 18 - 9) % 360)
        end_angle = float((start_angle + 18) % 360)
        
        return start_angle, end_angle


def create_default_calibration(
    image_width: int,
    image_height: int,
    board_radius_pixels: float
) -> CalibrationData:
    """
    Create default calibration based on image dimensions.
    
    Uses standard dartboard proportions relative to board radius.
    
    Args:
        image_width: Camera image width in pixels
        image_height: Camera image height in pixels
        board_radius_pixels: Estimated dartboard radius in image
        
    Returns:
        CalibrationData with estimated values
    """
    # Standard dartboard measurements (relative to outer double radius = 1.0):
    # Bull's eye: 0.019
    # Bull outer: 0.049
    # Triple inner: 0.560
    # Triple outer: 0.622
    # Double inner: 0.933
    # Double outer: 1.000
    
    center_x = image_width // 2
    center_y = image_height // 2
    
    return CalibrationData(
        center_x=center_x,
        center_y=center_y,
        bull_eye_radius=board_radius_pixels * 0.019,
        bull_radius=board_radius_pixels * 0.049,
        triple_inner_radius=board_radius_pixels * 0.560,
        triple_outer_radius=board_radius_pixels * 0.622,
        double_inner_radius=board_radius_pixels * 0.933,
        double_outer_radius=board_radius_pixels * 1.000
    )
