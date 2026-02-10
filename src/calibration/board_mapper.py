"""
Dartboard coordinate mapping system.

Transforms pixel coordinates from camera to dartboard segments and scoring zones
using polar coordinate mathematics and configurable calibration.

Author: Mario Neuhauser
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
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
        perspective_corners: Optional 4 corner points for perspective correction
                           Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                           Order: top-left, top-right, bottom-right, bottom-left
    """
    center_x: int
    center_y: int
    bull_eye_radius: float
    bull_radius: float
    triple_inner_radius: float
    triple_outer_radius: float
    double_inner_radius: float
    double_outer_radius: float
    perspective_corners: Optional[List[Tuple[int, int]]] = field(default=None)
    # Perspective transformation parameters
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    skew_x: float = 0.0
    skew_y: float = 0.0
    # Ring group scaling parameters
    triple_scale: float = 1.0
    double_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bull_eye_radius": self.bull_eye_radius,
            "bull_radius": self.bull_radius,
            "triple_inner_radius": self.triple_inner_radius,
            "triple_outer_radius": self.triple_outer_radius,
            "double_inner_radius": self.double_inner_radius,
            "double_outer_radius": self.double_outer_radius,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "rotation": self.rotation,
            "skew_x": self.skew_x,
            "skew_y": self.skew_y,
            "triple_scale": self.triple_scale,
            "double_scale": self.double_scale
        }
        if self.perspective_corners:
            data["perspective_corners"] = self.perspective_corners
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create from dictionary loaded from JSON."""
        # Convert perspective_corners list of lists to list of tuples if present
        if "perspective_corners" in data and data["perspective_corners"]:
            data["perspective_corners"] = [tuple(pt) for pt in data["perspective_corners"]]
        
        # Set defaults for new perspective/scale fields (backward compatibility)
        data.setdefault("scale_x", 1.0)
        data.setdefault("scale_y", 1.0)
        data.setdefault("rotation", 0.0)
        data.setdefault("skew_x", 0.0)
        data.setdefault("skew_y", 0.0)
        data.setdefault("triple_scale", 1.0)
        data.setdefault("double_scale", 1.0)
        
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
    
    def apply_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective correction to image using ring points.
        
        This transforms a tilted/angled view of the dartboard into a frontal view
        by fitting an ellipse to the marked ring points and transforming it to a circle.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Corrected image (same size as input)
            
        Raises:
            RuntimeError: If calibration not set
        """
        if self.calibration is None:
            raise RuntimeError("Calibration must be set before perspective correction")
        
        # If no ring points defined (need at least 3 for ellipse), return original image
        if not self.calibration.perspective_corners or len(self.calibration.perspective_corners) < 3:
            return image
        
        height, width = image.shape[:2]
        
        try:
            # Fit ellipse to ring points
            points_array = np.array(self.calibration.perspective_corners, dtype=np.float32)
            
            if len(points_array) < 5:
                # OpenCV fitEllipse needs at least 5 points, use minimum bounding ellipse for 3-4 points
                ellipse = cv2.fitEllipse(points_array)
            else:
                ellipse = cv2.fitEllipse(points_array)
            
            # Ellipse format: ((center_x, center_y), (width, height), angle)
            center, axes, angle = ellipse
            
            # Calculate transformation to make ellipse a perfect circle
            # We want to transform the ellipse to a circle with radius = average of axes
            target_radius = (axes[0] + axes[1]) / 4  # Use quarter of sum for reasonable size
            
            # Create source points on the ellipse perimeter (8 points evenly distributed)
            num_points = 8
            src_ellipse_points = []
            for i in range(num_points):
                t = 2 * np.pi * i / num_points
                # Point on ellipse (parametric form)
                x = center[0] + (axes[0]/2) * np.cos(t) * np.cos(np.radians(angle)) - (axes[1]/2) * np.sin(t) * np.sin(np.radians(angle))
                y = center[1] + (axes[0]/2) * np.cos(t) * np.sin(np.radians(angle)) + (axes[1]/2) * np.sin(t) * np.cos(np.radians(angle))
                src_ellipse_points.append([x, y])
            
            # Create corresponding points on perfect circle
            dst_circle_points = []
            for i in range(num_points):
                t = 2 * np.pi * i / num_points
                x = center[0] + target_radius * np.cos(t)
                y = center[1] + target_radius * np.sin(t)
                dst_circle_points.append([x, y])
            
            src_points = np.float32(src_ellipse_points)
            dst_points = np.float32(dst_circle_points)
            
            # Find homography matrix (8 points -> more robust than 4-point perspective)
            matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
            
            if matrix is None:
                logger.warning("Failed to compute homography, returning original image")
                return image
            
            # Apply transformation
            corrected = cv2.warpPerspective(image, matrix, (width, height), 
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))
            
            logger.debug(f"Perspective correction applied (ellipse {axes} at {angle}° → circle)")
            return corrected
            
        except Exception as e:
            logger.warning(f"Failed to apply perspective correction: {e}, returning original image")
            return image
    
    def set_perspective_corners(self, corners: List[Tuple[int, int]]) -> None:
        """
        Set ring points for perspective correction.
        
        Args:
            corners: List of 3+ points (x, y) on the outer ring of the dartboard.
                    These points are used to fit an ellipse and correct perspective.
                    
        Raises:
            ValueError: If fewer than 3 points provided
            RuntimeError: If calibration not set
        """
        if self.calibration is None:
            raise RuntimeError("Calibration must be set before setting ring points")
        
        if len(corners) < 3:
            raise ValueError(f"Expected at least 3 ring points, got {len(corners)}")
        
        self.calibration.perspective_corners = corners
        logger.info(f"Perspective ring points set: {len(corners)} points")
    
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
        double_outer_radius=board_radius_pixels * 1.000,
        # Perspective/scale defaults
        scale_x=1.0,
        scale_y=1.0,
        rotation=0.0,
        skew_x=0.0,
        skew_y=0.0,
        triple_scale=1.0,
        double_scale=1.0
    )
