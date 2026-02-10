"""
Perspective-aware dartboard calibration system.

Uses 4-point homography to handle extreme camera angles.
Automatically detects dartboard and allows manual refinement.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PerspectiveCalibrator:
    """
    Handles dartboard calibration with perspective correction.
    
    Uses 4 reference points on the dartboard to calculate homography matrix.
    Works with any camera angle, even extreme side views.
    """
    
    # Standard dartboard dimensions (PDC standard in mm)
    DARTBOARD_DIAMETER = 451  # mm
    DARTBOARD_RADIUS = 225.5  # mm
    
    # Ring radii in mm (from center)
    BULL_EYE_RADIUS = 6.35  # Inner bull (red)
    BULL_RADIUS = 15.9  # Outer bull (green)
    TRIPLE_INNER_RADIUS = 99  # Inner edge of triple ring
    TRIPLE_OUTER_RADIUS = 107  # Outer edge of triple ring
    DOUBLE_INNER_RADIUS = 162  # Inner edge of double ring
    DOUBLE_OUTER_RADIUS = 170  # Outer edge of double ring (board edge)
    
    # Segment angles (degrees) - 20 segments starting at top (20)
    SEGMENT_ANGLES = [90, 78, 60, 42, 24, 6, 348, 330, 312, 294,
                      276, 258, 240, 222, 204, 186, 168, 150, 132, 114]
    
    SEGMENT_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                       3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    # Zone mapping (compatible with BoardMapper)
    ZONE_NAMES = {
        'bull_eye': 'bull_eye',
        'bull': 'bull',
        'triple': 'triple',
        'double': 'double',
        'single': 'single',
        'miss': 'miss'
    }
    
    def __init__(self):
        """Initialize perspective calibrator."""
        self.homography_matrix: Optional[np.ndarray] = None
        self.reference_points: List[Tuple[float, float]] = []  # Image coordinates
        self.board_center_px: Optional[Tuple[int, int]] = None
        self.board_radius_px: Optional[float] = None
        self.rotation_offset: float = 0.0  # Degrees to rotate entire dartboard
        
    def auto_detect_board(self, image: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, float]]:
        """
        Automatically detect dartboard in image.
        
        Args:
            image: Input BGR image
            debug: Show debug visualization
            
        Returns:
            (center_x, center_y, radius) or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-strategy detection
        strategies = [
            self._detect_by_hough_circles,
            self._detect_by_contours,
            self._detect_by_edge_density
        ]
        
        for strategy in strategies:
            result = strategy(gray, image)
            if result is not None:
                cx, cy, radius = result
                
                # Validate detection
                if 50 < radius < min(image.shape[:2]) * 0.6:
                    logger.info(f"Board detected at ({cx}, {cy}) with radius {radius:.1f}")
                    self.board_center_px = (int(cx), int(cy))
                    self.board_radius_px = radius
                    return (int(cx), int(cy), radius)
        
        logger.warning("Could not auto-detect dartboard")
        return None
    
    def _detect_by_hough_circles(self, gray: np.ndarray, color: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect board using Hough circle transform."""
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Blur
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        # Detect circles with relaxed parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=int(min(gray.shape) * 0.15),
            maxRadius=int(min(gray.shape) * 0.5)
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Take largest circle
            largest = max(circles[0], key=lambda c: c[2])
            return (largest[0], largest[1], largest[2])
        
        return None
    
    def _detect_by_contours(self, gray: np.ndarray, color: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect board by finding largest circular contour."""
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if circular enough
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.3:  # Very relaxed for perspective views
            # Fit ellipse
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (cx, cy), (w, h), angle = ellipse
                radius = (w + h) / 4.0
                return (cx, cy, radius)
        
        return None
    
    def _detect_by_edge_density(self, gray: np.ndarray, color: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect board by analyzing edge density (dartboards have many radial edges)."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find center of mass of edges
        moments = cv2.moments(edges)
        
        if moments['m00'] == 0:
            return None
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Estimate radius by checking edge distribution
        y_coords, x_coords = np.where(edges > 0)
        
        if len(x_coords) == 0:
            return None
        
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Take 90th percentile as radius estimate
        radius = np.percentile(distances, 90)
        
        return (cx, cy, radius)
    
    def set_reference_points(self, points: List[Tuple[float, float]], 
                           known_positions: List[Tuple[float, float]]):
        """
        Set 4 reference points for homography calculation.
        
        Args:
            points: 4 points in image coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            known_positions: 4 corresponding points in dartboard coordinates (mm from center)
        """
        if len(points) != 4 or len(known_positions) != 4:
            raise ValueError("Need exactly 4 reference points")
        
        self.reference_points = points
        
        # Calculate homography matrix
        src_pts = np.float32(points)
        dst_pts = np.float32(known_positions)
        
        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        
        logger.info(f"Homography calculated from {len(points)} reference points")
    
    def image_to_dartboard(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform image coordinates to dartboard coordinates (mm from center).
        
        Args:
            x, y: Pixel coordinates in image
            
        Returns:
            (x_mm, y_mm) in dartboard coordinate system, or None if no calibration
        """
        if self.homography_matrix is None:
            return None
        
        point = np.array([[x, y]], dtype=np.float32)
        point = point.reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return (transformed[0][0][0], transformed[0][0][1])
    
    def dartboard_to_image(self, x_mm: float, y_mm: float) -> Optional[Tuple[int, int]]:
        """
        Transform dartboard coordinates (mm) to image pixel coordinates.
        
        Args:
            x_mm, y_mm: Coordinates in mm from dartboard center
            
        Returns:
            (x_px, y_px) or None if no calibration
        """
        if self.homography_matrix is None:
            return None
        
        # Use inverse homography
        inv_homography = np.linalg.inv(self.homography_matrix)
        
        point = np.array([[x_mm, y_mm]], dtype=np.float32)
        point = point.reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(point, inv_homography)
        
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def generate_reference_points_for_4_segments(self) -> List[Tuple[float, float]]:
        """
        Generate ideal reference points for 4 key segments (20, 3, 6, 11).
        These are at 90°, 0°, 270°, 180° on the double ring outer edge.
        
        Returns:
            List of 4 (x_mm, y_mm) coordinates on dartboard
        """
        radius_mm = self.DOUBLE_OUTER_RADIUS
        
        # Top (20), Right (6), Bottom (6), Left (11)
        # In standard dartboard: 20 is at top (90°)
        points = [
            (0, radius_mm),  # Top (20) - 90°
            (radius_mm, 0),  # Right (6) - 0°
            (0, -radius_mm),  # Bottom (3) - 270°
            (-radius_mm, 0),  # Left (11) - 180°
        ]
        
        return points
    
    def draw_overlay(self, image: np.ndarray, show_segments: bool = True,
                    show_rings: bool = True, show_reference_points: bool = True,
                    show_numbers: bool = True) -> np.ndarray:
        """
        Draw calibrated dartboard overlay on image.
        
        Args:
            image: Input image
            show_segments: Draw segment lines
            show_rings: Draw ring circles/ellipses
            show_reference_points: Draw the 4 reference points
            show_numbers: Draw segment numbers
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        
        if self.homography_matrix is None:
            return overlay
        
        # Draw rings
        if show_rings:
            rings = [
                (self.BULL_EYE_RADIUS, (0, 255, 0), 2),
                (self.BULL_RADIUS, (0, 255, 255), 2),
                (self.TRIPLE_INNER_RADIUS, (255, 255, 0), 2),
                (self.TRIPLE_OUTER_RADIUS, (255, 255, 0), 2),
                (self.DOUBLE_INNER_RADIUS, (255, 0, 0), 2),
                (self.DOUBLE_OUTER_RADIUS, (255, 0, 0), 2),
            ]
            
            for radius_mm, color, thickness in rings:
                self._draw_ring(overlay, radius_mm, color, thickness)
        
        # Draw segments
        if show_segments:
            for angle_deg in self.SEGMENT_ANGLES:
                adjusted_angle = angle_deg + self.rotation_offset
                self._draw_segment_line(overlay, adjusted_angle, (128, 128, 128), 1)
        
        # Draw segment numbers
        if show_numbers:
            for i, (angle_deg, number) in enumerate(zip(self.SEGMENT_ANGLES, self.SEGMENT_NUMBERS)):
                adjusted_angle = angle_deg + self.rotation_offset
                self._draw_segment_number(overlay, adjusted_angle, number)
        
        # Draw center
        center_px = self.dartboard_to_image(0, 0)
        if center_px:
            cv2.drawMarker(overlay, center_px, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Draw reference points
        if show_reference_points and len(self.reference_points) == 4:
            for i, (x, y) in enumerate(self.reference_points):
                cv2.circle(overlay, (int(x), int(y)), 10, (255, 0, 255), -1)
                cv2.circle(overlay, (int(x), int(y)), 12, (255, 255, 255), 2)
                cv2.putText(overlay, str(i+1), (int(x)+15, int(y)+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay
    
    def _draw_ring(self, image: np.ndarray, radius_mm: float, color: tuple, thickness: int):
        """Draw a ring at given radius using homography."""
        # Sample points around the ring
        num_points = 72  # 5° intervals
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        points = []
        for angle in angles:
            x_mm = radius_mm * np.cos(angle)
            y_mm = radius_mm * np.sin(angle)
            
            px_coords = self.dartboard_to_image(x_mm, y_mm)
            if px_coords:
                points.append(px_coords)
        
        # Draw polygon connecting points
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)
    
    def _draw_segment_line(self, image: np.ndarray, angle_deg: float, color: tuple, thickness: int):
        """Draw a segment line from center to outer edge."""
        angle_rad = np.radians(angle_deg)
        
        # Line from inner bull to outer edge
        start_mm = (self.BULL_RADIUS * np.cos(angle_rad), 
                   self.BULL_RADIUS * np.sin(angle_rad))
        end_mm = (self.DOUBLE_OUTER_RADIUS * np.cos(angle_rad),
                 self.DOUBLE_OUTER_RADIUS * np.sin(angle_rad))
        
        start_px = self.dartboard_to_image(*start_mm)
        end_px = self.dartboard_to_image(*end_mm)
        
        if start_px and end_px:
            cv2.line(image, start_px, end_px, color, thickness, cv2.LINE_AA)
    
    def _draw_segment_number(self, image: np.ndarray, angle_deg: float, number: int):
        """Draw segment number at outer edge."""
        angle_rad = np.radians(angle_deg)
        
        # Position at outer edge + some offset
        radius_mm = self.DOUBLE_OUTER_RADIUS + 15  # 15mm outside the board
        x_mm = radius_mm * np.cos(angle_rad)
        y_mm = radius_mm * np.sin(angle_rad)
        
        pos_px = self.dartboard_to_image(x_mm, y_mm)
        
        if pos_px:
            # Draw number
            text = str(number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for centering
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center text at position
            text_x = pos_px[0] - text_w // 2
            text_y = pos_px[1] + text_h // 2
            
            # Draw background
            cv2.rectangle(image,
                        (text_x - 5, text_y - text_h - 5),
                        (text_x + text_w + 5, text_y + 5),
                        (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(image, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    def map_coordinate_to_field(self, x_px: int, y_px: int):
        """
        Map pixel coordinate to dartboard field using perspective calibration.
        
        Args:
            x_px, y_px: Pixel coordinates in image
            
        Returns:
            DartboardField object with zone, segment, score, or None if out of bounds
        """
        if self.homography_matrix is None:
            return None
        
        # Transform to dartboard coordinates (mm)
        coords_mm = self.image_to_dartboard(x_px, y_px)
        if coords_mm is None:
            return None
        
        x_mm, y_mm = coords_mm
        
        # Calculate distance from center
        distance = np.sqrt(x_mm**2 + y_mm**2)
        
        # Calculate angle (0° = right, 90° = top, counter-clockwise)
        angle_rad = np.arctan2(y_mm, x_mm)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        # Determine zone based on distance
        if distance <= self.BULL_EYE_RADIUS:
            zone = 'bull_eye'
            segment = 25  # Bull's eye has no segment number
            score = 50
        elif distance <= self.BULL_RADIUS:
            zone = 'bull'
            segment = 25
            score = 25
        elif distance > self.DOUBLE_OUTER_RADIUS:
            zone = 'miss'
            segment = 0
            score = 0
        else:
            # Determine segment based on angle
            # Dartboard segments go clockwise from 20 at top (90°)
            # arctan2 gives counter-clockwise angles, so we need to convert
            # Start at 90° (top) and go clockwise (subtract angle)
            # Compensate for rotation offset (if board is rotated in image)
            # Add 9° offset so boundaries are at multiples of 18°
            adjusted_angle = (90 - angle_deg + self.rotation_offset + 9) % 360
            segment_index = int(adjusted_angle / 18)
            segment = self.SEGMENT_NUMBERS[segment_index]
            
            # Determine zone (triple, double, or single)
            if self.TRIPLE_INNER_RADIUS <= distance <= self.TRIPLE_OUTER_RADIUS:
                zone = 'triple'
                score = segment * 3
            elif self.DOUBLE_INNER_RADIUS <= distance <= self.DOUBLE_OUTER_RADIUS:
                zone = 'double'
                score = segment * 2
            else:
                zone = 'single'
                score = segment
        
        # Create DartboardField-like object
        from types import SimpleNamespace
        return SimpleNamespace(
            zone=zone,
            segment=segment,
            score=score,
            x=x_px,
            y=y_px
        )
