"""
Dartboard Pattern Detection for Stereo Calibration.

Uses the standardized dartboard geometry as calibration pattern instead of
traditional checkerboard. Provides convenient calibration using the permanently
mounted dartboard.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DartboardDimensions:
    """
    PDC/WINMAU standard dartboard dimensions in millimeters.
    
    All measurements from center point. Dartboard assumed to be in
    XY plane with center at origin and Z=0.
    """
    # Center markers
    BULL_EYE_RADIUS: float = 6.35  # 12.7mm diameter
    BULL_OUTER_RADIUS: float = 15.9  # 31.8mm diameter
    
    # Scoring rings
    TRIPLE_INNER_RADIUS: float = 99.0
    TRIPLE_OUTER_RADIUS: float = 107.0
    DOUBLE_INNER_RADIUS: float = 162.0
    DOUBLE_OUTER_RADIUS: float = 170.0
    
    # Board dimensions
    BOARD_RADIUS: float = 225.5  # 451mm diameter
    
    # Segments
    NUM_SEGMENTS: int = 20
    SEGMENT_ANGLE: float = 18.0  # 360/20
    
    # Segment order (clockwise from top)
    SEGMENTS: List[int] = None
    
    def __post_init__(self):
        """Initialize segment order."""
        if self.SEGMENTS is None:
            object.__setattr__(self, 'SEGMENTS', [
                20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                3, 19, 7, 16, 8, 11, 14, 9, 12, 5
            ])


class DartboardPatternDetector:
    """
    Detect dartboard features for stereo calibration.
    
    Uses dartboard's standardized geometry (rings, segments) as calibration
    reference points. More convenient than checkerboard since board is
    permanently mounted.
    
    Detection Strategy:
    1. Detect concentric circles (Bull, Triple, Double rings)
    2. Find radial segment boundaries (20 lines at 18° intervals)
    3. Compute ring-segment intersections (80 calibration points)
    4. Match to known 3D world coordinates
    """
    
    def __init__(self, dimensions: Optional[DartboardDimensions] = None):
        """
        Initialize dartboard pattern detector.
        
        Args:
            dimensions: Dartboard dimensions (default: PDC standard)
        """
        self.dims = dimensions if dimensions else DartboardDimensions()
        
        # Detection parameters (tunable)
        self.min_circle_distance = 15  # Minimum distance between detected circles
        self.canny_threshold1 = 30
        self.canny_threshold2 = 100
        self.hough_circle_dp = 1.0
        self.hough_circle_param1 = 50  # Lowered for better edge detection
        self.hough_circle_param2 = 20  # Lowered to accept more circles
        self.hough_line_threshold = 60
        self.use_clahe = True  # Enhance contrast for better detection
        self.simplified_fallback = True  # Use simplified detection if circles fail
        
    def get_3d_reference_points(self) -> np.ndarray:
        """
        Generate 3D world coordinates for dartboard calibration points.
        
        Creates reference points at ring-segment intersections:
        - Bull's Eye center (1 point)
        - Bull outer ring × 20 segments (20 points)
        - Triple ring (inner+outer) × 20 segments (40 points)
        - Double ring (inner+outer) × 20 segments (40 points)
        Total: 101 calibration points maximum
        
        Returns:
            Array of shape (N, 3) with [X, Y, Z] coordinates in mm
            Z=0 (dartboard plane), X/Y relative to center
        """
        points = []
        
        # 1. Center point (Bull's Eye)
        points.append([0.0, 0.0, 0.0])
        
        # 2. Generate points at the 6 standard dartboard rings
        # Start at top (20 segment) = -90° in standard coordinates
        start_angle_deg = -90.0
        
        # Standard PDC dartboard radii (in mm)
        standard_radii = [
            self.dims.BULL_OUTER_RADIUS,      # 15.9 mm
            self.dims.TRIPLE_INNER_RADIUS,    # 99.0 mm
            self.dims.TRIPLE_OUTER_RADIUS,    # 107.0 mm
            self.dims.DOUBLE_INNER_RADIUS,    # 162.0 mm
            self.dims.DOUBLE_OUTER_RADIUS,    # 170.0 mm
        ]
        
        for i in range(self.dims.NUM_SEGMENTS):
            # Angle for this segment (center of wire)
            angle_deg = start_angle_deg + i * self.dims.SEGMENT_ANGLE
            angle_rad = np.deg2rad(angle_deg)
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Generate points at each standard radius
            for radius in standard_radii:
                x = radius * cos_a
                y = radius * sin_a
                points.append([x, y, 0.0])
        
        return np.array(points, dtype=np.float32)
    
    def detect_dartboard_features(
        self, 
        image: np.ndarray,
        visualize: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect dartboard calibration features in image.
        
        Args:
            image: Input image (BGR or grayscale)
            visualize: If True, return annotated visualization
            
        Returns:
            Tuple of (image_points, visualization) or None if detection failed
            - image_points: Array of shape (N, 2) with [x, y] pixel coordinates
            - visualization: Annotated image (if visualize=True, else None)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Detect dartboard center and rings
        circles_result = self._detect_concentric_circles(gray)
        
        # Try simplified fallback if circle detection fails OR too few rings found
        if circles_result is None or (circles_result is not None and len(circles_result[1]) < 3):
            if circles_result is not None:
                logger.info(f"Only {len(circles_result[1])} rings found, trying simplified fallback...")
            else:
                logger.info("Circle detection failed, trying simplified fallback...")
            
            circles_result = self._detect_board_simple(gray)
        
        if circles_result is None:
            logger.warning("Could not detect dartboard (tried all methods)")
            return None
        
        center, radii = circles_result
        logger.info(f"Detected dartboard center at {center}, radii: {radii}")
        
        # Step 2: Use standard 20-segment layout (PDC standard)
        # Segment boundaries are standardized at 18° intervals
        # Starting at -90° (top, 20 segment) in image coordinates
        logger.info("Using standard PDC 20-segment layout")
        segment_angles = [i * 18.0 - 90 for i in range(20)]  # -90°, -72°, -54°, ..., 252°
        
        logger.info(f"Using {len(segment_angles)} segment boundaries")
        
        # Step 3: Compute intersection points
        image_points = self._compute_intersection_points(center, radii, segment_angles)
        
        # Step 4: Create visualization if requested
        vis = None
        if visualize:
            vis = self._create_visualization(image, center, radii, segment_angles, image_points)
        
        return image_points, vis
    
    def _detect_concentric_circles(
        self, 
        gray: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], List[float]]]:
        """
        Detect concentric dartboard rings (Bull, Triple, Double).
        
        Returns:
            Tuple of (center, radii) or None
            - center: (x, y) pixel coordinates
            - radii: List of detected ring radii [bull, triple_inner, triple_outer, double_inner, double_outer]
        """
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Additional preprocessing: morphological closing to connect broken circles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # Try multiple parameter sets for robustness
        circles = None
        param_sets = [
            # (param1, param2, minRadius)
            (40, 15, 10),  # Very relaxed (for difficult lighting)
            (50, 20, 15),  # Relaxed (default)
            (45, 18, 12),  # Middle ground
            (60, 25, 15),  # Moderate
        ]
        
        for param1, param2, minRadius in param_sets:
            circles = cv2.HoughCircles(
                filtered,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_circle_dp,
                minDist=self.min_circle_distance,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=min(gray.shape) // 2
            )
            
            if circles is not None and len(circles[0]) >= 3:
                logger.debug(f"Found {len(circles[0])} circles with params: param1={param1}, param2={param2}")
                break
        
        if circles is None or len(circles[0]) < 3:
            logger.warning(f"Circle detection failed - found {len(circles[0]) if circles is not None else 0} circles")
            return None
        
        circles = circles[0]
        logger.debug(f"Initial circle candidates: {len(circles)}")
        
        # Step 1: Find the main dartboard circle (largest reasonable circle)
        # Dartboard should be one of the largest circles in the image
        # Sort by radius descending
        circles = circles[np.argsort(circles[:, 2])[::-1]]
        
        # Filter by reasonable dartboard size (10-40% of image dimension)
        h, w = gray.shape
        min_board_radius = min(h, w) * 0.10
        max_board_radius = min(h, w) * 0.40
        
        board_candidates = []
        for circle in circles:
            cx, cy, r = circle
            if min_board_radius < r < max_board_radius:
                # Check if circle is reasonably centered in image
                dist_from_img_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                if dist_from_img_center < min(w, h) * 0.3:  # Within 30% of center
                    board_candidates.append(circle)
        
        if not board_candidates:
            logger.warning("No board-sized circles found")
            return None
        
        # Use largest reasonable circle as board reference
        board_circle = board_candidates[0]
        board_cx, board_cy, board_radius = board_circle
        center = (int(board_cx), int(board_cy))
        
        logger.debug(f"Board center: {center}, radius: {board_radius:.1f}")
        
        # Step 2: Find concentric rings relative to board center
        # Only look at circles with center close to board center
        radii = []
        tolerance = board_radius * 0.15  # Increased to 15% tolerance
        
        for circle in circles:
            cx, cy, r = circle
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            
            # Must be concentric (close to same center)
            if dist < tolerance:
                # Must be smaller than board radius (inner rings)
                # Accept smaller circles too (down to 5% of board)
                if board_radius * 0.05 < r < board_radius * 0.95:
                    radii.append(float(r))
        
        if len(radii) < 2:
            logger.warning(f"Not enough concentric circles found: {len(radii)}")
            return None
        
        # Sort and keep unique radii (remove duplicates within 3% of board radius)
        radii = sorted(radii)
        unique_radii = [radii[0]]
        min_diff = max(board_radius * 0.03, 8)  # At least 8px or 3% of board
        
        for r in radii[1:]:
            if r - unique_radii[-1] > min_diff:
                unique_radii.append(r)
        
        # Try to get standard dartboard rings (3-6 rings expected)
        # Standard: Bull, Triple (inner+outer), Double (inner+outer) = 5-6 rings
        if len(unique_radii) < 3:
            logger.warning(f"Only {len(unique_radii)} rings found, expected 3-6 for dartboard")
        
        # Dartboard has maximum 6 concentric circles
        if len(unique_radii) > 6:
            logger.debug(f"Found {len(unique_radii)} rings, keeping best 6")
            # Keep 6 most evenly distributed rings
            indices = np.linspace(0, len(unique_radii)-1, 6, dtype=int)
            unique_radii = [unique_radii[i] for i in indices]
        
        logger.info(f"Detected {len(unique_radii)} unique rings: {[f'{r:.1f}' for r in unique_radii]}")
        
        return center, unique_radii
    
    def _detect_board_simple(
        self, 
        gray: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], List[float]]]:
        """
        Simplified fallback detection using contours and geometry.
        
        When Hough circles fail (e.g., angled camera views making circles appear as ellipses),
        find the board outline and estimate rings based on standard dartboard proportions.
        Better for stereo setups with strong perspective angles.
        
        Returns:
            Tuple of (center, radii) or None
        """
        logger.info("Using simplified contour-based detection (handles perspective better)...")
        
        # Enhance contrast
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Multiple thresholding strategies
        thresh_methods = []
        
        # Method 1: Otsu's method (adaptive)
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_methods.append(("Otsu", thresh1))
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 21, 5)
        thresh_methods.append(("Adaptive", thresh2))
        
        # Method 3: Fixed threshold (for very bright/dark scenes)
        _, thresh3 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        thresh_methods.append(("Fixed", thresh3))
        
        best_result = None
        best_score = 0
        
        for method_name, thresh in thresh_methods:
            # Clean up threshold
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Find board contour
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5000:  # Too small for dartboard
                    continue
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                # Circularity score (0-1, 1 = perfect circle)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Get bounding circle to check centrality
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                h, w = gray.shape
                dist_from_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                centrality = 1.0 - (dist_from_center / (min(w, h) / 2))
                centrality = max(0, centrality)
                
                # Combined score
                score = circularity * 0.7 + centrality * 0.3
                
                if score > best_score and circularity > 0.4:  # Relaxed for perspective
                    best_score = score
                    best_result = ((int(cx), int(cy)), radius, circularity, method_name)
        
        if best_result is None:
            logger.warning("No suitable dartboard contour found")
            return None
        
        center, radius, circularity, method = best_result
        logger.info(f"Board found via {method}: center={center}, radius={radius:.1f}, circularity={circularity:.2f}")
        
        # Estimate ring radii based on PDC standard proportions
        # For perspective views, board appears smaller → use outer ring as reference
        # Standard: Double outer = 170mm, Board visible radius ≈ 170mm  
        scale = radius / self.dims.DOUBLE_OUTER_RADIUS
        
        radii = [
            self.dims.BULL_OUTER_RADIUS * scale,
            self.dims.TRIPLE_INNER_RADIUS * scale,
            self.dims.TRIPLE_OUTER_RADIUS * scale,
            self.dims.DOUBLE_INNER_RADIUS * scale,
            self.dims.DOUBLE_OUTER_RADIUS * scale,
        ]
        
        logger.info(f"Estimated 5 rings: {[f'{r:.1f}px' for r in radii]}")
        
        return center, radii
    
    def _detect_segment_boundaries(
        self, 
        gray: np.ndarray, 
        center: Tuple[int, int]
    ) -> Optional[List[float]]:
        """
        Detect radial segment boundaries (wires between segments).
        
        Args:
            gray: Grayscale image
            center: Dartboard center (x, y)
            
        Returns:
            List of angles in degrees (0-360) or None
        """
        # Edge detection
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=self.hough_line_threshold)
        
        if lines is None or len(lines) < 10:
            return None
        
        # Convert lines to angles relative to center
        angles = []
        for line in lines:
            rho, theta = line[0]
            
            # Convert to angle relative to center (0° = right, 90° = down)
            # Normalize to 0-360° range
            angle_deg = np.rad2deg(theta) % 360
            
            # Check if line passes through center region
            # Distance from center to line
            cx, cy = center
            dist = abs(rho - (cx * np.cos(theta) + cy * np.sin(theta)))
            
            if dist < 30:  # Line should pass near center
                angles.append(angle_deg)
        
        if len(angles) < 15:
            return None
        
        # Cluster angles to find the 20 segment boundaries
        # Expected: 20 angles at 18° intervals
        angles = sorted(angles)
        
        # Remove duplicate angles (within 3° tolerance)
        unique_angles = [angles[0]]
        for angle in angles[1:]:
            if angle - unique_angles[-1] > 3:
                unique_angles.append(angle)
        
        # Limit to 20 segments (standard dartboard)
        if len(unique_angles) > 25:
            logger.debug(f"Too many angles ({len(unique_angles)}), filtering to best 20")
            # Keep every Nth angle to get ~20
            step = len(unique_angles) // 20
            unique_angles = unique_angles[::step][:20]
        
        # Should have ~20 segment boundaries
        return unique_angles if len(unique_angles) >= 15 else None
    
    def _compute_intersection_points(
        self,
        center: Tuple[int, int],
        radii: List[float],
        angles: List[float]
    ) -> np.ndarray:
        """
        Compute pixel coordinates of ring-segment intersections.
        
        Args:
            center: Board center (x, y)
            radii: List of ring radii in pixels
            angles: List of segment boundary angles in degrees
            
        Returns:
            Array of shape (N, 2) with [x, y] pixel coordinates
        """
        points = []
        cx, cy = center
        
        # Add center point
        points.append([float(cx), float(cy)])
        
        # For each segment angle and each ring radius, compute intersection
        # Angles are in degrees from image coordinate system (-90° = top)
        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            for radius in radii:
                # Image coordinates: x = right, y = down
                x = cx + radius * cos_a
                y = cy + radius * sin_a
                points.append([x, y])
        
        return np.array(points, dtype=np.float32)
    
    def _create_visualization(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        radii: List[float],
        angles: List[float],
        points: np.ndarray
    ) -> np.ndarray:
        """
        Create visualization of detected features.
        
        Args:
            image: Original image
            center: Board center
            radii: Detected ring radii
            angles: Segment boundary angles
            points: Detected intersection points
            
        Returns:
            Annotated image
        """
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw center
        cv2.circle(vis, center, 5, (0, 255, 0), -1)
        
        # Draw detected circles
        for radius in radii:
            cv2.circle(vis, center, int(radius), (255, 0, 0), 2)
        
        # Draw segment boundaries (radial lines from center)
        cx, cy = center
        max_radius = int(radii[-1] * 1.2) if radii else 200
        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            x2 = int(cx + max_radius * np.cos(angle_rad))
            y2 = int(cy + max_radius * np.sin(angle_rad))
            cv2.line(vis, center, (x2, y2), (0, 255, 255), 1)
        
        # Draw intersection points
        for point in points:
            pt = (int(point[0]), int(point[1]))
            cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(vis, f"Center: {center}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Rings: {len(radii)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis, f"Segments: {len(angles)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"Points: {len(points)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis
    
    def match_to_reference(
        self,
        image_points: np.ndarray,
        reference_3d: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Match detected image points to 3D reference points.
        
        Uses geometric constraints to establish correspondence:
        - Center point is unique
        - Radii cluster into rings
        - Angles cluster into segments
        
        Args:
            image_points: Detected points in image (N, 2)
            reference_3d: 3D reference points (M, 3)
            
        Returns:
            Tuple of (matched_3d, matched_2d) with same point count,
            or None if matching failed
        """
        # Simple matching strategy:
        # Sort both sets by distance from center, then by angle
        
        if len(image_points) < len(reference_3d) * 0.7:
            logger.warning(f"Too few detected points: {len(image_points)} vs {len(reference_3d)} expected")
            return None
        
        # Extract center (first point in both sets)
        img_center = image_points[0]
        
        # Sort remaining points by radius, then angle
        img_pts = image_points[1:].copy()
        ref_pts = reference_3d[1:].copy()
        
        # Compute polar coordinates for image points
        dx_img = img_pts[:, 0] - img_center[0]
        dy_img = img_pts[:, 1] - img_center[1]
        r_img = np.sqrt(dx_img**2 + dy_img**2)
        theta_img = np.arctan2(dy_img, dx_img)
        
        # Compute polar coordinates for reference points
        r_ref = np.sqrt(ref_pts[:, 0]**2 + ref_pts[:, 1]**2)
        theta_ref = np.arctan2(ref_pts[:, 1], ref_pts[:, 0])
        
        # Sort both by radius, then angle
        img_order = np.lexsort((theta_img, r_img))
        ref_order = np.lexsort((theta_ref, r_ref))
        
        # Match closest N points
        n_match = min(len(img_pts), len(ref_pts))
        
        matched_3d = np.vstack([reference_3d[0], ref_pts[ref_order[:n_match]]])
        matched_2d = np.vstack([image_points[0], img_pts[img_order[:n_match]]])
        
        return matched_3d, matched_2d


def test_detection():
    """Test dartboard pattern detection with webcam or image file."""
    import sys
    
    detector = DartboardPatternDetector()
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        sys.exit(1)
    
    print("📷 Press SPACE to detect pattern, ESC to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Dartboard Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            print("\n🎯 Detecting dartboard features...")
            result = detector.detect_dartboard_features(frame, visualize=True)
            
            if result is not None:
                points, vis = result
                print(f"✅ Detected {len(points)} calibration points")
                cv2.imshow("Detection Result", vis)
                cv2.waitKey(0)
            else:
                print("❌ Detection failed")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_detection()
