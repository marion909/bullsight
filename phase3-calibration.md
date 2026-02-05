# 🗺️ Phase 3 – Calibration & Dartboard Mapping

**Dependencies:** [Phase 2 – Vision Engine](phase2-vision.md) ✅  
**Next Phase:** [Phase 4 – Game Engine](phase4-game-engine.md)

---

## 🎯 Phase Goals

- Transform pixel coordinates to dartboard fields
- Implement mathematical mapping (polar coordinates)
- Build calibration UI for board setup
- Support all dartboard segments and scoring zones
- Achieve 100% test coverage for mapping logic

---

## 📋 Prerequisites

### From Phase 2
- ✅ DartCoordinate extraction working
- ✅ Camera positioned and tested
- ✅ Vision engine stable

### Phase 3 Requirements
- Dartboard mounted permanently
- All 20 segments visible and unobstructed
- Lighting consistent and even

---

## 🧠 Mapping Strategy Overview

**Goal:** Convert pixel coordinates (x, y) → Dartboard field (segment, zone)

### Dartboard Structure

```
Standard Dartboard Layout (clockwise from top):
        20
    1       5
  18          12
9               9
  14          14
    11      6
        3
        
Scoring Zones (from center outward):
1. Bull's Eye (50 points) - center circle
2. Bull (25 points) - outer bull ring
3. Inner Single - main scoring area
4. Triple Ring (3x score)
5. Outer Single - outer scoring area
6. Double Ring (2x score)
```

### Mathematical Approach

```
Pixel (x, y) → Polar (radius, angle) → Field (segment, zone)

1. Calculate relative position from board center:
   dx = x - center_x
   dy = y - center_y

2. Convert to polar coordinates:
   radius = sqrt(dx² + dy²)
   angle = atan2(dy, dx)

3. Normalize angle to 0-360° (starting at segment 6):
   angle_deg = (angle_rad * 180/π + 90) % 360

4. Map angle to segment (20 segments, 18° each):
   segment_index = floor(angle_deg / 18)
   segment = SEGMENTS[segment_index]

5. Map radius to zone:
   if radius < bull_radius: "Bull's Eye"
   elif radius < outer_bull_radius: "Bull"
   elif radius < triple_inner_radius: "Inner Single"
   elif radius < triple_outer_radius: "Triple"
   elif radius < double_inner_radius: "Outer Single"
   elif radius < double_outer_radius: "Double"
   else: "Miss"
```

---

## 🔧 Implementation Tasks

### 3.1 Board Mapper Core Module

**Task:** Implement coordinate-to-field transformation

**File:** `src/calibration/board_mapper.py`

```python
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
        segment (int): Segment number (1-20, or 25 for bull)
        zone (str): Scoring zone ('bull_eye', 'bull', 'single', 'double', 'triple', 'miss')
        score (int): Points value for this field
        multiplier (int): Score multiplier (1, 2, or 3)
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
        center_x (int): X coordinate of dartboard center
        center_y (int): Y coordinate of dartboard center
        bull_eye_radius (float): Radius of bull's eye in pixels
        bull_radius (float): Radius of outer bull ring
        triple_inner_radius (float): Inner radius of triple ring
        triple_outer_radius (float): Outer radius of triple ring
        double_inner_radius (float): Inner radius of double ring
        double_outer_radius (float): Outer radius of double ring
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
        calibration (CalibrationData): Board calibration parameters
        segments (List[int]): Dartboard segment sequence
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
        # Rotate by 90° so segment 6 is at the right (standard dart orientation)
        angle_deg = (np.degrees(angle_rad) + 90) % 360
        
        return radius, angle_deg
    
    def angle_to_segment(self, angle: float) -> int:
        """
        Map angle to dartboard segment number.
        
        Args:
            angle: Angle in degrees (0-360)
            
        Returns:
            Segment number (1-20)
        """
        # Each segment spans 18 degrees (360 / 20 = 18)
        # Segments start at -9° offset from top (to center segment 20 at top)
        adjusted_angle = (angle + 9) % 360
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
            Tuple of (start_angle, end_angle) in degrees
            
        Raises:
            ValueError: If segment not found
        """
        try:
            segment_index = self.segments.index(segment)
        except ValueError:
            raise ValueError(f"Invalid segment number: {segment}")
        
        # Each segment is 18 degrees, offset by -9 to center segment 20 at top
        start_angle = (segment_index * 18 - 9) % 360
        end_angle = (start_angle + 18) % 360
        
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
    center_x = image_width // 2
    center_y = image_height // 2
    
    # Standard dartboard proportions (as fraction of total radius)
    bull_eye_radius = board_radius_pixels * 0.04      # ~12.7mm of 170mm
    bull_radius = board_radius_pixels * 0.09          # ~31.8mm
    triple_inner_radius = board_radius_pixels * 0.60  # ~99mm
    triple_outer_radius = board_radius_pixels * 0.65  # ~107mm
    double_inner_radius = board_radius_pixels * 0.94  # ~162mm
    double_outer_radius = board_radius_pixels * 1.0   # ~170mm
    
    return CalibrationData(
        center_x=center_x,
        center_y=center_y,
        bull_eye_radius=bull_eye_radius,
        bull_radius=bull_radius,
        triple_inner_radius=triple_inner_radius,
        triple_outer_radius=triple_outer_radius,
        double_inner_radius=double_inner_radius,
        double_outer_radius=double_outer_radius
    )
```

**Expected Outcome:** Complete mapping system with validation

---

### 3.2 Calibration UI Module

**Task:** PyQt6 interface for board calibration

**File:** `src/ui/calibration_screen.py`

```python
"""
Calibration UI for dartboard setup.

Provides interactive interface for setting board center, bull radius,
and ring boundaries with visual feedback.

Author: Mario Neuhauser
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPen, QPixmap, QImage, QColor
import cv2
import numpy as np
from typing import Optional

from src.calibration.board_mapper import CalibrationData, create_default_calibration


class CalibrationScreen(QWidget):
    """
    Interactive calibration screen for dartboard setup.
    
    Allows user to:
    - Set board center point
    - Adjust bull radius
    - Configure triple and double ring boundaries
    - Preview calibration overlay
    
    Signals:
        calibration_complete: Emitted when calibration is finalized
    """
    
    calibration_complete = pyqtSignal(CalibrationData)
    
    def __init__(self, camera_image: np.ndarray):
        """
        Initialize calibration screen.
        
        Args:
            camera_image: Current camera image for calibration overlay
        """
        super().__init__()
        self.camera_image = camera_image
        self.image_height, self.image_width = camera_image.shape[:2]
        
        # Initialize with default calibration
        self.calibration = create_default_calibration(
            self.image_width,
            self.image_height,
            board_radius_pixels=min(self.image_width, self.image_height) * 0.4
        )
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup calibration UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Dartboard Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        # Image display with overlay
        self.image_label = QLabel()
        self.update_image_display()
        layout.addWidget(self.image_label)
        
        # Center point group
        center_group = QGroupBox("Board Center")
        center_layout = QVBoxLayout()
        
        self.center_label = QLabel(
            f"X: {self.calibration.center_x}, Y: {self.calibration.center_y}"
        )
        center_layout.addWidget(self.center_label)
        
        center_btn = QPushButton("Click Image to Set Center")
        center_btn.setMinimumHeight(60)
        center_btn.clicked.connect(self.enable_center_selection)
        center_layout.addWidget(center_btn)
        
        center_group.setLayout(center_layout)
        layout.addWidget(center_group)
        
        # Radii adjustments
        radii_group = QGroupBox("Ring Radii")
        radii_layout = QVBoxLayout()
        
        # Bull's eye radius
        self.bull_eye_slider = self.create_slider(
            "Bull's Eye Radius",
            10, 100,
            int(self.calibration.bull_eye_radius),
            self.on_bull_eye_changed
        )
        radii_layout.addWidget(self.bull_eye_slider[0])
        
        # Bull radius
        self.bull_slider = self.create_slider(
            "Bull Radius",
            20, 150,
            int(self.calibration.bull_radius),
            self.on_bull_changed
        )
        radii_layout.addWidget(self.bull_slider[0])
        
        # Triple inner
        self.triple_inner_slider = self.create_slider(
            "Triple Inner Radius",
            100, 400,
            int(self.calibration.triple_inner_radius),
            self.on_triple_inner_changed
        )
        radii_layout.addWidget(self.triple_inner_slider[0])
        
        # Triple outer
        self.triple_outer_slider = self.create_slider(
            "Triple Outer Radius",
            110, 420,
            int(self.calibration.triple_outer_radius),
            self.on_triple_outer_changed
        )
        radii_layout.addWidget(self.triple_outer_slider[0])
        
        # Double inner
        self.double_inner_slider = self.create_slider(
            "Double Inner Radius",
            300, 600,
            int(self.calibration.double_inner_radius),
            self.on_double_inner_changed
        )
        radii_layout.addWidget(self.double_inner_slider[0])
        
        # Double outer
        self.double_outer_slider = self.create_slider(
            "Double Outer Radius",
            310, 650,
            int(self.calibration.double_outer_radius),
            self.on_double_outer_changed
        )
        radii_layout.addWidget(self.double_outer_slider[0])
        
        radii_group.setLayout(radii_layout)
        layout.addWidget(radii_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.setMinimumHeight(60)
        reset_btn.clicked.connect(self.reset_calibration)
        button_layout.addWidget(reset_btn)
        
        save_btn = QPushButton("Save Calibration")
        save_btn.setMinimumHeight(60)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        save_btn.clicked.connect(self.save_calibration)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.center_selection_mode = False
    
    def create_slider(
        self,
        label: str,
        min_val: int,
        max_val: int,
        current: int,
        callback
    ) -> tuple:
        """Create labeled slider widget."""
        group = QWidget()
        layout = QVBoxLayout()
        
        label_widget = QLabel(f"{label}: {current}")
        layout.addWidget(label_widget)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(current)
        slider.valueChanged.connect(lambda v: self.update_slider_label(label_widget, label, v))
        slider.valueChanged.connect(callback)
        layout.addWidget(slider)
        
        group.setLayout(layout)
        return group, slider, label_widget
    
    def update_slider_label(self, label: QLabel, name: str, value: int) -> None:
        """Update slider label with current value."""
        label.setText(f"{name}: {value}")
    
    def enable_center_selection(self) -> None:
        """Enable center point selection mode."""
        self.center_selection_mode = True
        self.image_label.mousePressEvent = self.on_image_click
    
    def on_image_click(self, event) -> None:
        """Handle mouse click on image for center selection."""
        if not self.center_selection_mode:
            return
        
        pos = event.pos()
        self.calibration.center_x = pos.x()
        self.calibration.center_y = pos.y()
        
        self.center_label.setText(
            f"X: {self.calibration.center_x}, Y: {self.calibration.center_y}"
        )
        
        self.center_selection_mode = False
        self.update_image_display()
    
    def on_bull_eye_changed(self, value: int) -> None:
        """Handle bull's eye radius change."""
        self.calibration.bull_eye_radius = float(value)
        self.update_image_display()
    
    def on_bull_changed(self, value: int) -> None:
        """Handle bull radius change."""
        self.calibration.bull_radius = float(value)
        self.update_image_display()
    
    def on_triple_inner_changed(self, value: int) -> None:
        """Handle triple inner radius change."""
        self.calibration.triple_inner_radius = float(value)
        self.update_image_display()
    
    def on_triple_outer_changed(self, value: int) -> None:
        """Handle triple outer radius change."""
        self.calibration.triple_outer_radius = float(value)
        self.update_image_display()
    
    def on_double_inner_changed(self, value: int) -> None:
        """Handle double inner radius change."""
        self.calibration.double_inner_radius = float(value)
        self.update_image_display()
    
    def on_double_outer_changed(self, value: int) -> None:
        """Handle double outer radius change."""
        self.calibration.double_outer_radius = float(value)
        self.update_image_display()
    
    def update_image_display(self) -> None:
        """Update image display with calibration overlay."""
        display_image = self.camera_image.copy()
        
        # Draw calibration overlay
        center = (self.calibration.center_x, self.calibration.center_y)
        
        # Draw rings
        cv2.circle(display_image, center, int(self.calibration.bull_eye_radius), (0, 0, 255), 2)
        cv2.circle(display_image, center, int(self.calibration.bull_radius), (0, 255, 255), 2)
        cv2.circle(display_image, center, int(self.calibration.triple_inner_radius), (0, 255, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.triple_outer_radius), (0, 255, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.double_inner_radius), (255, 0, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.double_outer_radius), (255, 0, 0), 2)
        
        # Draw center crosshair
        cv2.drawMarker(display_image, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        # Convert to QPixmap
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.image_label.setPixmap(pixmap)
    
    def reset_calibration(self) -> None:
        """Reset calibration to default values."""
        self.calibration = create_default_calibration(
            self.image_width,
            self.image_height,
            board_radius_pixels=min(self.image_width, self.image_height) * 0.4
        )
        
        # Update UI elements
        self.center_label.setText(
            f"X: {self.calibration.center_x}, Y: {self.calibration.center_y}"
        )
        self.bull_eye_slider[1].setValue(int(self.calibration.bull_eye_radius))
        self.bull_slider[1].setValue(int(self.calibration.bull_radius))
        self.triple_inner_slider[1].setValue(int(self.calibration.triple_inner_radius))
        self.triple_outer_slider[1].setValue(int(self.calibration.triple_outer_radius))
        self.double_inner_slider[1].setValue(int(self.calibration.double_inner_radius))
        self.double_outer_slider[1].setValue(int(self.calibration.double_outer_radius))
        
        self.update_image_display()
    
    def save_calibration(self) -> None:
        """Emit calibration complete signal."""
        self.calibration_complete.emit(self.calibration)
```

**Expected Outcome:** Interactive calibration UI with real-time preview

---

### 3.3 Unit Tests for BoardMapper

**Task:** Comprehensive mapping logic tests

**File:** `tests/unit/test_board_mapper.py`

```python
"""
Unit tests for BoardMapper coordinate transformation.

Tests all mapping logic including polar conversion, segment mapping,
zone detection, and score calculation.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

from src.calibration.board_mapper import (
    BoardMapper, CalibrationData, DartboardField,
    create_default_calibration, DARTBOARD_SEGMENTS
)


class TestCalibrationData:
    """Tests for CalibrationData dataclass."""
    
    def test_calibration_creation(self):
        """Test CalibrationData initialization."""
        cal = CalibrationData(
            center_x=400, center_y=300,
            bull_eye_radius=10.0, bull_radius=25.0,
            triple_inner_radius=100.0, triple_outer_radius=110.0,
            double_inner_radius=160.0, double_outer_radius=170.0
        )
        
        assert cal.center_x == 400
        assert cal.center_y == 300
        assert cal.bull_eye_radius == 10.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        cal = CalibrationData(400, 300, 10.0, 25.0, 100.0, 110.0, 160.0, 170.0)
        data = cal.to_dict()
        
        assert isinstance(data, dict)
        assert data["center_x"] == 400
        assert data["bull_eye_radius"] == 10.0
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "center_x": 400, "center_y": 300,
            "bull_eye_radius": 10.0, "bull_radius": 25.0,
            "triple_inner_radius": 100.0, "triple_outer_radius": 110.0,
            "double_inner_radius": 160.0, "double_outer_radius": 170.0
        }
        
        cal = CalibrationData.from_dict(data)
        assert cal.center_x == 400
        assert cal.bull_eye_radius == 10.0


class TestDartboardField:
    """Tests for DartboardField dataclass."""
    
    def test_field_creation(self):
        """Test DartboardField initialization."""
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        
        assert field.segment == 20
        assert field.zone == "triple"
        assert field.score == 60
        assert field.multiplier == 3
    
    def test_field_repr(self):
        """Test string representations."""
        assert "Bull's Eye" in repr(DartboardField(25, "bull_eye", 50, 1))
        assert "Bull" in repr(DartboardField(25, "bull", 25, 1))
        assert "Miss" in repr(DartboardField(0, "miss", 0, 0))
        assert "Triple 20" in repr(DartboardField(20, "triple", 60, 3))


class TestBoardMapper:
    """Tests for BoardMapper class."""
    
    @pytest.fixture
    def calibration(self):
        """Create test calibration data."""
        return CalibrationData(
            center_x=400, center_y=300,
            bull_eye_radius=10.0,
            bull_radius=25.0,
            triple_inner_radius=100.0,
            triple_outer_radius=110.0,
            double_inner_radius=160.0,
            double_outer_radius=170.0
        )
    
    @pytest.fixture
    def mapper(self, calibration):
        """Create BoardMapper with calibration."""
        mapper = BoardMapper(calibration)
        return mapper
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = BoardMapper()
        assert mapper.calibration is None
        assert mapper.segments == DARTBOARD_SEGMENTS
    
    def test_set_calibration(self, calibration):
        """Test setting calibration."""
        mapper = BoardMapper()
        mapper.set_calibration(calibration)
        
        assert mapper.calibration == calibration
    
    def test_save_and_load_calibration(self, mapper, tmp_path):
        """Test calibration persistence."""
        filepath = tmp_path / "test_calibration.json"
        
        # Save
        mapper.save_calibration(filepath)
        assert filepath.exists()
        
        # Load
        new_mapper = BoardMapper()
        new_mapper.load_calibration(filepath)
        
        assert new_mapper.calibration.center_x == mapper.calibration.center_x
        assert new_mapper.calibration.bull_eye_radius == mapper.calibration.bull_eye_radius
    
    def test_load_nonexistent_file(self):
        """Test error when loading nonexistent file."""
        mapper = BoardMapper()
        with pytest.raises(FileNotFoundError):
            mapper.load_calibration(Path("nonexistent.json"))
    
    def test_save_without_calibration(self, tmp_path):
        """Test error when saving without calibration."""
        mapper = BoardMapper()
        with pytest.raises(RuntimeError, match="No calibration data"):
            mapper.save_calibration(tmp_path / "test.json")
    
    def test_pixel_to_polar_center(self, mapper):
        """Test polar conversion at center."""
        radius, angle = mapper.pixel_to_polar(400, 300)
        
        assert radius == pytest.approx(0.0)
        # Angle undefined at center, but should be valid number
        assert 0 <= angle < 360
    
    def test_pixel_to_polar_right(self, mapper):
        """Test polar conversion at right (90 degrees)."""
        radius, angle = mapper.pixel_to_polar(500, 300)
        
        assert radius == pytest.approx(100.0)
        assert angle == pytest.approx(90.0)
    
    def test_pixel_to_polar_top(self, mapper):
        """Test polar conversion at top (0 degrees)."""
        radius, angle = mapper.pixel_to_polar(400, 200)
        
        assert radius == pytest.approx(100.0)
        assert angle == pytest.approx(0.0)
    
    def test_pixel_to_polar_without_calibration(self):
        """Test error when converting without calibration."""
        mapper = BoardMapper()
        with pytest.raises(RuntimeError, match="Calibration must be set"):
            mapper.pixel_to_polar(100, 100)
    
    def test_angle_to_segment_top(self, mapper):
        """Test segment mapping at top (segment 20)."""
        segment = mapper.angle_to_segment(0)
        assert segment == 20
    
    def test_angle_to_segment_right(self, mapper):
        """Test segment mapping at right (segment 6)."""
        segment = mapper.angle_to_segment(90)
        assert segment == 6
    
    def test_angle_to_segment_all_20(self, mapper):
        """Test all 20 segments can be mapped."""
        segments_found = set()
        
        for angle in range(0, 360, 5):
            segment = mapper.angle_to_segment(angle)
            segments_found.add(segment)
        
        # Should find all 20 unique segments
        assert len(segments_found) == 20
        assert segments_found == set(DARTBOARD_SEGMENTS)
    
    def test_radius_to_zone_bull_eye(self, mapper):
        """Test zone detection for bull's eye."""
        zone = mapper.radius_to_zone(5.0)
        assert zone == "bull_eye"
    
    def test_radius_to_zone_bull(self, mapper):
        """Test zone detection for bull."""
        zone = mapper.radius_to_zone(20.0)
        assert zone == "bull"
    
    def test_radius_to_zone_triple(self, mapper):
        """Test zone detection for triple."""
        zone = mapper.radius_to_zone(105.0)
        assert zone == "triple"
    
    def test_radius_to_zone_double(self, mapper):
        """Test zone detection for double."""
        zone = mapper.radius_to_zone(165.0)
        assert zone == "double"
    
    def test_radius_to_zone_single_inner(self, mapper):
        """Test zone detection for inner single."""
        zone = mapper.radius_to_zone(80.0)
        assert zone == "single"
    
    def test_radius_to_zone_single_outer(self, mapper):
        """Test zone detection for outer single."""
        zone = mapper.radius_to_zone(140.0)
        assert zone == "single"
    
    def test_radius_to_zone_miss(self, mapper):
        """Test zone detection for miss."""
        zone = mapper.radius_to_zone(200.0)
        assert zone == "miss"
    
    def test_radius_to_zone_without_calibration(self):
        """Test error when detecting zone without calibration."""
        mapper = BoardMapper()
        with pytest.raises(RuntimeError, match="Calibration must be set"):
            mapper.radius_to_zone(100.0)
    
    def test_calculate_score_bull_eye(self, mapper):
        """Test score calculation for bull's eye."""
        score, multiplier = mapper.calculate_score(25, "bull_eye")
        assert score == 50
        assert multiplier == 1
    
    def test_calculate_score_bull(self, mapper):
        """Test score calculation for bull."""
        score, multiplier = mapper.calculate_score(25, "bull")
        assert score == 25
        assert multiplier == 1
    
    def test_calculate_score_triple(self, mapper):
        """Test score calculation for triple."""
        score, multiplier = mapper.calculate_score(20, "triple")
        assert score == 60
        assert multiplier == 3
    
    def test_calculate_score_double(self, mapper):
        """Test score calculation for double."""
        score, multiplier = mapper.calculate_score(20, "double")
        assert score == 40
        assert multiplier == 2
    
    def test_calculate_score_single(self, mapper):
        """Test score calculation for single."""
        score, multiplier = mapper.calculate_score(20, "single")
        assert score == 20
        assert multiplier == 1
    
    def test_calculate_score_miss(self, mapper):
        """Test score calculation for miss."""
        score, multiplier = mapper.calculate_score(0, "miss")
        assert score == 0
        assert multiplier == 0
    
    def test_map_coordinate_center(self, mapper):
        """Test complete mapping for center (bull's eye)."""
        field = mapper.map_coordinate(400, 300)
        
        assert field.segment == 25
        assert field.zone == "bull_eye"
        assert field.score == 50
    
    def test_map_coordinate_triple_20(self, mapper):
        """Test mapping for triple 20 (top)."""
        # Top center, in triple ring
        field = mapper.map_coordinate(400, 300 - 105)
        
        assert field.segment == 20
        assert field.zone == "triple"
        assert field.score == 60
    
    def test_map_coordinate_double_6(self, mapper):
        """Test mapping for double 6 (right)."""
        # Right, in double ring
        field = mapper.map_coordinate(400 + 165, 300)
        
        assert field.segment == 6
        assert field.zone == "double"
        assert field.score == 12
    
    def test_map_coordinate_miss(self, mapper):
        """Test mapping for miss (outside board)."""
        field = mapper.map_coordinate(400 + 200, 300)
        
        assert field.zone == "miss"
        assert field.score == 0
    
    def test_validate_calibration_valid(self, mapper):
        """Test validation with valid calibration."""
        checks = mapper.validate_calibration()
        
        assert checks["all_checks_passed"] is True
        assert checks["calibration_set"] is True
        assert checks["bull_eye_smaller_than_bull"] is True
    
    def test_validate_calibration_none(self):
        """Test validation without calibration."""
        mapper = BoardMapper()
        checks = mapper.validate_calibration()
        
        assert checks["calibration_set"] is False
    
    def test_validate_calibration_invalid_order(self):
        """Test validation with invalid radius order."""
        bad_cal = CalibrationData(
            center_x=400, center_y=300,
            bull_eye_radius=50.0,  # Too large!
            bull_radius=25.0,
            triple_inner_radius=100.0,
            triple_outer_radius=110.0,
            double_inner_radius=160.0,
            double_outer_radius=170.0
        )
        mapper = BoardMapper(bad_cal)
        checks = mapper.validate_calibration()
        
        assert checks["all_checks_passed"] is False
        assert checks["bull_eye_smaller_than_bull"] is False
    
    def test_get_segment_bounds(self, mapper):
        """Test segment boundary calculation."""
        start, end = mapper.get_segment_bounds(20)
        
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert 0 <= start < 360
        assert 0 <= end < 360
    
    def test_get_segment_bounds_invalid(self, mapper):
        """Test error for invalid segment."""
        with pytest.raises(ValueError, match="Invalid segment"):
            mapper.get_segment_bounds(99)


def test_create_default_calibration():
    """Test default calibration generation."""
    cal = create_default_calibration(800, 600, 200.0)
    
    assert cal.center_x == 400
    assert cal.center_y == 300
    assert cal.bull_eye_radius == pytest.approx(8.0)
    assert cal.bull_radius == pytest.approx(18.0)
    assert cal.double_outer_radius == pytest.approx(200.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.calibration", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
pytest tests/unit/test_board_mapper.py -v --cov=src.calibration --cov-report=term-missing
```

**Expected Coverage:** 100%

---

## ⚠️ Phase 3 Risks & Mitigations

### Risk: Camera Distortion

**Symptoms:** Coordinates at edges map incorrectly, segments skewed

**Mitigation:**
1. Ensure camera is perfectly perpendicular to board
2. Use camera calibration matrix for lens distortion correction
3. Test all 20 segments during calibration
4. Consider polynomial correction for edge pixels

**Test All Segments:**
```python
def test_all_segments_reachable(mapper):
    """Verify all 20 segments can be hit."""
    for segment in DARTBOARD_SEGMENTS:
        start_angle, end_angle = mapper.get_segment_bounds(segment)
        mid_angle = (start_angle + end_angle) / 2
        
        # Test at triple ring
        radius = mapper.calibration.triple_inner_radius + 5
        # ... convert polar to pixel and test
```

---

## ✅ Phase 3 Completion Checklist

### Code Implementation
- [ ] `src/calibration/board_mapper.py` complete (100% coverage)
- [ ] `src/ui/calibration_screen.py` complete
- [ ] CalibrationData JSON serialization working
- [ ] All docstrings present

### Testing
- [ ] Unit tests pass (100% coverage)
- [ ] All 20 segments tested
- [ ] All 5 zones tested (bull_eye, bull, single, double, triple)
- [ ] Edge cases covered (center, edge, out of bounds)
- [ ] Calibration validation working

### Calibration Accuracy
- [ ] Bull's eye correctly identified (center ±5 pixels)
- [ ] All segments map correctly (tested physically)
- [ ] Triple ring accuracy >95%
- [ ] Double ring accuracy >95%
- [ ] Zone boundaries verified

### UI
- [ ] Calibration screen displays camera feed
- [ ] Ring overlays visible and adjustable
- [ ] Center selection works
- [ ] Save/load calibration functional
- [ ] Touch-friendly controls (60px+ buttons)

---

## 📊 Expected Coverage Report

```
src/calibration/board_mapper.py          387      0   100%
src/ui/calibration_screen.py            245      0   100%
tests/unit/test_board_mapper.py          428      0   100%
------------------------------------------------------------
TOTAL                                   1060      0   100%
```

---

## 🔗 Next Steps

Once Phase 3 is complete with 100% coverage:

**Proceed to:** [Phase 4 – Game Engine](phase4-game-engine.md)

**Phase 4 Requirements:**
- ✅ Pixel-to-field mapping working
- ✅ DartboardField objects available
- ✅ Calibration persistence functional

**Phase 4 Will Use:**
- `map_coordinate()` from [board_mapper.py](src/calibration/board_mapper.py)
- `DartboardField.score` for game logic
- `CalibrationData` loading at game start

---

**Phase Status:** 🔴 Not Started  
**Estimated Duration:** 1 week  
**Test Coverage:** 0% → Target: 100%  
**Dependencies:** Phase 2 ✅
