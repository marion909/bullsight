"""
Calibration UI for dartboard setup.

Provides interactive interface for setting board center and ring boundaries
with visual feedback on camera image.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QGroupBox, QMessageBox, QInputDialog, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox, QLineEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import QPainter, QPen, QPixmap, QImage, QColor
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List, Tuple

from src.calibration.board_mapper import CalibrationData, create_default_calibration

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.main import BullSightApp


class GroundTruthDialog(QDialog):
    """Dialog to enter ground truth data for test image."""
    
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.setWindowTitle("Enter Ground Truth Data")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        
        # Show captured image (small preview)
        height, width = self.frame.shape[:2]
        scale = 400 / width
        small_frame = cv2.resize(self.frame, None, fx=scale, fy=scale)
        
        q_image = QImage(small_frame.data, small_frame.shape[1], small_frame.shape[0],
                        small_frame.shape[1] * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
        
        # Form for ground truth data
        form_layout = QFormLayout()
        
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, width)
        self.x_spin.setValue(width // 2)
        form_layout.addRow("Dart X Position (pixels):", self.x_spin)
        
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, height)
        self.y_spin.setValue(height // 2)
        form_layout.addRow("Dart Y Position (pixels):", self.y_spin)
        
        self.segment_edit = QLineEdit()
        self.segment_edit.setPlaceholderText("e.g., T20, D16, Bull, 5")
        form_layout.addRow("Segment Hit:", self.segment_edit)
        
        self.score_spin = QSpinBox()
        self.score_spin.setRange(0, 180)
        self.score_spin.setValue(20)
        form_layout.addRow("Score:", self.score_spin)
        
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("e.g., Center triple, Edge of board")
        form_layout.addRow("Description (optional):", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_ground_truth(self) -> dict:
        """Get entered ground truth data."""
        return {
            "x": self.x_spin.value(),
            "y": self.y_spin.value(),
            "segment": self.segment_edit.text() or "Unknown",
            "score": self.score_spin.value(),
            "description": self.description_edit.text() or ""
        }


class CalibrationScreen(QWidget):
    """
    Interactive calibration screen for dartboard setup.
    
    Allows user to set board center and adjust ring boundaries
    with real-time visual feedback.
    
    Signals:
        calibration_complete: Emitted when calibration is finalized
    """
    
    calibration_complete = Signal(CalibrationData)
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize calibration screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        self.camera_image: Optional[np.ndarray] = None
        self.image_height = 480
        self.image_width = 640
        
        # Track displayed image size for coordinate mapping
        self.displayed_pixmap: Optional[QPixmap] = None
        
        # State
        self.selecting_center = False
        self.drag_mode_enabled = False
        self.dragging_point = None  # (point_type, point_index) or None
        
        # Perspective transformation values
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.rotation = 0.0
        self.skew_x = 0.0
        self.skew_y = 0.0
        self.triple_scale = 1.0
        self.double_scale = 1.0
        
        # Will be set in showEvent() to use loaded calibration
        self.calibration = None
        
        # Control points for manual dragging
        # Each point: {"ring": ring_name, "angle": degrees, "x": pixel_x, "y": pixel_y}
        self.control_points: List[dict] = []
        
        # Ellipse parameters per ring (for deformed drawing)
        self.ring_ellipses = {}
        
        # Setup live camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        
        self.setup_ui()
    
    def showEvent(self, event):
        """Start camera and load calibration when screen is shown."""
        super().showEvent(event)
        
        # Load calibration from mapper (after app has loaded config)
        if self.calibration is None:
            if self.app.mapper.calibration:
                self.calibration = self.app.mapper.calibration
                logger.info(f"Using loaded calibration: center=({self.calibration.center_x}, {self.calibration.center_y})")
            else:
                # Initialize with default calibration
                estimated_radius = min(self.image_width, self.image_height) * 0.4
                self.calibration = create_default_calibration(
                    self.image_width,
                    self.image_height,
                    board_radius_pixels=estimated_radius
                )
                logger.info("Created default calibration")
        
        # Update all UI sliders with calibration values (including perspective transformations)
        self.update_ui_from_calibration()
        
        if self.app.start_camera():
            self.camera_timer.start(100)  # Update every 100ms
    
    def hideEvent(self, event):
        """Stop camera updates when screen is hidden."""
        super().hideEvent(event)
        self.camera_timer.stop()
    
    def update_camera_feed(self):
        """Update camera feed continuously."""
        try:
            if self.app.camera and self.app.camera.is_started:
                stereo = self.app.camera.capture_stereo()
                if stereo is not None:
                    self.camera_image = stereo.left
                    self.update_image_display()
        except Exception as e:
            pass  # Silently ignore camera errors during live feed
    
    def setup_ui(self) -> None:
        """Setup calibration UI layout."""
        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        # Left side: Camera image
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Title
        title = QLabel("Dartboard Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Click 'Set Center' and click on bull's eye\n"
            "2. Use sliders (right) to adjust overlay\n"
            "3. Match rings and segment lines\n"
            "4. Click 'Save Calibration'"
        )
        instructions.setStyleSheet("padding: 10px; background: #f0f0f0;")
        instructions.setWordWrap(True)
        left_layout.addWidget(instructions)
        
        # Image display with overlay
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_press
        self.image_label.mouseMoveEvent = self.on_image_move
        self.image_label.mouseReleaseEvent = self.on_image_release
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setMaximumSize(800, 600)
        self.update_image_display()
        left_layout.addWidget(self.image_label)
        left_layout.addStretch()
        
        main_layout.addWidget(left_widget, stretch=2)
        
        # Right side: Controls in scroll area
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Center point control
        center_group = QGroupBox("Board Center")
        center_layout = QVBoxLayout()
        
        self.center_label = QLabel("X: 0, Y: 0")
        center_layout.addWidget(self.center_label)
        
        # Drag mode toggle
        self.drag_mode_btn = QPushButton("🎯 Enable Drag Mode")
        self.drag_mode_btn.setMinimumHeight(60)
        self.drag_mode_btn.setStyleSheet("background-color: #9E9E9E; color: white; font-weight: bold;")
        self.drag_mode_btn.setCheckable(True)
        self.drag_mode_btn.clicked.connect(self.toggle_drag_mode)
        center_layout.addWidget(self.drag_mode_btn)
        
        drag_info = QLabel("Drag control points on rings/segments to adjust calibration manually")
        drag_info.setStyleSheet("padding: 5px; background: #e3f2fd; font-size: 11px;")
        drag_info.setWordWrap(True)
        center_layout.addWidget(drag_info)
        
        self.center_btn = QPushButton("Set Center")
        self.center_btn.setMinimumHeight(60)
        self.center_btn.clicked.connect(self.enable_center_selection)
        center_layout.addWidget(self.center_btn)
        
        self.capture_ref_btn = QPushButton("Capture Reference Image")
        self.capture_ref_btn.setMinimumHeight(60)
        self.capture_ref_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.capture_ref_btn.clicked.connect(self.capture_reference_image)
        center_layout.addWidget(self.capture_ref_btn)
        
        self.capture_test_btn = QPushButton("Capture Test Image")
        self.capture_test_btn.setMinimumHeight(60)
        self.capture_test_btn.setStyleSheet("background-color: #FF9800; color: white;")
        self.capture_test_btn.setToolTip("Capture image with dart and enter ground truth for parameter optimization")
        self.capture_test_btn.clicked.connect(self.capture_test_image)
        center_layout.addWidget(self.capture_test_btn)
        
        center_group.setLayout(center_layout)
        controls_layout.addWidget(center_group)
        
        # Radii adjustments
        radii_group = QGroupBox("Ring Radii (pixels)")
        radii_layout = QVBoxLayout()
        
        # Global scaling info
        scale_info = QLabel("Global Scaling: Stretch/compress ring groups together")
        scale_info.setStyleSheet("padding: 5px; background: #fff3cd; font-weight: bold;")
        scale_info.setWordWrap(True)
        radii_layout.addWidget(scale_info)
        
        # Triple ring scale
        radii_layout.addWidget(QLabel("Triple Ring Scale"))
        self.triple_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.triple_scale_slider.setRange(50, 150)
        self.triple_scale_slider.setValue(100)
        self.triple_scale_slider.valueChanged.connect(self.on_triple_scale_changed)
        radii_layout.addWidget(self.triple_scale_slider)
        self.triple_scale_label = QLabel("100%")
        radii_layout.addWidget(self.triple_scale_label)
        
        # Double ring scale
        radii_layout.addWidget(QLabel("Double Ring Scale"))
        self.double_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.double_scale_slider.setRange(50, 150)
        self.double_scale_slider.setValue(100)
        self.double_scale_slider.valueChanged.connect(self.on_double_scale_changed)
        radii_layout.addWidget(self.double_scale_slider)
        self.double_scale_label = QLabel("100%")
        radii_layout.addWidget(self.double_scale_label)
        
        # Separator
        separator = QLabel("Individual Ring Adjustments:")
        separator.setStyleSheet("margin-top: 10px; font-weight: bold;")
        radii_layout.addWidget(separator)
        
        # Bull's eye
        radii_layout.addWidget(QLabel("Bull's Eye Radius"))
        self.bull_eye_slider = QSlider(Qt.Orientation.Horizontal)
        self.bull_eye_slider.setRange(3, 30)
        self.bull_eye_slider.setValue(12)
        self.bull_eye_slider.valueChanged.connect(self.on_bull_eye_changed)
        radii_layout.addWidget(self.bull_eye_slider)
        self.bull_eye_label = QLabel("12 px")
        radii_layout.addWidget(self.bull_eye_label)
        
        # Bull
        radii_layout.addWidget(QLabel("Bull Radius"))
        self.bull_slider = QSlider(Qt.Orientation.Horizontal)
        self.bull_slider.setRange(8, 60)
        self.bull_slider.setValue(25)
        self.bull_slider.valueChanged.connect(self.on_bull_changed)
        radii_layout.addWidget(self.bull_slider)
        self.bull_label = QLabel("25 px")
        radii_layout.addWidget(self.bull_label)
        
        # Triple inner
        radii_layout.addWidget(QLabel("Triple Inner Radius"))
        self.triple_inner_slider = QSlider(Qt.Orientation.Horizontal)
        self.triple_inner_slider.setRange(30, 250)
        self.triple_inner_slider.setValue(80)
        self.triple_inner_slider.valueChanged.connect(self.on_triple_inner_changed)
        radii_layout.addWidget(self.triple_inner_slider)
        self.triple_inner_label = QLabel("80 px")
        radii_layout.addWidget(self.triple_inner_label)
        
        # Triple outer
        radii_layout.addWidget(QLabel("Triple Outer Radius"))
        self.triple_outer_slider = QSlider(Qt.Orientation.Horizontal)
        self.triple_outer_slider.setRange(40, 280)
        self.triple_outer_slider.setValue(95)
        self.triple_outer_slider.valueChanged.connect(self.on_triple_outer_changed)
        radii_layout.addWidget(self.triple_outer_slider)
        self.triple_outer_label = QLabel("95 px")
        radii_layout.addWidget(self.triple_outer_label)
        
        # Double inner
        radii_layout.addWidget(QLabel("Double Inner Radius"))
        self.double_inner_slider = QSlider(Qt.Orientation.Horizontal)
        self.double_inner_slider.setRange(100, 400)
        self.double_inner_slider.setValue(150)
        self.double_inner_slider.valueChanged.connect(self.on_double_inner_changed)
        radii_layout.addWidget(self.double_inner_slider)
        self.double_inner_label = QLabel("150 px")
        radii_layout.addWidget(self.double_inner_label)
        
        # Double outer
        radii_layout.addWidget(QLabel("Double Outer Radius"))
        self.double_outer_slider = QSlider(Qt.Orientation.Horizontal)
        self.double_outer_slider.setRange(110, 450)
        self.double_outer_slider.setValue(165)
        self.double_outer_slider.valueChanged.connect(self.on_double_outer_changed)
        radii_layout.addWidget(self.double_outer_slider)
        self.double_outer_label = QLabel("165 px")
        radii_layout.addWidget(self.double_outer_label)
        
        radii_group.setLayout(radii_layout)
        controls_layout.addWidget(radii_group)
        
        # Perspective adjustments
        perspective_group = QGroupBox("Perspective Adjustment (for angled camera)")
        perspective_layout = QVBoxLayout()
        
        perspective_info = QLabel(
            "Adjust these if camera is mounted at an angle (ellipse instead of circle)"
        )
        perspective_info.setStyleSheet("padding: 5px; background: #e8f4f8;")
        perspective_info.setWordWrap(True)
        perspective_layout.addWidget(perspective_info)
        
        # Scale X (horizontal stretch)
        perspective_layout.addWidget(QLabel("Horizontal Stretch"))
        self.scale_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_x_slider.setRange(20, 300)
        self.scale_x_slider.setValue(100)
        self.scale_x_slider.valueChanged.connect(self.on_scale_x_changed)
        perspective_layout.addWidget(self.scale_x_slider)
        self.scale_x_label = QLabel("100%")
        perspective_layout.addWidget(self.scale_x_label)
        
        # Scale Y (vertical stretch)
        perspective_layout.addWidget(QLabel("Vertical Stretch"))
        self.scale_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_y_slider.setRange(20, 300)
        self.scale_y_slider.setValue(100)
        self.scale_y_slider.valueChanged.connect(self.on_scale_y_changed)
        perspective_layout.addWidget(self.scale_y_slider)
        self.scale_y_label = QLabel("100%")
        perspective_layout.addWidget(self.scale_y_label)
        
        # Rotation
        perspective_layout.addWidget(QLabel("Rotation (degrees)"))
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        perspective_layout.addWidget(self.rotation_slider)
        self.rotation_label = QLabel("0°")
        perspective_layout.addWidget(self.rotation_label)
        
        # Skew X (horizontal shear for angled view)
        perspective_layout.addWidget(QLabel("Horizontal Skew (for top/bottom view)"))
        self.skew_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.skew_x_slider.setRange(-100, 100)
        self.skew_x_slider.setValue(0)
        self.skew_x_slider.valueChanged.connect(self.on_skew_x_changed)
        perspective_layout.addWidget(self.skew_x_slider)
        self.skew_x_label = QLabel("0")
        perspective_layout.addWidget(self.skew_x_label)
        
        # Skew Y (vertical shear for angled view)
        perspective_layout.addWidget(QLabel("Vertical Skew (for side view)"))
        self.skew_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.skew_y_slider.setRange(-100, 100)
        self.skew_y_slider.setValue(0)
        self.skew_y_slider.valueChanged.connect(self.on_skew_y_changed)
        perspective_layout.addWidget(self.skew_y_slider)
        self.skew_y_label = QLabel("0")
        perspective_layout.addWidget(self.skew_y_label)
        
        perspective_group.setLayout(perspective_layout)
        controls_layout.addWidget(perspective_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        back_btn = QPushButton("Back to Menu")
        back_btn.setMinimumHeight(60)
        back_btn.clicked.connect(self.go_back)
        button_layout.addWidget(back_btn)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.setMinimumHeight(60)
        reset_btn.clicked.connect(self.reset_calibration)
        button_layout.addWidget(reset_btn)
        
        save_btn = QPushButton("Save Calibration")
        save_btn.setMinimumHeight(60)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_calibration)
        button_layout.addWidget(save_btn)
        
        controls_layout.addLayout(button_layout)
        
        # Wrap controls in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(controls_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def go_back(self) -> None:
        """Navigate back to start screen."""
        self.app.show_screen("start")
    
    def update_ui_from_calibration(self) -> None:
        """Update UI controls with values from loaded calibration."""
        if self.calibration is None:
            return
        
        # Block signals to prevent auto-save while loading
        self.bull_eye_slider.blockSignals(True)
        self.bull_slider.blockSignals(True)
        self.triple_inner_slider.blockSignals(True)
        self.triple_outer_slider.blockSignals(True)
        self.double_inner_slider.blockSignals(True)
        self.double_outer_slider.blockSignals(True)
        self.scale_x_slider.blockSignals(True)
        self.scale_y_slider.blockSignals(True)
        self.rotation_slider.blockSignals(True)
        self.skew_x_slider.blockSignals(True)
        self.skew_y_slider.blockSignals(True)
        self.triple_scale_slider.blockSignals(True)
        self.double_scale_slider.blockSignals(True)
        
        # Update center label
        self.center_label.setText(f"X: {int(self.calibration.center_x)}, Y: {int(self.calibration.center_y)}")
        
        # Update ring radius sliders
        self.bull_eye_slider.setValue(int(self.calibration.bull_eye_radius))
        self.bull_eye_label.setText(f"{int(self.calibration.bull_eye_radius)} px")
        
        self.bull_slider.setValue(int(self.calibration.bull_radius))
        self.bull_label.setText(f"{int(self.calibration.bull_radius)} px")
        
        self.triple_inner_slider.setValue(int(self.calibration.triple_inner_radius))
        self.triple_inner_label.setText(f"{int(self.calibration.triple_inner_radius)} px")
        
        self.triple_outer_slider.setValue(int(self.calibration.triple_outer_radius))
        self.triple_outer_label.setText(f"{int(self.calibration.triple_outer_radius)} px")
        
        self.double_inner_slider.setValue(int(self.calibration.double_inner_radius))
        self.double_inner_label.setText(f"{int(self.calibration.double_inner_radius)} px")
        
        self.double_outer_slider.setValue(int(self.calibration.double_outer_radius))
        self.double_outer_label.setText(f"{int(self.calibration.double_outer_radius)} px")
        
        # Update perspective sliders
        self.scale_x = self.calibration.scale_x
        self.scale_x_slider.setValue(int(self.calibration.scale_x * 100))
        self.scale_x_label.setText(f"{int(self.calibration.scale_x * 100)}%")
        
        self.scale_y = self.calibration.scale_y
        self.scale_y_slider.setValue(int(self.calibration.scale_y * 100))
        self.scale_y_label.setText(f"{int(self.calibration.scale_y * 100)}%")
        
        self.rotation = self.calibration.rotation
        self.rotation_slider.setValue(int(self.calibration.rotation))
        self.rotation_label.setText(f"{int(self.calibration.rotation)}°")
        
        self.skew_x = self.calibration.skew_x
        self.skew_x_slider.setValue(int(self.calibration.skew_x * 100))
        self.skew_x_label.setText(f"{int(self.calibration.skew_x * 100)}")
        
        self.skew_y = self.calibration.skew_y
        self.skew_y_slider.setValue(int(self.calibration.skew_y * 100))
        self.skew_y_label.setText(f"{int(self.calibration.skew_y * 100)}")
        
        self.triple_scale = self.calibration.triple_scale
        self.triple_scale_slider.setValue(int(self.calibration.triple_scale * 100))
        self.triple_scale_label.setText(f"{int(self.calibration.triple_scale * 100)}%")
        
        self.double_scale = self.calibration.double_scale
        self.double_scale_slider.setValue(int(self.calibration.double_scale * 100))
        self.double_scale_label.setText(f"{int(self.calibration.double_scale * 100)}%")
        
        # Unblock signals
        self.bull_eye_slider.blockSignals(False)
        self.bull_slider.blockSignals(False)
        self.triple_inner_slider.blockSignals(False)
        self.triple_outer_slider.blockSignals(False)
        self.double_inner_slider.blockSignals(False)
        self.double_outer_slider.blockSignals(False)
        self.scale_x_slider.blockSignals(False)
        self.scale_y_slider.blockSignals(False)
        self.rotation_slider.blockSignals(False)
        self.skew_x_slider.blockSignals(False)
        self.skew_y_slider.blockSignals(False)
        self.triple_scale_slider.blockSignals(False)
        self.double_scale_slider.blockSignals(False)
        
        # Update display
        self.update_image_display()
    
    def enable_center_selection(self) -> None:
        """Enable center point selection mode."""
        self.selecting_center = True
        self.center_btn.setText("Click on Bull's Eye...")
        self.center_btn.setStyleSheet("background-color: #FFC107; color: black;")
    
    def capture_reference_image(self) -> None:
        """Capture and save reference image for dart detection."""
        # Ensure camera is running
        if not self.app.start_camera():
            QMessageBox.critical(
                self,
                "Camera Error",
                "Could not access camera. Please check camera connection."
            )
            return
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Capture Reference Image",
            "Make sure the dartboard is COMPLETELY EMPTY (no darts!)\n\n"
            "The camera will capture in 3 seconds.\n"
            "Ready to capture?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Visual countdown
        from PySide6.QtCore import QTimer
        import time
        
        self.capture_ref_btn.setText("Capturing in 3...")
        QTimer.singleShot(1000, lambda: self.capture_ref_btn.setText("Capturing in 2..."))
        QTimer.singleShot(2000, lambda: self.capture_ref_btn.setText("Capturing in 1..."))
        QTimer.singleShot(3000, self._perform_capture)
    
    def _perform_capture(self) -> None:
        """Perform the actual capture and save."""
        import time
        from pathlib import Path
        
        # Wait for camera stabilization
        time.sleep(0.5)
        
        # Capture multiple frames and select the best one
        frames = []
        for i in range(10):
            stereo = self.app.camera.capture_stereo()
            if stereo is not None:
                frames.append(stereo.left)
            time.sleep(0.1)
        
        if not frames:
            QMessageBox.critical(
                self,
                "Capture Failed",
                "Could not capture frames from camera."
            )
            self.capture_ref_btn.setText("Capture Reference Image")
            return
        
        # Use middle frame (most stable)
        reference_frame = frames[len(frames) // 2]
        
        # Save to config directory
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        reference_path = config_dir / "reference_board.jpg"
        
        # Save image and update dart detector
        if hasattr(self.app, 'detector'):
            self.app.detector.set_reference_image(reference_frame)
            self.app.detector.save_reference_to_file(str(reference_path))
        else:
            # Fallback if detector not initialized
            cv2.imwrite(str(reference_path), reference_frame)
        
        # Show success message
        QMessageBox.information(
            self,
            "Reference Captured",
            f"Reference image saved successfully!\n\n"
            f"Saved to: {reference_path}\n\n"
            f"You can now use this for dart detection optimization."
        )
        
        # Reset button
        self.capture_ref_btn.setText("Capture Reference Image")
        
        # Update display with new reference
        self.camera_image = reference_frame
        self.update_image_display()
    
    def capture_test_image(self) -> None:
        """Capture test image with ground truth data for parameter optimization."""
        # Ensure camera is running
        if not self.app.start_camera():
            QMessageBox.critical(
                self,
                "Camera Error",
                "Could not access camera. Please check camera connection."
            )
            return
        
        # Show instructions
        reply = QMessageBox.question(
            self,
            "Capture Test Image",
            "Instructions:\n\n"
            "1. Throw a dart at the board\n"
            "2. Click Yes to capture\n"
            "3. Enter the dart's position manually\n\n"
            "This creates test data for automatic parameter optimization.\n\n"
            "Ready to capture?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Capture frame with stabilization delay
        import time
        time.sleep(0.5)
        
        stereo = self.app.camera.capture_stereo()
        frame = stereo.left if stereo is not None else None
        if frame is None:
            QMessageBox.critical(self, "Capture Failed", "Could not capture frame.")
            return
        
        # Show ground truth dialog
        dialog = GroundTruthDialog(frame, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            ground_truth = dialog.get_ground_truth()
            self._save_test_image(frame, ground_truth)
    
    def _save_test_image(self, frame: np.ndarray, ground_truth: dict) -> None:
        """Save test image with ground truth data."""
        # Create test_images directory
        test_dir = Path("config/test_images")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"test_{timestamp}.jpg"
        image_path = test_dir / image_filename
        
        # Save image
        cv2.imwrite(str(image_path), frame)
        
        # Load or create test data JSON
        json_path = Path("config/test_images.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = {"test_images": []}
        
        # Add new test image
        test_entry = {
            "image_path": str(image_path),
            "dart_x": ground_truth["x"],
            "dart_y": ground_truth["y"],
            "segment": ground_truth["segment"],
            "score": ground_truth["score"],
            "description": ground_truth["description"]
        }
        test_data["test_images"].append(test_entry)
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        QMessageBox.information(
            self,
            "Test Image Saved",
            f"Test image saved successfully!\n\n"
            f"Image: {image_path}\n"
            f"Position: ({ground_truth['x']}, {ground_truth['y']})\n"
            f"Segment: {ground_truth['segment']}\n"
            f"Score: {ground_truth['score']}\n\n"
            f"Total test images: {len(test_data['test_images'])}\n\n"
            f"Use 'python scripts/optimize_parameters.py' to find optimal parameters!"
        )
    
    def _initialize_control_points(self):
        """Initialize control points at key ring/segment positions."""
        if self.calibration is None:
            return
        
        self.control_points = []
        
        # Key angles: 20 (top), 3 (right), 6 (bottom), 11 (left) = 0°, 90°, 180°, 270°
        # Dartboard angles: top=90°, right=0° (or 18°), bottom=270°, left=180° (or 162°)
        key_angles = [90, 18, 270, 162]  # Top, Right, Bottom, Left in dartboard coordinates
        
        rings = [
            ("bull_eye", self.calibration.bull_eye_radius),
            ("bull", self.calibration.bull_radius),
            ("triple_inner", self.calibration.triple_inner_radius),
            ("triple_outer", self.calibration.triple_outer_radius),
            ("double_inner", self.calibration.double_inner_radius),
            ("double_outer", self.calibration.double_outer_radius)
        ]
        
        # Create control points for each ring at key angles
        for ring_name, radius in rings:
            for angle in key_angles:
                # Calculate point position (without perspective transform for now)
                angle_rad = np.radians(angle)
                px = self.calibration.center_x + radius * np.cos(angle_rad)
                py = self.calibration.center_y + radius * np.sin(angle_rad)
                
                self.control_points.append({
                    "ring": ring_name,
                    "angle": angle,
                    "x": px,
                    "y": py
                })
    
    def toggle_drag_mode(self):
        """Toggle drag mode on/off."""
        self.drag_mode_enabled = self.drag_mode_btn.isChecked()
        
        if self.drag_mode_enabled:
            self.drag_mode_btn.setText("✋ Drag Mode Active")
            self.drag_mode_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self._initialize_control_points()  # Refresh points
            self._update_calibration_from_points()  # Calculate initial ellipses
            # Disable sliders in drag mode
            self.bull_eye_slider.setEnabled(False)
            self.bull_slider.setEnabled(False)
            self.triple_inner_slider.setEnabled(False)
            self.triple_outer_slider.setEnabled(False)
            self.double_inner_slider.setEnabled(False)
            self.double_outer_slider.setEnabled(False)
            self.scale_x_slider.setEnabled(False)
            self.scale_y_slider.setEnabled(False)
            self.rotation_slider.setEnabled(False)
            self.skew_x_slider.setEnabled(False)
            self.skew_y_slider.setEnabled(False)
            self.triple_scale_slider.setEnabled(False)
            self.double_scale_slider.setEnabled(False)
        else:
            self.drag_mode_btn.setText("🎯 Enable Drag Mode")
            self.drag_mode_btn.setStyleSheet("background-color: #9E9E9E; color: white; font-weight: bold;")
            self.dragging_point = None
            self.ring_ellipses = {}  # Clear ellipses
            # Re-enable sliders
            self.bull_eye_slider.setEnabled(True)
            self.bull_slider.setEnabled(True)
            self.triple_inner_slider.setEnabled(True)
            self.triple_outer_slider.setEnabled(True)
            self.double_inner_slider.setEnabled(True)
            self.double_outer_slider.setEnabled(True)
            self.scale_x_slider.setEnabled(True)
            self.scale_y_slider.setEnabled(True)
            self.rotation_slider.setEnabled(True)
            self.skew_x_slider.setEnabled(True)
            self.skew_y_slider.setEnabled(True)
            self.triple_scale_slider.setEnabled(True)
            self.double_scale_slider.setEnabled(True)
        
        self.update_image_display()
    
    def _display_to_original_coords(self, display_x: int, display_y: int) -> tuple:
        """Convert display coordinates to original image coordinates."""
        if self.camera_image is None or self.displayed_pixmap is None:
            return (0, 0)
        
        # Get displayed image size
        display_width = self.displayed_pixmap.width()
        display_height = self.displayed_pixmap.height()
        
        # Get original image size
        original_height, original_width = self.camera_image.shape[:2]
        
        # Calculate scale factors
        scale_x = original_width / display_width
        scale_y = original_height / display_height
        
        # Map coordinates
        original_x = int(display_x * scale_x)
        original_y = int(display_y * scale_y)
        
        # Clamp to image bounds
        original_x = max(0, min(original_x, original_width - 1))
        original_y = max(0, min(original_y, original_height - 1))
        
        return (original_x, original_y)
    
    def _original_to_display_coords(self, original_x: float, original_y: float) -> tuple:
        """Convert original image coordinates to display coordinates."""
        if self.camera_image is None or self.displayed_pixmap is None:
            return (0, 0)
        
        # Get displayed image size
        display_width = self.displayed_pixmap.width()
        display_height = self.displayed_pixmap.height()
        
        # Get original image size
        original_height, original_width = self.camera_image.shape[:2]
        
        # Calculate scale factors
        scale_x = display_width / original_width
        scale_y = display_height / original_height
        
        # Map coordinates
        display_x = int(original_x * scale_x)
        display_y = int(original_y * scale_y)
        
        return (display_x, display_y)
    
    def _find_nearest_control_point(self, x: int, y: int, threshold: int = 15) -> Optional[int]:
        """Find control point near given position (in display coords)."""
        nearest_idx = None
        nearest_dist = threshold
        
        for idx, point in enumerate(self.control_points):
            disp_x, disp_y = self._original_to_display_coords(point["x"], point["y"])
            dist = np.sqrt((x - disp_x)**2 + (y - disp_y)**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        
        return nearest_idx
    
    def on_image_press(self, event) -> None:
        """Handle mouse press on image."""
        if self.camera_image is None or self.displayed_pixmap is None:
            return
        
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Handle center selection mode
        if self.selecting_center:
            original_x, original_y = self._display_to_original_coords(click_x, click_y)
            
            # Update calibration
            self.calibration.center_x = original_x
            self.calibration.center_y = original_y
            
            # Auto-save calibration
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
            
            # Update UI
            self.center_label.setText(f"X: {original_x}, Y: {original_y}")
            self.center_btn.setText("Set Center")
            self.center_btn.setStyleSheet("")
            self.selecting_center = False
            
            # Refresh control points and display
            self._initialize_control_points()
            self.update_image_display()
            return
        
        # Handle drag mode
        if self.drag_mode_enabled:
            point_idx = self._find_nearest_control_point(click_x, click_y)
            if point_idx is not None:
                self.dragging_point = point_idx
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def on_image_move(self, event) -> None:
        """Handle mouse move on image."""
        if not self.drag_mode_enabled or self.dragging_point is None:
            return
        
        if self.camera_image is None or self.displayed_pixmap is None:
            return
        
        # Get mouse position
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()
        
        # Convert to original coordinates
        original_x, original_y = self._display_to_original_coords(mouse_x, mouse_y)
        
        # Update control point position
        self.control_points[self.dragging_point]["x"] = original_x
        self.control_points[self.dragging_point]["y"] = original_y
        
        # Update calibration from control points
        self._update_calibration_from_points()
        
        # Refresh display
        self.update_image_display()
    
    def on_image_release(self, event) -> None:
        """Handle mouse release on image."""
        if self.dragging_point is not None:
            self.dragging_point = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            # Save calibration after drag complete
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
    
    def _fit_ellipse_from_points(self, points):
        """Fit ellipse parameters from control points.
        Returns: (center_x, center_y, width, height, angle)
        """
        if len(points) < 3:
            return None
        
        # Extract point coordinates
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        
        # Fit ellipse using OpenCV
        if len(coords) >= 5:
            ellipse = cv2.fitEllipse(coords)
            # ellipse = ((cx, cy), (width, height), angle)
            return ellipse
        else:
            # Not enough points for fitEllipse, use bounding ellipse
            center = coords.mean(axis=0)
            # Calculate distances to estimate axes
            dists = np.sqrt(((coords - center)**2).sum(axis=1))
            avg_radius = dists.mean()
            return ((center[0], center[1]), (avg_radius*2, avg_radius*2), 0)
    
    def _update_calibration_from_points(self):
        """Update calibration parameters based on control point positions."""
        if self.calibration is None or len(self.control_points) == 0:
            return
        
        # Group points by ring
        rings_points = {}
        for point in self.control_points:
            ring = point["ring"]
            if ring not in rings_points:
                rings_points[ring] = []
            rings_points[ring].append(point)
        
        # Store ellipse parameters for each ring (for drawing)
        self.ring_ellipses = {}
        
        center_x = self.calibration.center_x
        center_y = self.calibration.center_y
        
        for ring_name, points in rings_points.items():
            # Fit ellipse to points
            ellipse = self._fit_ellipse_from_points(points)
            if ellipse:
                self.ring_ellipses[ring_name] = ellipse
                
                # Calculate average radius for calibration (for compatibility)
                (cx, cy), (w, h), angle = ellipse
                avg_radius = (w + h) / 4.0  # Average of semi-axes
                
                # Update calibration with average radius
                if ring_name == "bull_eye":
                    self.calibration.bull_eye_radius = avg_radius
                    self.bull_eye_slider.blockSignals(True)
                    self.bull_eye_slider.setValue(int(avg_radius))
                    self.bull_eye_label.setText(f"{int(avg_radius)} px")
                    self.bull_eye_slider.blockSignals(False)
                elif ring_name == "bull":
                    self.calibration.bull_radius = avg_radius
                    self.bull_slider.blockSignals(True)
                    self.bull_slider.setValue(int(avg_radius))
                    self.bull_label.setText(f"{int(avg_radius)} px")
                    self.bull_slider.blockSignals(False)
                elif ring_name == "triple_inner":
                    self.calibration.triple_inner_radius = avg_radius
                    self.triple_inner_slider.blockSignals(True)
                    self.triple_inner_slider.setValue(int(avg_radius / self.triple_scale))
                    self.triple_inner_label.setText(f"{int(avg_radius / self.triple_scale)} px")
                    self.triple_inner_slider.blockSignals(False)
                elif ring_name == "triple_outer":
                    self.calibration.triple_outer_radius = avg_radius
                    self.triple_outer_slider.blockSignals(True)
                    self.triple_outer_slider.setValue(int(avg_radius / self.triple_scale))
                    self.triple_outer_label.setText(f"{int(avg_radius / self.triple_scale)} px")
                    self.triple_outer_slider.blockSignals(False)
                elif ring_name == "double_inner":
                    self.calibration.double_inner_radius = avg_radius
                    self.double_inner_slider.blockSignals(True)
                    self.double_inner_slider.setValue(int(avg_radius / self.double_scale))
                    self.double_inner_label.setText(f"{int(avg_radius / self.double_scale)} px")
                    self.double_inner_slider.blockSignals(False)
                elif ring_name == "double_outer":
                    self.calibration.double_outer_radius = avg_radius
                    self.double_outer_slider.blockSignals(True)
                    self.double_outer_slider.setValue(int(avg_radius / self.double_scale))
                    self.double_outer_label.setText(f"{int(avg_radius / self.double_scale)} px")
                    self.double_outer_slider.blockSignals(False)
    
    def on_bull_eye_changed(self, value: int) -> None:
        """Handle bull's eye radius slider change."""
        if self.calibration is None:
            return
        self.calibration.bull_eye_radius = float(value)
        self.bull_eye_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_bull_changed(self, value: int) -> None:
        """Handle bull radius slider change."""
        if self.calibration is None:
            return
        self.calibration.bull_radius = float(value)
        self.bull_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_triple_inner_changed(self, value: int) -> None:
        """Handle triple inner radius slider change."""
        if self.calibration is None:
            return
        self.calibration.triple_inner_radius = float(value * self.triple_scale)
        self.triple_inner_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_triple_outer_changed(self, value: int) -> None:
        """Handle triple outer radius slider change."""
        if self.calibration is None:
            return
        self.calibration.triple_outer_radius = float(value * self.triple_scale)
        self.triple_outer_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_double_inner_changed(self, value: int) -> None:
        """Handle double inner radius slider change."""
        if self.calibration is None:
            return
        self.calibration.double_inner_radius = float(value * self.double_scale)
        self.double_inner_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_double_outer_changed(self, value: int) -> None:
        """Handle double outer radius slider change."""
        if self.calibration is None:
            return
        self.calibration.double_outer_radius = float(value * self.double_scale)
        self.double_outer_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.set_calibration(self.calibration)
        self.app.save_calibration()
    
    def on_scale_x_changed(self, value: int) -> None:
        """Handle horizontal scale slider change."""
        self.scale_x = value / 100.0
        self.scale_x_label.setText(f"{value}%")
        if self.calibration is not None:
            self.calibration.scale_x = self.scale_x
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()
    
    def on_scale_y_changed(self, value: int) -> None:
        """Handle vertical scale slider change."""
        self.scale_y = value / 100.0
        self.scale_y_label.setText(f"{value}%")
        if self.calibration is not None:
            self.calibration.scale_y = self.scale_y
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()
    
    def on_rotation_changed(self, value: int) -> None:
        """Handle rotation slider change."""
        self.rotation = float(value)
        self.rotation_label.setText(f"{value}°")
        if self.calibration is not None:
            self.calibration.rotation = self.rotation
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()
    
    def on_skew_x_changed(self, value: int) -> None:
        """Handle horizontal skew slider change."""
        self.skew_x = value / 100.0
        self.skew_x_label.setText(f"{value}")
        if self.calibration is not None:
            self.calibration.skew_x = self.skew_x
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()
    
    def on_skew_y_changed(self, value: int) -> None:
        """Handle vertical skew slider change."""
        self.skew_y = value / 100.0
        self.skew_y_label.setText(f"{value}")
        if self.calibration is not None:
            self.calibration.skew_y = self.skew_y
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()    
    def on_triple_scale_changed(self, value: int):
        """Handle triple ring scale slider change."""
        self.triple_scale = value / 100.0
        self.triple_scale_label.setText(f"{value}%")
        # Update calibration with scaled values
        if self.calibration is not None:
            self.calibration.triple_scale = self.triple_scale
            self.calibration.triple_inner_radius = float(self.triple_inner_slider.value() * self.triple_scale)
            self.calibration.triple_outer_radius = float(self.triple_outer_slider.value() * self.triple_scale)
            # Auto-save
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()
    
    def on_double_scale_changed(self, value: int):
        """Handle double ring scale slider change."""
        self.double_scale = value / 100.0
        self.double_scale_label.setText(f"{value}%")
        # Update calibration with scaled values
        if self.calibration is not None:
            self.calibration.double_scale = self.double_scale
            self.calibration.double_inner_radius = float(self.double_inner_slider.value() * self.double_scale)
            self.calibration.double_outer_radius = float(self.double_outer_slider.value() * self.double_scale)
            # Auto-save
            self.app.mapper.set_calibration(self.calibration)
            self.app.save_calibration()
        self.update_image_display()    
    def update_image_display(self) -> None:
        """Update image with calibration overlay."""
        if self.camera_image is None or self.calibration is None:
            # Show placeholder
            self.image_label.setText("No camera image available\nPlease connect camera")
            return
        
        # Update dimensions
        self.image_height, self.image_width = self.camera_image.shape[:2]
        
        # Copy image
        display_image = self.camera_image.copy()
        
        # Draw calibration overlay
        center = (self.calibration.center_x, self.calibration.center_y)
        
        # Get ring radii from calibration and apply global scaling
        bull_eye = int(self.calibration.bull_eye_radius)
        bull = int(self.calibration.bull_radius)
        triple_inner = int(self.calibration.triple_inner_radius)
        triple_outer = int(self.calibration.triple_outer_radius)
        double_inner = int(self.calibration.double_inner_radius)
        double_outer = int(self.calibration.double_outer_radius)
        
        # Draw rings - use fitted ellipses if in drag mode, otherwise use transformed circles
        if self.drag_mode_enabled and hasattr(self, 'ring_ellipses') and self.ring_ellipses:
            # Draw fitted ellipses from control points
            ring_configs = [
                ("bull_eye", (0, 255, 0), 2),
                ("bull", (0, 255, 255), 2),
                ("triple_inner", (255, 255, 0), 2),
                ("triple_outer", (255, 255, 0), 2),
                ("double_inner", (255, 0, 0), 2),
                ("double_outer", (255, 0, 0), 2)
            ]
            
            for ring_name, color, thickness in ring_configs:
                if ring_name in self.ring_ellipses:
                    ellipse = self.ring_ellipses[ring_name]
                    (cx, cy), (w, h), angle = ellipse
                    center_pt = (int(cx), int(cy))
                    axes = (int(w/2), int(h/2))
                    cv2.ellipse(display_image, center_pt, axes, angle, 0, 360, color, thickness)
        else:
            # Draw rings as ellipses (applying perspective transformation with skew)
            def draw_ellipse_ring(radius, color, thickness=2):
                # Apply skew by adjusting the center for each radius
                skewed_center = (
                    int(center[0] + radius * self.skew_x),
                    int(center[1] + radius * self.skew_y)
                )
                axes = (int(radius * self.scale_x), int(radius * self.scale_y))
                cv2.ellipse(display_image, skewed_center, axes, self.rotation, 0, 360, color, thickness)
            
            # Draw all rings
            draw_ellipse_ring(bull_eye, (0, 255, 0), 2)      # Green
            draw_ellipse_ring(bull, (0, 255, 255), 2)        # Cyan
            draw_ellipse_ring(triple_inner, (255, 255, 0), 2) # Yellow
            draw_ellipse_ring(triple_outer, (255, 255, 0), 2) # Yellow
            draw_ellipse_ring(double_inner, (255, 0, 0), 2)  # Blue
            draw_ellipse_ring(double_outer, (255, 0, 0), 2)  # Blue
        
        # Draw center cross
        cv2.drawMarker(display_image, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Draw control points if drag mode is enabled
        if self.drag_mode_enabled and len(self.control_points) > 0:
            for idx, point in enumerate(self.control_points):
                px = int(point["x"])
                py = int(point["y"])
                
                # Different colors for different rings
                if point["ring"] in ["bull_eye", "bull"]:
                    color = (0, 255, 0)  # Green
                elif point["ring"] in ["triple_inner", "triple_outer"]:
                    color = (255, 255, 0)  # Yellow
                else:  # double rings
                    color = (255, 0, 0)  # Blue
                
                # Highlight dragging point
                if idx == self.dragging_point:
                    cv2.circle(display_image, (px, py), 12, (0, 255, 255), -1)  # Filled cyan
                    cv2.circle(display_image, (px, py), 14, (255, 255, 255), 2)  # White border
                else:
                    cv2.circle(display_image, (px, py), 8, color, -1)  # Filled circle
                    cv2.circle(display_image, (px, py), 10, (255, 255, 255), 2)  # White border
        
        # Draw all 20 segment lines (also transformed)
        segment_angles = [9, 27, 45, 63, 81, 99, 117, 135, 153, 171, 189, 207, 225, 243, 261, 279, 297, 315, 333, 351]
        max_radius = double_outer * 1.1  # Slightly beyond outer ring
        
        for angle_deg in segment_angles:
            # Adjust angle for rotation
            adjusted_angle = angle_deg + self.rotation
            angle_rad = np.radians(adjusted_angle)
            # Apply scale and skew
            end_x = int(center[0] + max_radius * self.scale_x * np.cos(angle_rad) + max_radius * self.skew_x)
            end_y = int(center[1] + max_radius * self.scale_y * np.sin(angle_rad) + max_radius * self.skew_y)
            cv2.line(display_image, center, (end_x, end_y), (128, 128, 128), 1)  # Gray segment lines
        
        # Convert to QPixmap
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label (max 800x600)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.displayed_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)
    
    def reset_calibration(self) -> None:
        """Reset to default calibration."""
        estimated_radius = min(self.image_width, self.image_height) * 0.4
        self.calibration = create_default_calibration(
            self.image_width,
            self.image_height,
            board_radius_pixels=estimated_radius
        )
        
        # Update UI controls
        self.center_label.setText(
            f"X: {self.calibration.center_x}, Y: {self.calibration.center_y}"
        )
        self.bull_eye_slider.setValue(int(self.calibration.bull_eye_radius))
        self.bull_slider.setValue(int(self.calibration.bull_radius))
        self.triple_inner_slider.setValue(int(self.calibration.triple_inner_radius))
        self.triple_outer_slider.setValue(int(self.calibration.triple_outer_radius))
        self.double_inner_slider.setValue(int(self.calibration.double_inner_radius))
        self.double_outer_slider.setValue(int(self.calibration.double_outer_radius))
        
        # Reset perspective/scale sliders
        self.scale_x_slider.setValue(100)
        self.scale_y_slider.setValue(100)
        self.rotation_slider.setValue(0)
        self.skew_x_slider.setValue(0)
        self.skew_y_slider.setValue(0)
        self.triple_scale_slider.setValue(100)
        self.double_scale_slider.setValue(100)
        
        self.update_image_display()
    
    def save_calibration(self) -> None:
        """Save and emit calibration."""
        # Apply to main app's board mapper
        self.app.mapper.set_calibration(self.calibration)
        
        # Save to file
        if self.app.save_calibration():
            QMessageBox.information(
                self,
                "Calibration Saved",
                "Calibration has been saved successfully."
            )
        
        # Emit signal
        self.calibration_complete.emit(self.calibration)
        
        # Return to start screen
        self.app.show_screen("start")
    
    def cancel_calibration(self) -> None:
        """Cancel calibration and return."""
        self.app.show_screen("start")
