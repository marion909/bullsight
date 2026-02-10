"""
Simple perspective-corrected calibration UI.

Fast, automatic calibration using 4-point homography.
Works with extreme camera angles.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QGroupBox, QSlider
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QCursor
import cv2
import numpy as np
import logging
from typing import Optional, TYPE_CHECKING, List, Tuple

from src.calibration.perspective_calibrator import PerspectiveCalibrator
from src.calibration.board_mapper import CalibrationData

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.main import BullSightApp


class PerspectiveCalibrationScreen(QWidget):
    """
    Perspective-corrected calibration screen.
    
    User flow:
    1. Auto-detect board (optional)
    2. Click 4 reference points: 20 (top), 6 (right), 3 (bottom), 11 (left)
    3. System calculates homography
    4. Overlay shows perspective-corrected rings
    5. Save calibration
    """
    
    calibration_complete = Signal(CalibrationData)
    
    def __init__(self, app: 'BullSightApp'):
        super().__init__()
        self.app = app
        
        # Stereo camera images
        self.left_image: Optional[np.ndarray] = None
        self.right_image: Optional[np.ndarray] = None
        
        # Active camera (LEFT or RIGHT)
        self.active_camera = "LEFT"
        
        # Separate calibrators for left and right
        self.left_calibrator = PerspectiveCalibrator()
        self.right_calibrator = PerspectiveCalibrator()
        
        # UI state
        self.selecting_points = False
        self.selected_points: List[Tuple[int, int]] = []
        self.point_names = ["20 (Top)", "6 (Right)", "3 (Bottom)", "11 (Left)"]
        
        # Display scaling factors
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        
        # Displayed pixmaps
        self.displayed_left_pixmap: Optional[QPixmap] = None
        self.displayed_right_pixmap: Optional[QPixmap] = None
        
        # Camera timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        
        self.setup_ui()
    
    def showEvent(self, event):
        """Start camera when shown."""
        super().showEvent(event)
        if self.app.start_camera():
            self.camera_timer.start(100)
    
    def hideEvent(self, event):
        """Stop camera when hidden."""
        super().hideEvent(event)
        self.camera_timer.stop()
    
    def update_camera_feed(self):
        """Update camera feed."""
        try:
            if self.app.camera and self.app.camera.is_started:
                stereo = self.app.camera.capture_stereo()
                if stereo is not None:
                    self.left_image = stereo.left
                    self.right_image = stereo.right
                    self.update_display()
        except:
            pass
    
    def setup_ui(self):
        """Setup UI."""
        layout = QHBoxLayout()
        
        # Left: Stereo Images
        left_layout = QVBoxLayout()
        
        title = QLabel("🎯 Stereo Perspective Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(title)
        
        self.status_label = QLabel("Select camera and click 'Auto-Detect' or 'Start Point Selection'")
        self.status_label.setStyleSheet("padding: 10px; background: #e3f2fd; font-size: 14px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Camera selection
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("Active Camera:"))
        
        self.left_camera_btn = QPushButton("📷 LEFT Camera")
        self.left_camera_btn.setCheckable(True)
        self.left_camera_btn.setChecked(True)
        self.left_camera_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.left_camera_btn.clicked.connect(lambda: self.select_camera("LEFT"))
        camera_select_layout.addWidget(self.left_camera_btn)
        
        self.right_camera_btn = QPushButton("📷 RIGHT Camera")
        self.right_camera_btn.setCheckable(True)
        self.right_camera_btn.setChecked(False)
        self.right_camera_btn.setStyleSheet("padding: 10px;")
        self.right_camera_btn.clicked.connect(lambda: self.select_camera("RIGHT"))
        camera_select_layout.addWidget(self.right_camera_btn)
        
        left_layout.addLayout(camera_select_layout)
        
        # Stereo image display
        images_layout = QHBoxLayout()
        
        # Left camera
        self.left_image_label = QLabel()
        self.left_image_label.setMinimumSize(400, 300)
        self.left_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_image_label.setMouseTracking(True)
        self.left_image_label.mousePressEvent = lambda e: self.on_image_click(e, "LEFT")
        self.left_image_label.setScaledContents(False)
        self.left_image_label.setStyleSheet("border: 3px solid #2196F3;")
        images_layout.addWidget(self.left_image_label)
        
        # Right camera
        self.right_image_label = QLabel()
        self.right_image_label.setMinimumSize(400, 300)
        self.right_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_image_label.setMouseTracking(True)
        self.right_image_label.mousePressEvent = lambda e: self.on_image_click(e, "RIGHT")
        self.right_image_label.setScaledContents(False)
        self.right_image_label.setStyleSheet("border: 3px solid #666;")
        images_layout.addWidget(self.right_image_label)
        
        left_layout.addLayout(images_layout)
        
        layout.addLayout(left_layout, stretch=3)
        
        # Right: Controls
        right_layout = QVBoxLayout()
        
        # Instructions
        instructions = QGroupBox("Instructions")
        inst_layout = QVBoxLayout()
        
        inst_text = QLabel(
            "<b>Quick Calibration:</b><br><br>"
            "1. <b>Auto-Detect</b>: Find dartboard automatically<br>"
            "2. <b>Select 4 Points</b>: Click on outer ring at:<br>"
            "   • Point 1: Number <b>20</b> (top)<br>"
            "   • Point 2: Number <b>6</b> (right)<br>"
            "   • Point 3: Number <b>3</b> (bottom)<br>"
            "   • Point 4: Number <b>11</b> (left)<br>"
            "3. Homography calculated automatically<br>"
            "4. <b>Save</b> when overlay looks correct"
        )
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)
        instructions.setLayout(inst_layout)
        right_layout.addWidget(instructions)
        
        # Auto-detect button
        self.detect_btn = QPushButton("🔍 Auto-Detect Board")
        self.detect_btn.setMinimumHeight(60)
        self.detect_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold;")
        self.detect_btn.clicked.connect(self.auto_detect_board)
        right_layout.addWidget(self.detect_btn)
        
        # Point selection button
        self.select_points_btn = QPushButton("📍 Start Point Selection")
        self.select_points_btn.setMinimumHeight(60)
        self.select_points_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold;")
        self.select_points_btn.setCheckable(True)
        self.select_points_btn.clicked.connect(self.toggle_point_selection)
        right_layout.addWidget(self.select_points_btn)
        
        # Point status
        self.point_status_label = QLabel("Points selected: 0/4")
        self.point_status_label.setStyleSheet("padding: 10px; background: #fff3cd; font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.point_status_label)
        
        # Rotation control
        rotation_group = QGroupBox("Rotation Adjustment")
        rotation_layout = QVBoxLayout()
        
        rotation_info = QLabel("Rotate the overlay to match your dartboard orientation")
        rotation_info.setStyleSheet("font-size: 11px; padding: 5px;")
        rotation_info.setWordWrap(True)
        rotation_layout.addWidget(rotation_info)
        
        rotation_slider_layout = QHBoxLayout()
        rotation_slider_layout.addWidget(QLabel("Rotation:"))
        
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        rotation_slider_layout.addWidget(self.rotation_slider)
        
        self.rotation_label = QLabel("0°")
        self.rotation_label.setMinimumWidth(50)
        rotation_slider_layout.addWidget(self.rotation_label)
        
        rotation_layout.addLayout(rotation_slider_layout)
        rotation_group.setLayout(rotation_layout)
        right_layout.addWidget(rotation_group)
        
        # Reset points
        self.reset_points_btn = QPushButton("🔄 Reset Points")
        self.reset_points_btn.setMinimumHeight(50)
        self.reset_points_btn.clicked.connect(self.reset_points)
        right_layout.addWidget(self.reset_points_btn)
        
        right_layout.addStretch()
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        back_btn = QPushButton("← Back")
        back_btn.setMinimumHeight(50)
        back_btn.clicked.connect(lambda: self.app.show_screen("start"))
        button_layout.addWidget(back_btn)
        
        save_btn = QPushButton("💾 Save Calibration")
        save_btn.setMinimumHeight(50)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_calibration)
        button_layout.addWidget(save_btn)
        
        right_layout.addLayout(button_layout)
        
        layout.addLayout(right_layout, stretch=1)
        self.setLayout(layout)
    
    def select_camera(self, camera: str):
        """Select active camera for calibration."""
        self.active_camera = camera
        
        # Update button styles
        if camera == "LEFT":
            self.left_camera_btn.setChecked(True)
            self.right_camera_btn.setChecked(False)
            self.left_camera_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
            self.right_camera_btn.setStyleSheet("padding: 10px;")
            self.left_image_label.setStyleSheet("border: 3px solid #2196F3;")
            self.right_image_label.setStyleSheet("border: 3px solid #666;")
        else:
            self.left_camera_btn.setChecked(False)
            self.right_camera_btn.setChecked(True)
            self.left_camera_btn.setStyleSheet("padding: 10px;")
            self.right_camera_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
            self.left_image_label.setStyleSheet("border: 3px solid #666;")
            self.right_image_label.setStyleSheet("border: 3px solid #2196F3;")
        
        # Reset points for newly selected camera
        self.reset_points()
        
        # Update rotation slider to match selected camera's calibrator
        calibrator = self.left_calibrator if camera == "LEFT" else self.right_calibrator
        self.rotation_slider.setValue(int(calibrator.rotation_offset))
        
        self.status_label.setText(f"Active: {camera} Camera - Click 'Auto-Detect' or 'Start Point Selection'")
        self.update_display()
    
    def get_active_image(self):
        """Get image for active camera."""
        return self.left_image if self.active_camera == "LEFT" else self.right_image
    
    def get_active_calibrator(self):
        """Get calibrator for active camera."""
        return self.left_calibrator if self.active_camera == "LEFT" else self.right_calibrator
    
    def auto_detect_board(self):
        """Auto-detect dartboard."""
        camera_image = self.get_active_image()
        if camera_image is None:
            QMessageBox.warning(self, "No Image", "Please wait for camera to start")
            return
        
        self.status_label.setText(f"🔍 Detecting dartboard in {self.active_camera} camera...")
        self.detect_btn.setEnabled(False)
        
        # Detect
        calibrator = self.get_active_calibrator()
        result = calibrator.auto_detect_board(camera_image)
        
        if result:
            cx, cy, radius = result
            self.status_label.setText(
                f"✅ Board detected in {self.active_camera} camera at ({cx}, {cy}) with radius {int(radius)}px\n"
                f"Now click 'Start Point Selection' and select 4 reference points"
            )
            self.status_label.setStyleSheet("padding: 10px; background: #c8e6c9; font-size: 14px;")
        else:
            self.status_label.setText(
                "❌ Could not detect board automatically\n"
                "Click 'Start Point Selection' to mark points manually"
            )
            self.status_label.setStyleSheet("padding: 10px; background: #ffcdd2; font-size: 14px;")
        
        self.detect_btn.setEnabled(True)
        self.update_display()
    
    def toggle_point_selection(self):
        """Toggle point selection mode."""
        self.selecting_points = self.select_points_btn.isChecked()
        
        if self.selecting_points:
            self.select_points_btn.setText("✋ Selecting Points... (Click on board)")
            self.select_points_btn.setStyleSheet("background-color: #FF9800; color: white; font-size: 14px; font-weight: bold;")
            self.status_label.setText(
                f"👆 Click on segment <b>{self.point_names[len(self.selected_points)]}</b> on the OUTER ring"
            )
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.select_points_btn.setText("📍 Start Point Selection")
            self.select_points_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; font-weight: bold;")
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def reset_points(self):
        """Reset selected points."""
        self.selected_points = []
        calibrator = self.get_active_calibrator()
        calibrator.homography_matrix = None
        calibrator.reference_points = []
        self.point_status_label.setText("Points selected: 0/4")
        self.selecting_points = False
        self.select_points_btn.setChecked(False)
        self.toggle_point_selection()
        self.status_label.setText(f"{self.active_camera} camera: Points reset. Click 'Start Point Selection' to begin")
        self.status_label.setStyleSheet("padding: 10px; background: #e3f2fd; font-size: 14px;")
        calibrator.rotation_offset = 0.0
        self.rotation_slider.setValue(0)
        self.update_display()
    
    def on_rotation_changed(self, value: int):
        """Handle rotation slider change."""
        calibrator = self.get_active_calibrator()
        calibrator.rotation_offset = float(value)
        self.rotation_label.setText(f"{value}°")
        self.update_display()
    
    def on_image_click(self, event, camera: str):
        """Handle image click."""
        # Only allow clicks on active camera
        if camera != self.active_camera:
            return
            
        if not self.selecting_points:
            return
        
        camera_image = self.get_active_image()
        image_label = self.left_image_label if camera == "LEFT" else self.right_image_label
        
        if camera_image is None or not hasattr(self, 'displayed_left_pixmap' if camera == "LEFT" else 'displayed_right_pixmap'):
            return
        
        if len(self.selected_points) >= 4:
            return
        
        # Get click position relative to label
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Get pixmap dimensions
        displayed_pixmap = self.displayed_left_pixmap if camera == "LEFT" else self.displayed_right_pixmap
        pixmap_width = displayed_pixmap.width()
        pixmap_height = displayed_pixmap.height()
        
        # Get label dimensions
        label_width = image_label.width()
        label_height = image_label.height()
        
        # Calculate offset (centered)
        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2
        
        # Convert to pixmap coordinates
        pixmap_x = click_x - offset_x
        pixmap_y = click_y - offset_y
        
        # Check bounds
        if pixmap_x < 0 or pixmap_x >= pixmap_width or pixmap_y < 0 or pixmap_y >= pixmap_height:
            logger.warning(f"Click outside image bounds: ({pixmap_x}, {pixmap_y})")
            return
        
        # Convert to original image coordinates using stored scale factors
        original_x = int(pixmap_x / self.display_scale_x)
        original_y = int(pixmap_y / self.display_scale_y)
        
        # Clamp to image bounds
        original_height, original_width = camera_image.shape[:2]
        original_x = max(0, min(original_x, original_width - 1))
        original_y = max(0, min(original_y, original_height - 1))
        
        # Add point
        self.selected_points.append((original_x, original_y))
        
        self.point_status_label.setText(f"Points selected: {len(self.selected_points)}/4")
        
        if len(self.selected_points) < 4:
            self.status_label.setText(
                f"✅ Point {len(self.selected_points)} saved\n"
                f"👆 Now click on segment <b>{self.point_names[len(self.selected_points)]}</b>"
            )
        else:
            # All 4 points selected - calculate homography
            self.calculate_homography()
            self.selecting_points = False
            self.select_points_btn.setChecked(False)
            self.toggle_point_selection()
        
        self.update_display()
    
    def calculate_homography(self):
        """Calculate homography from 4 selected points."""
        if len(self.selected_points) != 4:
            return
        
        calibrator = self.get_active_calibrator()
        
        # Get reference positions on dartboard (outer ring at 4 cardinal directions)
        known_positions = calibrator.generate_reference_points_for_4_segments()
        
        try:
            calibrator.set_reference_points(self.selected_points, known_positions)
            
            self.status_label.setText(
                f"✅ <b>{self.active_camera} Camera calibration successful!</b><br>"
                "Homography matrix calculated from 4 points<br>"
                "Check if overlay matches the dartboard, adjust rotation if needed, then click 'Save Calibration'"
            )
            self.status_label.setStyleSheet("padding: 10px; background: #c8e6c9; font-size: 14px;")
            
            logger.info(f"{self.active_camera} perspective calibration successful")
            
        except Exception as e:
            logger.error(f"Homography calculation failed for {self.active_camera}: {e}")
            QMessageBox.critical(
                self,
                "Calibration Error",
                f"Could not calculate homography for {self.active_camera}:\n{str(e)}\n\nPlease reset and try again."
            )
            self.reset_points()
    
    def update_display(self):
        """Update image display with overlay for both cameras."""
        # Update left camera
        self._update_camera_display("LEFT", self.left_image, self.left_image_label, self.left_calibrator)
        
        # Update right camera
        self._update_camera_display("RIGHT", self.right_image, self.right_image_label, self.right_calibrator)
    
    def _update_camera_display(self, camera: str, image: Optional[np.ndarray], 
                               label: QLabel, calibrator: PerspectiveCalibrator):
        """Update single camera display."""
        if image is None:
            label.setText(f"Waiting for {camera} camera...")
            return
        
        display_image = image.copy()
        
        # Draw detected board
        if calibrator.board_center_px and calibrator.board_radius_px:
            cx, cy = calibrator.board_center_px
            radius = int(calibrator.board_radius_px)
            cv2.circle(display_image, (cx, cy), radius, (255, 0, 255), 3)
            cv2.drawMarker(display_image, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
        
        # Draw selected points only for active camera
        if camera == self.active_camera:
            for i, (x, y) in enumerate(self.selected_points):
                cv2.circle(display_image, (x, y), 12, (0, 255, 255), -1)
                cv2.circle(display_image, (x, y), 15, (255, 255, 255), 3)
                cv2.putText(display_image, str(i+1), (x+20, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(display_image, self.point_names[i].split()[0], (x+20, y+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw overlay if calibrated
        if calibrator.homography_matrix is not None:
            display_image = calibrator.draw_overlay(
                display_image,
                show_segments=True,
                show_rings=True,
                show_reference_points=True,
                show_numbers=True  # Show segment numbers
            )
        
        # Convert to QPixmap
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        
        # Store pixmap and scale factors
        if camera == "LEFT":
            self.displayed_left_pixmap = scaled_pixmap
        else:
            self.displayed_right_pixmap = scaled_pixmap
        
        # Store scale factors for coordinate transformation (use active camera's scale)
        if camera == self.active_camera:
            self.display_scale_x = scaled_pixmap.width() / width
            self.display_scale_y = scaled_pixmap.height() / height
        
        label.setPixmap(scaled_pixmap)
    
    def save_calibration(self):
        """Save calibration for both cameras."""
        # Check if at least one camera is calibrated
        if self.left_calibrator.homography_matrix is None and self.right_calibrator.homography_matrix is None:
            QMessageBox.warning(
                self,
                "Not Calibrated",
                "Please complete the 4-point calibration for at least one camera first"
            )
            return
        
        try:
            # Save homography and reference points to JSON
            import json
            from pathlib import Path
            
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            
            saved_cameras = []
            
            # Save LEFT camera if calibrated
            if self.left_calibrator.homography_matrix is not None:
                calib_path = config_dir / "perspective_calibration_left.json"
                
                data = {
                    "camera": "LEFT",
                    "homography_matrix": self.left_calibrator.homography_matrix.tolist(),
                    "reference_points": self.left_calibrator.reference_points,
                    "board_center_px": self.left_calibrator.board_center_px,
                    "board_radius_px": self.left_calibrator.board_radius_px,
                    "rotation_offset": self.left_calibrator.rotation_offset
                }
                
                with open(calib_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"LEFT perspective calibration saved to {calib_path}")
                saved_cameras.append(f"LEFT Camera → {calib_path}")
            
            # Save RIGHT camera if calibrated
            if self.right_calibrator.homography_matrix is not None:
                calib_path = config_dir / "perspective_calibration_right.json"
                
                data = {
                    "camera": "RIGHT",
                    "homography_matrix": self.right_calibrator.homography_matrix.tolist(),
                    "reference_points": self.right_calibrator.reference_points,
                    "board_center_px": self.right_calibrator.board_center_px,
                    "board_radius_px": self.right_calibrator.board_radius_px,
                    "rotation_offset": self.right_calibrator.rotation_offset
                }
                
                with open(calib_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"RIGHT perspective calibration saved to {calib_path}")
                saved_cameras.append(f"RIGHT Camera → {calib_path}")
            
            QMessageBox.information(
                self,
                "Calibration Saved",
                "Stereo perspective calibration saved successfully!\n\n"
                + "\n".join(saved_cameras) + "\n\n"
                "The overlays will now be perspective-corrected.\n"
                "You can test them in the ML Demo with 'Show Board Overlay'."
            )
            
            self.app.show_screen("start")
            
        except Exception as e:
            logger.error(f"Failed to save perspective calibration: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not save calibration:\n{str(e)}"
            )
