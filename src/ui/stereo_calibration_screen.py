"""
Stereo Calibration Wizard UI.

Interactive wizard for calibrating dual-camera stereo system using
checkerboard pattern detection.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QSpinBox, QMessageBox, QTextEdit
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from src.calibration.stereo_calibration_data import StereoCalibrationData, create_stereo_calibration_data

logger = logging.getLogger(__name__)


class StereoCalibrationWizard(QWidget):
    """
    Interactive wizard for stereo camera calibration.
    
    Workflow:
    1. Show live dual-camera feed
    2. Detect checkerboard pattern in both views
    3. Capture calibration image pairs when pattern detected
    4. After collecting enough pairs (15-30), run calibration
    5. Save results and show reprojection error
    """
    
    def __init__(self, app):
        """
        Initialize stereo calibration wizard.
        
        Args:
            app: Main BullSight application instance
        """
        super().__init__()
        self.app = app
        
        # Calibration parameters
        self.pattern_size = (9, 6)  # Internal corners (columns, rows)
        self.square_size = 1.0  # Square size in arbitrary units (e.g., mm)
        
        # Captured calibration data
        self.obj_points = []  # 3D points in real world space
        self.img_points_left = []  # 2D points in left image plane
        self.img_points_right = []  # 2D points in right image plane
        self.calibration_images_left = []  # Stored images for verification
        self.calibration_images_right = []
        
        # UI state
        self.target_captures = 20  # Minimum captures needed
        self.is_calibrating = False
        
        self.setup_ui()
        
        # Timer for live feed
        self.feed_timer = QTimer()
        self.feed_timer.timeout.connect(self.update_feed)
        self.feed_timer.start(50)  # 20 FPS
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🎯 Stereo Camera Calibration Wizard")
        title.setStyleSheet("font-size: 28px; font-weight: bold; padding: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        info = QLabel(
            "📋 Instructions:\n"
            "1. Print a checkerboard pattern (9×6 internal corners)\n"
            "2. Hold it flat in view of both cameras\n"
            "3. Move and rotate pattern to different positions\n"
            "4. Press SPACE when both cameras detect the pattern\n"
            "5. Collect 20+ images from various angles\n"
            "6. Click 'Calibrate Cameras' to compute stereo parameters"
        )
        info.setStyleSheet("background: #e8f4f8; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Pattern settings
        settings_group = QGroupBox("Pattern Settings")
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(4, 20)
        self.columns_spin.setValue(9)
        self.columns_spin.valueChanged.connect(self.on_pattern_changed)
        settings_layout.addWidget(self.columns_spin)
        
        settings_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(4, 20)
        self.rows_spin.setValue(6)
        self.rows_spin.valueChanged.connect(self.on_pattern_changed)
        settings_layout.addWidget(self.rows_spin)
        
        settings_layout.addWidget(QLabel("Square Size (mm):"))
        self.square_spin = QSpinBox()
        self.square_spin.setRange(1, 100)
        self.square_spin.setValue(25)
        self.square_spin.valueChanged.connect(self.on_square_size_changed)
        settings_layout.addWidget(self.square_spin)
        
        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Camera views
        views_layout = QHBoxLayout()
        
        # Left camera view
        left_group = QGroupBox("📷 Left Camera")
        left_layout = QVBoxLayout()
        self.left_image_label = QLabel("Waiting for camera...")
        self.left_image_label.setMinimumSize(640, 480)
        self.left_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_image_label.setStyleSheet("border: 2px solid #0066cc; background: #000;")
        left_layout.addWidget(self.left_image_label)
        
        self.left_status_label = QLabel("⏳ No pattern detected")
        self.left_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(self.left_status_label)
        
        left_group.setLayout(left_layout)
        views_layout.addWidget(left_group)
        
        # Right camera view
        right_group = QGroupBox("📷 Right Camera")
        right_layout = QVBoxLayout()
        self.right_image_label = QLabel("Waiting for camera...")
        self.right_image_label.setMinimumSize(640, 480)
        self.right_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_image_label.setStyleSheet("border: 2px solid #00cc66; background: #000;")
        right_layout.addWidget(self.right_image_label)
        
        self.right_status_label = QLabel("⏳ No pattern detected")
        self.right_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        right_layout.addWidget(self.right_status_label)
        
        right_group.setLayout(right_layout)
        views_layout.addWidget(right_group)
        
        layout.addLayout(views_layout)
        
        # Progress section
        progress_group = QGroupBox("📊 Calibration Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Captured: 0 / 20 image pairs")
        self.progress_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(20)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.capture_btn = QPushButton("📸 Capture (SPACE)")
        self.capture_btn.setMinimumHeight(50)
        self.capture_btn.setStyleSheet("font-size: 16px; background-color: #28a745; color: white; font-weight: bold;")
        self.capture_btn.clicked.connect(self.capture_calibration_pair)
        self.capture_btn.setEnabled(False)
        buttons_layout.addWidget(self.capture_btn)
        
        self.calibrate_btn = QPushButton("🎯 Calibrate Cameras")
        self.calibrate_btn.setMinimumHeight(50)
        self.calibrate_btn.setStyleSheet("font-size: 16px; background-color: #007bff; color: white; font-weight: bold;")
        self.calibrate_btn.clicked.connect(self.run_calibration)
        self.calibrate_btn.setEnabled(False)
        buttons_layout.addWidget(self.calibrate_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.setMinimumHeight(50)
        self.clear_btn.setStyleSheet("font-size: 16px; background-color: #dc3545; color: white; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear_captures)
        buttons_layout.addWidget(self.clear_btn)
        
        layout.addLayout(buttons_layout)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Calibration results will appear here...")
        layout.addWidget(self.results_text)
        
        # Back button
        back_btn = QPushButton("← Back to Menu")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)
        
        self.setLayout(layout)
        
        # Enable keyboard shortcuts
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self.capture_btn.isEnabled():
                self.capture_calibration_pair()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def on_pattern_changed(self):
        """Handle pattern size change."""
        self.pattern_size = (self.columns_spin.value(), self.rows_spin.value())
        logger.info(f"Pattern size changed to {self.pattern_size}")
    
    def on_square_size_changed(self):
        """Handle square size change."""
        self.square_size = self.square_spin.value()
        logger.info(f"Square size changed to {self.square_size}mm")
    
    def update_feed(self):
        """Update camera feed with checkerboard detection."""
        if self.app.camera is None:
            # Try to start camera
            if not self.app.start_camera():
                self.left_image_label.setText("⚠️ Camera not available\n\nConnect 2 USB cameras")
                self.right_image_label.setText("⚠️ Camera not available\n\nConnect 2 USB cameras")
                self.feed_timer.stop()
                return
        
        if not self.app.camera.is_started:
            # Camera exists but not started, try to start it
            if not self.app.camera.start():
                self.left_image_label.setText("⚠️ Failed to start cameras")
                self.right_image_label.setText("⚠️ Failed to start cameras")
                self.feed_timer.stop()
                return
        
        if self.is_calibrating:
            return  # Don't update during calibration
        
        # Capture stereo frame
        stereo = self.app.camera.capture_stereo()
        if stereo is None:
            return
        
        # Detect checkerboard in both images
        gray_left = cv2.cvtColor(stereo.left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(stereo.right, cv2.COLOR_BGR2GRAY)
        
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Draw checkerboard if detected
        vis_left = stereo.left.copy()
        vis_right = stereo.right.copy()
        
        if ret_left:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis_left, self.pattern_size, corners_left, ret_left)
            self.left_status_label.setText("✅ Pattern detected!")
            self.left_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.left_status_label.setText("⏳ No pattern detected")
            self.left_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: orange;")
        
        if ret_right:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis_right, self.pattern_size, corners_right, ret_right)
            self.right_status_label.setText("✅ Pattern detected!")
            self.right_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.right_status_label.setText("⏳ No pattern detected")
            self.right_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: orange;")
        
        # Enable capture button only if both patterns detected
        both_detected = ret_left and ret_right
        self.capture_btn.setEnabled(both_detected)
        
        # Store corners for capture
        if both_detected:
            self.current_corners_left = corners_left
            self.current_corners_right = corners_right
        
        # Display images
        self.display_image(vis_left, self.left_image_label)
        self.display_image(vis_right, self.right_image_label)
    
    def display_image(self, image: np.ndarray, label: QLabel):
        """Display OpenCV image in QLabel."""
        if image is None:
            return
        
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(640, 480,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def capture_calibration_pair(self):
        """Capture current image pair for calibration."""
        if not hasattr(self, 'current_corners_left') or not hasattr(self, 'current_corners_right'):
            return
        
        # Prepare object points for this checkerboard
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # Store points
        self.obj_points.append(objp)
        self.img_points_left.append(self.current_corners_left)
        self.img_points_right.append(self.current_corners_right)
        
        # Store images for verification
        stereo = self.app.camera.capture_stereo()
        if stereo:
            self.calibration_images_left.append(stereo.left.copy())
            self.calibration_images_right.append(stereo.right.copy())
        
        # Update progress
        count = len(self.obj_points)
        self.progress_label.setText(f"Captured: {count} / {self.target_captures} image pairs")
        self.progress_bar.setValue(count)
        
        # Enable calibration button if enough captures
        if count >= self.target_captures:
            self.calibrate_btn.setEnabled(True)
        
        # Visual feedback
        original_text = self.capture_btn.text()
        self.capture_btn.setText("✅ Captured!")
        QTimer.singleShot(500, lambda: self.capture_btn.setText(original_text))
        
        logger.info(f"Captured calibration pair {count}/{self.target_captures}")
    
    def clear_captures(self):
        """Clear all captured calibration data."""
        reply = QMessageBox.question(
            self, "Clear All Captures",
            "Are you sure you want to delete all captured calibration images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.obj_points.clear()
            self.img_points_left.clear()
            self.img_points_right.clear()
            self.calibration_images_left.clear()
            self.calibration_images_right.clear()
            
            self.progress_label.setText("Captured: 0 / 20 image pairs")
            self.progress_bar.setValue(0)
            self.calibrate_btn.setEnabled(False)
            self.results_text.clear()
            
            logger.info("Cleared all calibration captures")
    
    def run_calibration(self):
        """Run stereo calibration with captured images."""
        if len(self.obj_points) < self.target_captures:
            QMessageBox.warning(self, "Not Enough Images",
                              f"Need at least {self.target_captures} image pairs. Currently have {len(self.obj_points)}.")
            return
        
        self.is_calibrating = True
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setText("⏳ Calibrating...")
        
        try:
            # Get image size from first image
            h, w = self.calibration_images_left[0].shape[:2]
            image_size = (w, h)
            
            # Calibrate left camera individually
            ret_left, K_left, D_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
                self.obj_points, self.img_points_left, image_size, None, None
            )
            
            # Calibrate right camera individually
            ret_right, K_right, D_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
                self.obj_points, self.img_points_right, image_size, None, None
            )
            
            # Stereo calibration
            flags = cv2.CALIB_FIX_INTRINSIC  # Use individual calibrations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
            ret_stereo, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
                self.obj_points, self.img_points_left, self.img_points_right,
                K_left, D_left, K_right, D_right,
                image_size, criteria=criteria, flags=flags
            )
            
            # Create calibration data object
            stereo_calib = create_stereo_calibration_data(
                K_left, D_left, K_right, D_right,
                R, T, E, F, image_size, ret_stereo
            )
            
            # Save calibration
            save_path = Path("config/stereo_calibration.json")
            stereo_calib.save(save_path)
            
            # Display results
            baseline = stereo_calib.get_baseline()
            focal_avg = stereo_calib.get_focal_length_avg()
            
            results = f"""
🎉 Stereo Calibration Successful!

📊 Results:
• RMS Reprojection Error: {ret_stereo:.4f} pixels
• Baseline Distance: {baseline:.2f} mm
• Average Focal Length: {focal_avg:.1f} pixels
• Image Size: {w}x{h}

📐 Left Camera:
• fx: {K_left[0,0]:.2f}, fy: {K_left[1,1]:.2f}
• cx: {K_left[0,2]:.2f}, cy: {K_left[1,2]:.2f}

📐 Right Camera:
• fx: {K_right[0,0]:.2f}, fy: {K_right[1,1]:.2f}
• cx: {K_right[0,2]:.2f}, cy: {K_right[1,2]:.2f}

✅ Calibration saved to: {save_path}

You can now use stereo vision for 3D dart reconstruction!
"""
            
            self.results_text.setText(results)
            
            QMessageBox.information(self, "Calibration Complete",
                                  f"Stereo calibration successful!\n\nRMS Error: {ret_stereo:.4f} pixels\nBaseline: {baseline:.2f} mm")
            
            logger.info(f"Stereo calibration completed: RMS={ret_stereo:.4f}, baseline={baseline:.2f}mm")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Calibration Error", f"Failed to calibrate cameras:\n\n{str(e)}")
            self.results_text.setText(f"❌ Calibration Error:\n{str(e)}")
        
        finally:
            self.is_calibrating = False
            self.calibrate_btn.setText("🎯 Calibrate Cameras")
    
    def go_back(self):
        """Navigate back to main menu."""
        self.feed_timer.stop()
        self.app.show_screen("start")
    
    def showEvent(self, event):
        """Handle widget show event."""
        super().showEvent(event)
        self.feed_timer.start(50)
        logger.info("Stereo calibration wizard shown")
    
    def hideEvent(self, event):
        """Handle widget hide event."""
        super().hideEvent(event)
        self.feed_timer.stop()
        logger.info("Stereo calibration wizard hidden")
