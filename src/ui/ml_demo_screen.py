"""
ML Demo Screen for testing and visualizing dart detection.

Shows live or test images with ML detection overlays including
bounding boxes, confidence scores, and detected dart positions.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QSlider, QFileDialog, QComboBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from .model_finetune_dialog import ModelFinetuneDialog
from src.calibration.stereo_calibration_data import StereoCalibrationData

logger = logging.getLogger(__name__)


class MLDemoScreen(QWidget):
    """
    Demo screen for ML dart detection testing and visualization.
    
    Features:
    - Live camera feed with ML detection overlay
    - Load test images from disk
    - Adjust ML confidence threshold in real-time
    - Visualize bounding boxes and detection confidence
    - Show detection coordinates and scores
    """
    
    def __init__(self, app):
        """
        Initialize ML demo screen.
        
        Args:
            app: Main BullSight application instance
        """
        super().__init__()
        self.app = app
        self.camera_image_left: Optional[np.ndarray] = None
        self.camera_image_right: Optional[np.ndarray] = None
        self.test_image: Optional[np.ndarray] = None
        self.current_mode = "live"  # "live" or "test"
        
        # Training data paths
        self.training_data_dir = Path("training_data/finetuning_data/images/train")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load stereo calibration if available
        self.stereo_calib = None
        stereo_calib_path = Path("config/stereo_calibration.json")
        if stereo_calib_path.exists():
            try:
                self.stereo_calib = StereoCalibrationData.load(stereo_calib_path)
                logger.info(f"Stereo calibration loaded: baseline={self.stereo_calib.get_baseline():.1f}mm")
            except Exception as e:
                logger.warning(f"Failed to load stereo calibration: {e}")
        else:
            logger.info("No stereo calibration found - 3D reconstruction unavailable")
        
        self.setup_ui()
        
        # Timer for live camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_live_feed)
        
        # Update capture count on startup
        self.update_capture_count()
        
        # Enable keyboard input
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def keyPressEvent(self, event):
        """Handle keyboard events - SPACE to capture training image."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.capture_training_image()
            event.accept()
        else:
            super().keyPressEvent(event)
        
    def setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("ML Dart Detection Demo")
        title.setStyleSheet("font-size: 32px; font-weight: bold; padding: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Main content: horizontal split
        content_layout = QHBoxLayout()
        
        # Left side: Image display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        info_label = QLabel("ML detection + Training: Press SPACE to capture frames for model finetuning")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background: #e8f4f8; font-weight: bold; color: #0066cc;")
        left_layout.addWidget(info_label)
        
        # Training capture button and status
        capture_layout = QHBoxLayout()
        self.capture_btn = QPushButton("📸 Capture for Training")
        self.capture_btn.setMinimumHeight(35)
        self.capture_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;")
        self.capture_btn.clicked.connect(self.capture_training_image)
        capture_layout.addWidget(self.capture_btn)
        
        self.capture_count_label = QLabel("0 images")
        self.capture_count_label.setStyleSheet("font-weight: bold; padding: 5px;")
        capture_layout.addWidget(self.capture_count_label)
        
        left_layout.addLayout(capture_layout)
        
        # Dual-camera view: Two image labels side by side
        camera_views_layout = QHBoxLayout()
        
        # Left camera view
        left_cam_widget = QWidget()
        left_cam_layout = QVBoxLayout(left_cam_widget)
        left_cam_label = QLabel("📷 Left Camera")
        left_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_cam_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        left_cam_layout.addWidget(left_cam_label)
        
        self.image_label_left = QLabel("Loading left camera...")
        self.image_label_left.setMinimumSize(320, 240)
        self.image_label_left.setMaximumSize(640, 480)
        self.image_label_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_left.setStyleSheet("border: 2px solid #0066cc; background: #f0f0f0; font-size: 14px;")
        left_cam_layout.addWidget(self.image_label_left)
        camera_views_layout.addWidget(left_cam_widget)
        
        # Right camera view
        right_cam_widget = QWidget()
        right_cam_layout = QVBoxLayout(right_cam_widget)
        right_cam_label = QLabel("📷 Right Camera")
        right_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_cam_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        right_cam_layout.addWidget(right_cam_label)
        
        self.image_label_right = QLabel("Loading right camera...")
        self.image_label_right.setMinimumSize(320, 240)
        self.image_label_right.setMaximumSize(640, 480)
        self.image_label_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_right.setStyleSheet("border: 2px solid #00cc66; background: #f0f0f0; font-size: 14px;")
        right_cam_layout.addWidget(self.image_label_right)
        camera_views_layout.addWidget(right_cam_widget)
        
        left_layout.addLayout(camera_views_layout)
        
        left_layout.addStretch()
        content_layout.addWidget(left_widget, stretch=2)
        
        # Right side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Mode selection
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Live Camera", "Test Image"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        self.load_image_btn = QPushButton("Load Test Image")
        self.load_image_btn.setMinimumHeight(50)
        self.load_image_btn.clicked.connect(self.load_test_image)
        self.load_image_btn.setEnabled(False)
        mode_layout.addWidget(self.load_image_btn)
        
        mode_group.setLayout(mode_layout)
        controls_layout.addWidget(mode_group)
        
        # ML Settings
        ml_group = QGroupBox("ML Detection Settings")
        ml_layout = QVBoxLayout()
        
        ml_status_label = QLabel("ML Status:")
        ml_layout.addWidget(ml_status_label)
        
        if hasattr(self.app.detector, 'use_ml') and self.app.detector.use_ml:
            self.ml_status = QLabel("✅ ML Active (YOLOv8)")
            self.ml_status.setStyleSheet("color: green; font-weight: bold;")
            ml_layout.addWidget(self.ml_status)
            
            # Show model info
            ml_detector = self.app.detector.ml_detector
            if ml_detector:
                model_info = ml_detector.get_model_info()
                model_label = QLabel(f"Model: {model_info}")
                model_label.setWordWrap(True)
                model_label.setStyleSheet("padding: 8px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; border-radius: 5px; font-size: 11px;")
                ml_layout.addWidget(model_label)
                
                # Check if using DartsVision model
                if "DartsVision" in model_info or "trained" in model_info:
                    info_label = QLabel("✨ Using trained DartsVision model (99% precision)\n\nExcellent dart detection! Darts should be detected accurately.")
                    info_label.setWordWrap(True)
                    info_label.setStyleSheet("padding: 8px; background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; border-radius: 5px; font-size: 11px;")
                    ml_layout.addWidget(info_label)
                else:
                    warning_label = QLabel("⚠️ Using untrained base model\n\nThis model is not trained for darts. DartsVision trained model recommended.")
                    warning_label.setWordWrap(True)
                    warning_label.setStyleSheet("padding: 8px; background: #fff3cd; color: #856404; border: 1px solid #ffc107; border-radius: 5px; font-size: 11px;")
                    ml_layout.addWidget(warning_label)
        else:
            self.ml_status = QLabel("❌ ML Inactive (Classical CV)")
            self.ml_status.setStyleSheet("color: red; font-weight: bold;")
            ml_layout.addWidget(self.ml_status)
        
        # Confidence threshold
        ml_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        ml_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("0.50")
        ml_layout.addWidget(self.confidence_label)
        
        # Finetune button
        self.finetune_btn = QPushButton("📚 Finetune Model")
        self.finetune_btn.setMinimumHeight(40)
        self.finetune_btn.setStyleSheet("background-color: #5b9bd5; color: white; font-weight: bold; border-radius: 5px;")
        self.finetune_btn.clicked.connect(self.open_finetune_dialog)
        ml_layout.addWidget(self.finetune_btn)
        
        ml_group.setLayout(ml_layout)
        controls_layout.addWidget(ml_group)
        
        # Detection Results
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.detection_info = QLabel("No detections yet")
        self.detection_info.setWordWrap(True)
        self.detection_info.setStyleSheet("padding: 10px; background: #fff; font-family: monospace;")
        self.detection_info.setMinimumHeight(150)
        results_layout.addWidget(self.detection_info)
        
        results_group.setLayout(results_layout)
        controls_layout.addWidget(results_group)
        
        # Visualization Options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        self.show_all_btn = QPushButton("Show All Detections")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(True)
        self.show_all_btn.setMinimumHeight(40)
        viz_layout.addWidget(self.show_all_btn)
        
        self.show_board_overlay_btn = QPushButton("Show Board Overlay")
        self.show_board_overlay_btn.setCheckable(True)
        self.show_board_overlay_btn.setChecked(False)
        self.show_board_overlay_btn.setMinimumHeight(40)
        viz_layout.addWidget(self.show_board_overlay_btn)
        
        viz_group.setLayout(viz_layout)
        controls_layout.addWidget(viz_group)
        
        controls_layout.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        back_btn = QPushButton("Back to Menu")
        back_btn.setMinimumHeight(60)
        back_btn.clicked.connect(self.go_back)
        button_layout.addWidget(back_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumHeight(60)
        refresh_btn.setStyleSheet("background-color: #2196F3; color: white;")
        refresh_btn.clicked.connect(self.run_detection)
        button_layout.addWidget(refresh_btn)
        
        controls_layout.addLayout(button_layout)
        
        content_layout.addWidget(controls_widget, stretch=1)
        
        layout.addLayout(content_layout)
        self.setLayout(layout)
    
    def showEvent(self, event):
        """Start camera when screen is shown."""
        super().showEvent(event)
        
        # Ensure camera is started
        if self.app.camera is None:
            logger.info("Camera not started, attempting to start...")
            self.app.start_camera()
        
        if self.current_mode == "live":
            if self.app.camera is not None:
                self.camera_timer.start(100)  # Update every 100ms
            else:
                logger.warning("No camera available for live mode")
                self.image_label.setText("⚠️ No Camera Available\n\nCamera not detected or failed to start.\n\nTry:\n- Check camera connection\n- Enable demo mode: BULLSIGHT_DEMO_MODE=1\n- Or load a test image instead")
        logger.info("ML Demo screen shown")
    
    def hideEvent(self, event):
        """Stop camera when screen is hidden."""
        super().hideEvent(event)
        self.camera_timer.stop()
        logger.info("ML Demo screen hidden")
    
    def go_back(self) -> None:
        """Navigate back to start screen."""
        self.app.show_screen("start")
    
    def on_mode_changed(self, index: int) -> None:
        """Handle mode change."""
        if index == 0:  # Live Camera
            self.current_mode = "live"
            self.load_image_btn.setEnabled(False)
            self.camera_timer.start(100)
        else:  # Test Image
            self.current_mode = "test"
            self.load_image_btn.setEnabled(True)
            self.camera_timer.stop()
    
    def on_confidence_changed(self, value: int) -> None:
        """Handle confidence threshold change."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        
        # Update ML detector confidence if available
        if hasattr(self.app.detector, 'ml_detector') and self.app.detector.ml_detector:
            self.app.detector.ml_detector.confidence_threshold = confidence
        
        # Re-run detection
        self.run_detection()
    
    def load_test_image(self) -> None:
        """Load a test image from disk."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Test Image",
            str(Path("test_images")),
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.test_image = cv2.imread(file_path)
            if self.test_image is not None:
                logger.info(f"Loaded test image: {file_path}")
                self.run_detection()
            else:
                logger.error(f"Failed to load image: {file_path}")
    
    def update_live_feed(self) -> None:
        """Update live camera feed with ML detection."""
        if self.current_mode != "live":
            return
        
        if self.app.camera is None:
            self.camera_timer.stop()
            self.image_label_left.setText("⚠️ Camera disconnected")
            self.image_label_right.setText("⚠️ Camera disconnected")
            return
        
        stereo = self.app.camera.capture_stereo()
        if stereo is not None:
            self.camera_image_left = stereo.left
            self.camera_image_right = stereo.right
            self.run_detection()
        else:
            self.image_label_left.setText("⚠️ Failed to capture frame")
            self.image_label_right.setText("⚠️ Failed to capture frame")
    
    def run_detection(self) -> None:
        """Run ML detection on current image."""
        # Get current image (use left camera for detection in stereo mode)
        if self.current_mode == "live":
            image = self.camera_image_left
            image_right = self.camera_image_right
        else:
            image = self.test_image
            image_right = None
        
        if image is None:
            return
        
        # Check if ML is available
        if not hasattr(self.app.detector, 'use_ml') or not self.app.detector.use_ml:
            self.show_classical_detection(image)
            return
        
        # Run ML detection
        ml_detector = self.app.detector.ml_detector
        if ml_detector is None or not ml_detector.is_available():
            self.detection_info.setText("❌ ML detector not available\n\nInstall: pip install ultralytics torch\nThen restart with: BULLSIGHT_USE_ML=1")
            # Show plain image without detection
            self.display_image(image, self.image_label_left)
            return
        
        try:
            show_all = self.show_all_btn.isChecked()
            
            # Use stereo detection if both images and calibration available
            if self.current_mode == "live" and image_right is not None and self.stereo_calib is not None:
                # Stereo detection with 3D reconstruction
                max_darts = 10 if show_all else 1
                matches = ml_detector.detect_stereo(image, image_right, self.stereo_calib, max_darts=max_darts)
                
                # Visualize on both views
                vis_left = self.visualize_stereo_detections(image.copy(), matches, 'left')
                vis_right = self.visualize_stereo_detections(image_right.copy(), matches, 'right')
                
                # Display stereo info
                self.display_stereo_detections(matches)
                
                # Show both visualized images
                self.display_image(vis_left, self.image_label_left)
                self.display_image(vis_right, self.image_label_right)
                
            else:
                # Fallback to single-camera detection
                if show_all:
                    detections = ml_detector.detect_multiple(image, max_darts=10)
                else:
                    detections = ml_detector.detect_multiple(image, max_darts=1)
                
                # Visualize
                vis_image = self.visualize_detections(image.copy(), detections)
                
                # Update detection info with detailed dartboard mapping
                if detections:
                    info_text = f"🎯 Found {len(detections)} dart(s):\n\n"
                    total_score = 0
                    
                    for i, detection in enumerate(detections, 1):
                        if len(detection) == 4:  # (x, y, conf, class_name)
                            x, y, conf, class_name = detection
                        else:  # Fallback: (x, y, conf)
                            x, y, conf = detection
                            class_name = "unknown"
                        
                        info_text += f"Dart #{i}:\n"
                        info_text += f"  Confidence: {conf:.1%}\n"
                        info_text += f"  Position: ({int(x)}, {int(y)})\n"
                        
                        # Map to dartboard field if calibration exists
                        has_calibration = self.app.mapper.calibration is not None
                        if has_calibration:
                            try:
                                field = self.app.mapper.map_coordinate(int(x), int(y))
                                if field:
                                    # Format dartboard field nicely
                                    if field.zone == "bull_eye":
                                        field_str = "🎯 Bull's Eye (50 points)"
                                        points = 50
                                    elif field.zone == "bull":
                                        field_str = "🎪 Outer Bull (25 points)"
                                        points = 25
                                elif field.zone == "miss":
                                    field_str = "❌ Miss (0 points)"
                                    points = 0
                                elif field.zone == "triple":
                                    field_str = f"🔺 Triple {field.segment} ({field.score} points)"
                                    points = field.score
                                elif field.zone == "double":
                                    field_str = f"🔷 Double {field.segment} ({field.score} points)"
                                    points = field.score
                                else:  # single
                                    field_str = f"🔹 Single {field.segment} ({field.score} points)"
                                    points = field.score
                                
                                info_text += f"  Field: {field_str}\n"
                                total_score += points
                        except Exception as e:
                            logger.debug(f"Could not map coordinate: {e}")
                    
                    info_text += "\n"
                
                # Show total score if calibration exists
                if has_calibration:
                    info_text += f"━━━━━━━━━━━━━━━\n"
                    info_text += f"📊 Total Score: {total_score} points\n"
                else:
                    info_text += "\n⚠️ KALIBRIERUNG ERFORDERLICH:\n"
                    info_text += "Um Dartboard-Felder zu sehen,\n"
                    info_text += "gehe zu:\n"
                    info_text += "⚙️ Settings → 📐 Calibration\n"
                    info_text += "und führe die Kalibrierung durch.\n"
                else:
                    info_text = "No darts detected\n\nTry:\n- Moving the slider to adjust confidence threshold\n- Ensuring good lighting\n- Checking camera alignment"
                
                self.detection_info.setText(info_text)
                
                # Display image
                self.display_image(vis_image, self.image_label_left)
                
                # Show plain right view if available
                if self.current_mode == "live" and image_right is not None:
                    self.display_image(image_right, self.image_label_right)
            
        except Exception as e:
            logger.error(f"ML detection error: {e}", exc_info=True)
            self.detection_info.setText(f"Error: {e}")
            # Still display the raw images even if detection fails
            self.display_image(image, self.image_label_left)
            if self.current_mode == "live" and image_right is not None:
                self.display_image(image_right, self.image_label_right)
    
    def display_stereo_detections(self, matches: list) -> None:
        """Display stereo detection info with 3D positions."""
        if not matches:
            info_text = "No stereo dart matches found\n\n"
            if self.stereo_calib:
                info_text += "Try:\n- Adjusting confidence threshold\n- Ensuring both cameras see the dart\n- Checking that darts are within field of view"
            else:
                info_text += "⚠️ No stereo calibration loaded\n\nRun Stereo Calibration first!"
            self.detection_info.setText(info_text)
            return
        
        info_text = f"🎯 Found {len(matches)} stereo dart match(es):\n\n"
        total_score = 0
        
        for i, match in enumerate(matches, 1):
            x_l, y_l = match['left_2d']
            x_r, y_r = match['right_2d']
            conf_l = match['confidence_left']
            conf_r = match['confidence_right']
            epi_err = match.get('epipolar_error', 0.0)
            
            info_text += f"Dart #{i}:\n"
            info_text += f"  👁️ Left:  ({int(x_l)}, {int(y_l)}) @{conf_l:.1%}\n"
            info_text += f"  👁️ Right: ({int(x_r)}, {int(y_r)}) @{conf_r:.1%}\n"
            info_text += f"  📏 Epipolar Error: {epi_err:.2f}px\n"
            
            # Show 3D position if available
            if match.get('position_3d'):
                X, Y, Z = match['position_3d']
                info_text += f"  📐 3D Position: ({X:.1f}, {Y:.1f}, {Z:.1f})mm\n"
            
            # Map to dartboard field using left camera coordinates
            has_calibration = self.app.mapper.calibration is not None
            if has_calibration:
                try:
                    field = self.app.mapper.map_coordinate(int(x_l), int(y_l))
                    if field:
                        # Format dartboard field nicely
                        if field.zone == "bull_eye":
                            field_str = "🎯 Bull's Eye (50 points)"
                            points = 50
                        elif field.zone == "bull":
                            field_str = "🎪 Outer Bull (25 points)"
                            points = 25
                        elif field.zone == "miss":
                            field_str = "❌ Miss (0 points)"
                            points = 0
                        elif field.zone == "triple":
                            field_str = f"🔺 Triple {field.segment} ({field.score} points)"
                            points = field.score
                        elif field.zone == "double":
                            field_str = f"🔷 Double {field.segment} ({field.score} points)"
                            points = field.score
                        else:  # single
                            field_str = f"🔹 Single {field.segment} ({field.score} points)"
                            points = field.score
                        
                        info_text += f"  Field: {field_str}\n"
                        total_score += points
                except Exception as e:
                    logger.debug(f"Could not map coordinate: {e}")
            
            info_text += "\n"
        
        # Show total score if dartboard calibration exists
        if has_calibration and matches:
            info_text += f"━━━━━━━━━━━━━━━\n"
            info_text += f"📊 Total Score: {total_score} points\n"
            if self.stereo_calib:
                baseline = self.stereo_calib.get_baseline()
                info_text += f"\n📷 Stereo Baseline: {baseline:.1f}mm\n"
        
        self.detection_info.setText(info_text)
    
    def visualize_stereo_detections(self, image: np.ndarray, matches: list, side: str) -> np.ndarray:
        """Visualize stereo detections on one side (left or right)."""
        vis_image = image.copy()
        
        # Draw board overlay if enabled and on left camera
        if side == 'left' and self.show_board_overlay_btn.isChecked() and self.app.mapper.calibration:
            vis_image = self.draw_board_overlay(vis_image)
        
        # Draw detections
        for i, match in enumerate(matches, 1):
            # Get coordinates for this side
            if side == 'left':
                x, y = match['left_2d']
                conf = match['confidence_left']
                color = (0, 255, 0)  # Green for left
            else:
                x, y = match['right_2d']
                conf = match['confidence_right']
                color = (0, 255, 255)  # Yellow for right
            
            x, y = int(x), int(y)
            
            # Draw crosshair
            cv2.drawMarker(vis_image, (x, y), color, cv2.MARKER_CROSS, 30, 3)
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 15, color, 2)
            
            # Draw label
            label = f"#{i} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(vis_image, 
                        (x - 5, y - text_h - 30),
                        (x + text_w + 5, y - 25),
                        color, -1)
            
            # Draw text
            cv2.putText(vis_image, label, 
                       (x, y - 25), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return vis_image
    
    def show_classical_detection(self, image: np.ndarray) -> None:
        """Show classical CV detection (fallback)."""
        vis_image = image.copy()
        
        # Try classical detection if reference exists
        if self.app.detector.reference_image is not None:
            try:
                dart = self.app.detector.detect_dart(image)
                if dart:
                    # Draw detection
                    cv2.drawMarker(vis_image, (dart.x, dart.y), 
                                 (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
                    cv2.circle(vis_image, (dart.x, dart.y), 10, (0, 255, 0), 2)
                    
                    info_text = f"Classical CV Detection:\n\n"
                    info_text += f"Position: ({dart.x}, {dart.y})\n"
                    info_text += f"Confidence: {dart.confidence:.2%}\n"
                    info_text += f"Area: {dart.contour_area:.0f}px²\n"
                    
                    if self.app.mapper.calibration:
                        try:
                            field = self.app.mapper.map_coordinate(dart.x, dart.y)
                            if field:
                                info_text += f"Field: {field}\n"
                        except Exception as e:
                            logger.debug(f"Could not map coordinate: {e}")
                    
                    self.detection_info.setText(info_text)
                else:
                    self.detection_info.setText("No dart detected (Classical CV)")
            except Exception as e:
                self.detection_info.setText(f"Classical CV Error: {e}")
        else:
            self.detection_info.setText("⚠️ No reference image set\n\nGo to Calibration → Capture Reference Image")
        
        self.display_image(vis_image, self.image_label_left)
    
    def visualize_detections(self, image: np.ndarray, 
                            detections: list) -> np.ndarray:
        """
        Visualize ML detections on image.
        
        Args:
            image: Image to draw on
            detections: List of (x, y, confidence, class_name) or (x, y, confidence) tuples
        
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        # Draw board overlay if enabled
        if self.show_board_overlay_btn.isChecked() and self.app.mapper.calibration:
            vis_image = self.draw_board_overlay(vis_image)
        
        # Draw detections
        for i, detection in enumerate(detections, 1):
            # Unpack detection with fallback
            if len(detection) == 4:
                x, y, conf, class_name = detection
            else:
                x, y, conf = detection
                class_name = ""
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low
            
            # Draw crosshair
            cv2.drawMarker(vis_image, (x, y), color, 
                         cv2.MARKER_CROSS, 30, 3)
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 15, color, 2)
            
            # Draw label with class name if available
            if class_name:
                label = f"#{i} {class_name} {conf:.0%}"
            else:
                label = f"#{i} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(vis_image, 
                        (x - 5, y - text_h - 30),
                        (x + text_w + 5, y - 25),
                        color, -1)
            
            # Draw text
            cv2.putText(vis_image, label, 
                       (x, y - 25), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return vis_image
    
    def draw_board_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw dartboard calibration overlay."""
        calibration = self.app.mapper.calibration
        center = (calibration.center_x, calibration.center_y)
        
        # Draw rings (thin lines)
        cv2.circle(image, center, int(calibration.bull_eye_radius), (0, 255, 0), 1)
        cv2.circle(image, center, int(calibration.bull_radius), (0, 255, 255), 1)
        cv2.circle(image, center, int(calibration.triple_inner_radius), (255, 255, 0), 1)
        cv2.circle(image, center, int(calibration.triple_outer_radius), (255, 255, 0), 1)
        cv2.circle(image, center, int(calibration.double_inner_radius), (255, 0, 0), 1)
        cv2.circle(image, center, int(calibration.double_outer_radius), (255, 0, 0), 1)
        
        # Draw center
        cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        
        return image
    
    def display_image(self, image: np.ndarray, label: QLabel) -> None:
        """Display image in specified label."""
        if image is None:
            return
        
        # Convert to QPixmap
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(640, 480, 
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        
        # Show in specified label
        label.setPixmap(scaled_pixmap)
    
    def open_finetune_dialog(self) -> None:
        """Open model finetuning dialog."""
        dialog = ModelFinetuneDialog(self)
        dialog.model_updated.connect(self.on_model_updated)
        dialog.exec()
    
    def on_model_updated(self, model_path: str) -> None:
        """Handle model update from finetuning dialog."""
        logger.info(f"Model finetuned: {model_path}")
    
    def capture_training_image(self) -> None:
        """Capture current frame for training dataset."""
        if self.camera_image_left is None:
            logger.warning("No camera image available for capture")
            return
        
        try:
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"dart_training_{timestamp}.jpg"
            filepath = self.training_data_dir / filename
            
            # Save image
            cv2.imwrite(str(filepath), self.camera_image_left)
            
            logger.info(f"✓ Training image captured: {filename}")
            
            # Update UI
            self.update_capture_count()
            
            # Brief visual feedback
            original_text = self.capture_btn.text()
            self.capture_btn.setText("✅ Saved!")
            self.capture_btn.setStyleSheet("background-color: #20c997; color: white; font-weight: bold; border-radius: 5px;")
            
            # Reset after 1 second
            QTimer.singleShot(1000, lambda: self.reset_capture_button(original_text))
            
        except Exception as e:
            logger.error(f"Failed to capture training image: {e}")
    
    def reset_capture_button(self, original_text: str) -> None:
        """Reset capture button to normal state."""
        self.capture_btn.setText(original_text)
        self.capture_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;")
    
    def update_capture_count(self) -> None:
        """Update the count of captured training images."""
        try:
            # Count images in training directory
            if self.training_data_dir.exists():
                image_files = list(self.training_data_dir.glob("*.jpg")) + list(self.training_data_dir.glob("*.png"))
                count = len(image_files)
                self.capture_count_label.setText(f"{count} images")
                
                if count > 0:
                    self.capture_count_label.setStyleSheet("font-weight: bold; padding: 5px; color: #28a745;")
                else:
                    self.capture_count_label.setStyleSheet("font-weight: bold; padding: 5px;")
            else:
                self.capture_count_label.setText("0 images")
        except Exception as e:
            logger.error(f"Error updating capture count: {e}")
