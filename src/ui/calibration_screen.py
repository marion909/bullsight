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
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.calibration.board_mapper import CalibrationData, create_default_calibration

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
        
        # Initialize with default calibration
        estimated_radius = min(self.image_width, self.image_height) * 0.4
        self.calibration = create_default_calibration(
            self.image_width,
            self.image_height,
            board_radius_pixels=estimated_radius
        )
        
        # Setup live camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        
        self.setup_ui()
    
    def showEvent(self, event):
        """Start camera when screen is shown."""
        super().showEvent(event)
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
                self.camera_image = self.app.camera.capture()
                self.update_image_display()
        except Exception as e:
            pass  # Silently ignore camera errors during live feed
    
    def setup_ui(self) -> None:
        """Setup calibration UI layout."""
        # Create main widget for scroll area
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Title
        title = QLabel("Dartboard Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Click 'Set Center' and click on the bull's eye\n"
            "2. Adjust ring radii using sliders\n"
            "3. Verify overlay matches your dartboard\n"
            "4. Click 'Save Calibration'"
        )
        instructions.setStyleSheet("padding: 10px; background: #f0f0f0;")
        layout.addWidget(instructions)
        
        # Image display with overlay
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.setMinimumSize(640, 480)
        self.update_image_display()
        layout.addWidget(self.image_label)
        
        # Center point control
        center_group = QGroupBox("Board Center")
        center_layout = QVBoxLayout()
        
        self.center_label = QLabel(
            f"X: {self.calibration.center_x}, Y: {self.calibration.center_y}"
        )
        center_layout.addWidget(self.center_label)
        
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
        layout.addWidget(center_group)
        
        # Radii adjustments
        radii_group = QGroupBox("Ring Radii (pixels)")
        radii_layout = QVBoxLayout()
        
        # Bull's eye
        radii_layout.addWidget(QLabel("Bull's Eye Radius"))
        self.bull_eye_slider = QSlider(Qt.Orientation.Horizontal)
        self.bull_eye_slider.setRange(5, 50)
        self.bull_eye_slider.setValue(int(self.calibration.bull_eye_radius))
        self.bull_eye_slider.valueChanged.connect(self.on_bull_eye_changed)
        radii_layout.addWidget(self.bull_eye_slider)
        self.bull_eye_label = QLabel(f"{int(self.calibration.bull_eye_radius)} px")
        radii_layout.addWidget(self.bull_eye_label)
        
        # Bull
        radii_layout.addWidget(QLabel("Bull Radius"))
        self.bull_slider = QSlider(Qt.Orientation.Horizontal)
        self.bull_slider.setRange(10, 100)
        self.bull_slider.setValue(int(self.calibration.bull_radius))
        self.bull_slider.valueChanged.connect(self.on_bull_changed)
        radii_layout.addWidget(self.bull_slider)
        self.bull_label = QLabel(f"{int(self.calibration.bull_radius)} px")
        radii_layout.addWidget(self.bull_label)
        
        # Triple inner
        radii_layout.addWidget(QLabel("Triple Inner Radius"))
        self.triple_inner_slider = QSlider(Qt.Orientation.Horizontal)
        self.triple_inner_slider.setRange(50, 400)
        self.triple_inner_slider.setValue(int(self.calibration.triple_inner_radius))
        self.triple_inner_slider.valueChanged.connect(self.on_triple_inner_changed)
        radii_layout.addWidget(self.triple_inner_slider)
        self.triple_inner_label = QLabel(f"{int(self.calibration.triple_inner_radius)} px")
        radii_layout.addWidget(self.triple_inner_label)
        
        # Triple outer
        radii_layout.addWidget(QLabel("Triple Outer Radius"))
        self.triple_outer_slider = QSlider(Qt.Orientation.Horizontal)
        self.triple_outer_slider.setRange(60, 420)
        self.triple_outer_slider.setValue(int(self.calibration.triple_outer_radius))
        self.triple_outer_slider.valueChanged.connect(self.on_triple_outer_changed)
        radii_layout.addWidget(self.triple_outer_slider)
        self.triple_outer_label = QLabel(f"{int(self.calibration.triple_outer_radius)} px")
        radii_layout.addWidget(self.triple_outer_label)
        
        # Double inner
        radii_layout.addWidget(QLabel("Double Inner Radius"))
        self.double_inner_slider = QSlider(Qt.Orientation.Horizontal)
        self.double_inner_slider.setRange(200, 600)
        self.double_inner_slider.setValue(int(self.calibration.double_inner_radius))
        self.double_inner_slider.valueChanged.connect(self.on_double_inner_changed)
        radii_layout.addWidget(self.double_inner_slider)
        self.double_inner_label = QLabel(f"{int(self.calibration.double_inner_radius)} px")
        radii_layout.addWidget(self.double_inner_label)
        
        # Double outer
        radii_layout.addWidget(QLabel("Double Outer Radius"))
        self.double_outer_slider = QSlider(Qt.Orientation.Horizontal)
        self.double_outer_slider.setRange(210, 650)
        self.double_outer_slider.setValue(int(self.calibration.double_outer_radius))
        self.double_outer_slider.valueChanged.connect(self.on_double_outer_changed)
        radii_layout.addWidget(self.double_outer_slider)
        self.double_outer_label = QLabel(f"{int(self.calibration.double_outer_radius)} px")
        radii_layout.addWidget(self.double_outer_label)
        
        radii_group.setLayout(radii_layout)
        layout.addWidget(radii_group)
        
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
        
        layout.addLayout(button_layout)
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Set scroll area as main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
    
    def go_back(self) -> None:
        """Navigate back to start screen."""
        self.app.show_screen("start")
    
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
        
        # Trigger autofocus and wait for stabilization
        self.app.camera.trigger_autofocus()
        time.sleep(0.5)
        
        # Capture multiple frames and select the best one
        frames = []
        for i in range(10):
            frame = self.app.camera.capture()
            if frame is not None:
                frames.append(frame)
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
        
        # Capture frame
        self.app.camera.trigger_autofocus()
        import time
        time.sleep(1)
        
        frame = self.app.camera.capture()
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
    
    def on_image_click(self, event) -> None:
        """Handle click on image to set center."""
        if not self.selecting_center:
            return
        
        if self.camera_image is None or self.displayed_pixmap is None:
            return
        
        # Get click position on displayed image
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Get displayed image size
        display_width = self.displayed_pixmap.width()
        display_height = self.displayed_pixmap.height()
        
        # Get original image size
        original_height, original_width = self.camera_image.shape[:2]
        
        # Calculate scale factors
        scale_x = original_width / display_width
        scale_y = original_height / display_height
        
        # Map click coordinates to original image coordinates
        original_x = int(click_x * scale_x)
        original_y = int(click_y * scale_y)
        
        # Clamp to image bounds
        original_x = max(0, min(original_x, original_width - 1))
        original_y = max(0, min(original_y, original_height - 1))
        
        # Update calibration
        self.calibration.center_x = original_x
        self.calibration.center_y = original_y
        
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
        
        # Update UI
        self.center_label.setText(f"X: {original_x}, Y: {original_y}")
        self.center_btn.setText("Set Center")
        self.center_btn.setStyleSheet("")
        self.selecting_center = False
        
        # Refresh display
        self.update_image_display()
    
    def on_bull_eye_changed(self, value: int) -> None:
        """Handle bull's eye radius slider change."""
        self.calibration.bull_eye_radius = float(value)
        self.bull_eye_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def on_bull_changed(self, value: int) -> None:
        """Handle bull radius slider change."""
        self.calibration.bull_radius = float(value)
        self.bull_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def on_triple_inner_changed(self, value: int) -> None:
        """Handle triple inner radius slider change."""
        self.calibration.triple_inner_radius = float(value)
        self.triple_inner_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def on_triple_outer_changed(self, value: int) -> None:
        """Handle triple outer radius slider change."""
        self.calibration.triple_outer_radius = float(value)
        self.triple_outer_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def on_double_inner_changed(self, value: int) -> None:
        """Handle double inner radius slider change."""
        self.calibration.double_inner_radius = float(value)
        self.double_inner_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def on_double_outer_changed(self, value: int) -> None:
        """Handle double outer radius slider change."""
        self.calibration.double_outer_radius = float(value)
        self.double_outer_label.setText(f"{value} px")
        self.update_image_display()
        # Auto-save calibration
        self.app.mapper.calibrate(self.calibration)
        self.app.save_calibration()
    
    def update_image_display(self) -> None:
        """Update image with calibration overlay."""
        if self.camera_image is None:
            # Show placeholder
            self.image_label.setText("No camera image available\nPlease connect camera")
            return
        
        # Update dimensions
        self.image_height, self.image_width = self.camera_image.shape[:2]
        
        # Copy image
        display_image = self.camera_image.copy()
        
        # Draw calibration overlay
        center = (self.calibration.center_x, self.calibration.center_y)
        
        # Draw rings
        cv2.circle(display_image, center, int(self.calibration.bull_eye_radius), (0, 255, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.bull_radius), (0, 255, 255), 2)
        cv2.circle(display_image, center, int(self.calibration.triple_inner_radius), (255, 255, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.triple_outer_radius), (255, 255, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.double_inner_radius), (255, 0, 0), 2)
        cv2.circle(display_image, center, int(self.calibration.double_outer_radius), (255, 0, 0), 2)
        
        # Draw center cross
        cv2.drawMarker(display_image, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
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
