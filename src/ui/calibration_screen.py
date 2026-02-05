"""
Calibration UI for dartboard setup.

Provides interactive interface for setting board center and ring boundaries
with visual feedback on camera image.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QPen, QPixmap, QImage, QColor
import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING

from src.calibration.board_mapper import CalibrationData, create_default_calibration

if TYPE_CHECKING:
    from src.main import BullSightApp


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
        
        # State
        self.selecting_center = False
        
        # Initialize with default calibration
        estimated_radius = min(self.image_width, self.image_height) * 0.4
        self.calibration = create_default_calibration(
            self.image_width,
            self.image_height,
            board_radius_pixels=estimated_radius
        )
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup calibration UI layout."""
        layout = QVBoxLayout()
        
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
        self.setLayout(layout)
    
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
        
        # Save image
        cv2.imwrite(str(reference_path), reference_frame)
        
        # Update dart detector with new reference
        if hasattr(self.app, 'detector'):
            self.app.detector.set_reference_image(reference_frame)
            self.app.detector.save_reference_to_file(str(reference_path))
        
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
    
    def on_image_click(self, event) -> None:
        """Handle click on image to set center."""
        if not self.selecting_center:
            return
        
        # Get click position
        x = event.pos().x()
        y = event.pos().y()
        
        # Update calibration
        self.calibration.center_x = x
        self.calibration.center_y = y
        
        # Update UI
        self.center_label.setText(f"X: {x}, Y: {y}")
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
    
    def on_bull_changed(self, value: int) -> None:
        """Handle bull radius slider change."""
        self.calibration.bull_radius = float(value)
        self.bull_label.setText(f"{value} px")
        self.update_image_display()
    
    def on_triple_inner_changed(self, value: int) -> None:
        """Handle triple inner radius slider change."""
        self.calibration.triple_inner_radius = float(value)
        self.triple_inner_label.setText(f"{value} px")
        self.update_image_display()
    
    def on_triple_outer_changed(self, value: int) -> None:
        """Handle triple outer radius slider change."""
        self.calibration.triple_outer_radius = float(value)
        self.triple_outer_label.setText(f"{value} px")
        self.update_image_display()
    
    def on_double_inner_changed(self, value: int) -> None:
        """Handle double inner radius slider change."""
        self.calibration.double_inner_radius = float(value)
        self.double_inner_label.setText(f"{value} px")
        self.update_image_display()
    
    def on_double_outer_changed(self, value: int) -> None:
        """Handle double outer radius slider change."""
        self.calibration.double_outer_radius = float(value)
        self.double_outer_label.setText(f"{value} px")
        self.update_image_display()
    
    def update_image_display(self) -> None:
        """Update image with calibration overlay."""
        if self.camera_image is None:
            # Try to capture from camera
            if self.app.start_camera():
                self.camera_image = self.app.camera.capture()
        
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
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
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
