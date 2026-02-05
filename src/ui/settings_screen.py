"""
Settings screen for BullSight application.

Allows configuration of sound, display, and system settings.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from src.main import BullSightApp


class SettingsScreen(QWidget):
    """
    Settings screen for application configuration.
    
    Features:
    - Sound volume control
    - Sound effects toggle
    - System information display
    - Back to main menu
    
    Attributes:
        app: Main application instance
        sound_volume: Current sound volume (0-100)
        sound_enabled: Sound effects enabled flag
    """
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize settings screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        # Load settings from app
        self.sound_volume = app.settings.get("sound_volume", 70)
        self.sound_enabled = app.settings.get("sound_enabled", True)
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup settings UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Settings")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Sound settings
        sound_section = QLabel("Sound")
        sound_section.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(sound_section)
        
        # Sound toggle
        self.sound_checkbox = QCheckBox("Enable Sound Effects")
        self.sound_checkbox.setChecked(self.sound_enabled)
        self.sound_checkbox.setStyleSheet("font-size: 18px;")
        self.sound_checkbox.stateChanged.connect(self.toggle_sound)
        layout.addWidget(self.sound_checkbox)
        
        # Volume slider
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        volume_label.setStyleSheet("font-size: 18px;")
        volume_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(self.sound_volume)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_value_label = QLabel(f"{self.sound_volume}%")
        self.volume_value_label.setStyleSheet("font-size: 18px; min-width: 60px;")
        volume_layout.addWidget(self.volume_value_label)
        
        layout.addLayout(volume_layout)
        
        layout.addSpacing(30)
        
        # System info
        info_section = QLabel("System Information")
        info_section.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(info_section)
        
        info_text = f"""
        <p style="font-size: 16px; line-height: 1.6;">
        <b>Application:</b> BullSight v{self.app.applicationVersion()}<br>
        <b>Python:</b> {sys.version.split()[0]}<br>
        <b>Platform:</b> {sys.platform}<br>
        <b>Calibration:</b> {'Loaded' if self.app.mapper.calibration else 'Not configured'}
        </p>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Back button
        back_btn = QPushButton("Back to Main Menu")
        back_btn.setMinimumHeight(80)
        back_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)
        
        self.setLayout(layout)
    
    def toggle_sound(self, state: int) -> None:
        """
        Toggle sound effects.
        
        Args:
            state: Checkbox state
        """
        self.sound_enabled = (state == Qt.CheckState.Checked.value)
        self.volume_slider.setEnabled(self.sound_enabled)
        
        # Save setting
        self.app.settings.set("sound_enabled", self.sound_enabled)
        
        # Apply to pygame mixer if initialized
        try:
            import pygame
            if self.sound_enabled:
                pygame.mixer.unpause()
            else:
                pygame.mixer.pause()
        except:
            pass
    
    def change_volume(self, value: int) -> None:
        """
        Change sound volume.
        
        Args:
            value: Volume value (0-100)
        """
        self.sound_volume = value
        self.volume_value_label.setText(f"{value}%")
        
        # Save setting
        self.app.settings.set("sound_volume", value)
        
        # Apply to pygame mixer if initialized
        try:
            import pygame
            pygame.mixer.music.set_volume(value / 100.0)
        except:
            pass
    
    def go_back(self) -> None:
        """Return to start screen."""
        self.app.show_screen("start")
