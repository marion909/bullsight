"""
Start screen for BullSight application.

Main menu with options for new game, calibration, and settings.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.main import BullSightApp


class StartScreen(QWidget):
    """
    Start screen with main menu options.
    
    Features:
    - New game button (goes to player management)
    - Calibration button (goes to calibration screen)
    - Settings button (goes to settings screen)
    - Exit button
    
    Attributes:
        app: Main application instance
    """
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize start screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup start screen UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🎯 BullSight")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(48)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Professional Dart Scoring System")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(20)
        subtitle.setFont(subtitle_font)
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        # New Game button
        new_game_btn = QPushButton("New Game")
        new_game_btn.setMinimumHeight(80)
        new_game_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        new_game_btn.clicked.connect(self.start_new_game)
        layout.addWidget(new_game_btn)
        
        # Calibration button
        calibration_btn = QPushButton("Calibration")
        calibration_btn.setMinimumHeight(80)
        calibration_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        calibration_btn.clicked.connect(self.open_calibration)
        layout.addWidget(calibration_btn)
        
        # Stereo Calibration button
        stereo_calib_btn = QPushButton("🎯 Stereo Calibration")
        stereo_calib_btn.setMinimumHeight(80)
        stereo_calib_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #9C27B0;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        stereo_calib_btn.clicked.connect(self.open_stereo_calibration)
        layout.addWidget(stereo_calib_btn)
        
        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.setMinimumHeight(80)
        settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #FF9800;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(settings_btn)
        
        # ML Demo button
        ml_demo_btn = QPushButton("🤖 ML Detection Demo")
        ml_demo_btn.setMinimumHeight(80)
        ml_demo_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #9C27B0;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        ml_demo_btn.clicked.connect(self.open_ml_demo)
        layout.addWidget(ml_demo_btn)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.setMinimumHeight(80)
        exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        exit_btn.clicked.connect(self.exit_application)
        layout.addWidget(exit_btn)
        
        layout.addStretch()
        
        self.setLayout(layout)
    
    def start_new_game(self) -> None:
        """Navigate to player management screen."""
        self.app.show_screen("player_management")
    
    def open_calibration(self) -> None:
        """Navigate to calibration screen."""
        self.app.show_screen("calibration")
    
    def open_settings(self) -> None:
        """Navigate to settings screen."""
        self.app.show_screen("settings")
    
    def open_ml_demo(self) -> None:
        """Navigate to ML demo screen."""
        self.app.show_screen("ml_demo")
    
    def open_stereo_calibration(self) -> None:
        """Navigate to stereo calibration wizard."""
        self.app.show_screen("stereo_calibration")
    
    def exit_application(self) -> None:
        """Exit the application."""
        self.app.quit()
