"""
BullSight Dart Scoring System - Main Application.

Entry point for the dart scoring system with full hardware integration,
computer vision, and touch UI.

Author: Mario Neuhauser
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication, QStackedWidget
from PyQt6.QtCore import Qt

from src.ui.start_screen import StartScreen
from src.ui.player_management_screen import PlayerManagementScreen
from src.ui.game_mode_screen import GameModeScreen
from src.ui.live_score_screen import LiveScoreScreen
from src.ui.calibration_screen import CalibrationScreen
from src.ui.settings_screen import SettingsScreen
from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
from src.calibration.board_mapper import BoardMapper, CalibrationData
from src.game.game_engine import GameEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bullsight.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BullSightApp(QApplication):
    """
    Main application class for BullSight.
    
    Manages screen navigation, hardware components, and application lifecycle.
    
    Attributes:
        stack (QStackedWidget): Screen stack for navigation
        camera (CameraManager): Camera hardware manager
        detector (DartDetector): Dart detection engine
        mapper (BoardMapper): Coordinate mapping system
        game (GameEngine): Current game instance
        screens (dict): All application screens
    """
    
    def __init__(self, argv):
        """
        Initialize BullSight application.
        
        Args:
            argv: Command line arguments
        """
        super().__init__(argv)
        
        # Set application properties
        self.setApplicationName("BullSight")
        self.setApplicationVersion("1.0.0")
        
        # Initialize hardware components
        self.camera: Optional[CameraManager] = None
        self.detector = DartDetector()
        self.mapper = BoardMapper()
        self.game: Optional[GameEngine] = None
        
        # Create main window stack
        self.stack = QStackedWidget()
        self.stack.setWindowTitle("BullSight Dart Scoring")
        self.stack.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Initialize screens
        self.screens = {}
        self.setup_screens()
        
        # Load calibration if exists
        self.load_calibration()
        
        # Show start screen
        self.show_screen("start")
        self.stack.showFullScreen()
        
        logger.info("BullSight application initialized")
    
    def setup_screens(self) -> None:
        """Initialize all application screens."""
        self.screens["start"] = StartScreen(self)
        self.screens["player_management"] = PlayerManagementScreen(self)
        self.screens["game_mode"] = GameModeScreen(self)
        self.screens["live_score"] = LiveScoreScreen(self)
        self.screens["calibration"] = CalibrationScreen(self)
        self.screens["settings"] = SettingsScreen(self)
        
        for screen in self.screens.values():
            self.stack.addWidget(screen)
    
    def show_screen(self, screen_name: str) -> None:
        """
        Navigate to specified screen.
        
        Args:
            screen_name: Name of screen to show
        """
        if screen_name in self.screens:
            self.stack.setCurrentWidget(self.screens[screen_name])
            logger.info(f"Navigated to: {screen_name}")
    
    def start_camera(self) -> bool:
        """
        Initialize and start camera.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.camera is None:
                self.camera = CameraManager()
            self.camera.start()
            logger.info("Camera started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self) -> None:
        """Stop camera and release resources."""
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
    
    def load_calibration(self) -> bool:
        """
        Load calibration from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        calibration_path = Path("config/calibration.json")
        if calibration_path.exists():
            try:
                import json
                with open(calibration_path, 'r') as f:
                    data = json.load(f)
                
                calibration = CalibrationData(
                    center_x=data['center_x'],
                    center_y=data['center_y'],
                    inner_bull_radius=data['inner_bull_radius'],
                    outer_bull_radius=data['outer_bull_radius'],
                    inner_single_radius=data['inner_single_radius'],
                    triple_radius=data['triple_radius'],
                    outer_single_radius=data['outer_single_radius'],
                    double_radius=data['double_radius']
                )
                self.mapper.calibrate(calibration)
                logger.info("Calibration loaded")
                return True
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
        return False
    
    def save_calibration(self) -> bool:
        """
        Save current calibration to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        calibration_path = Path("config/calibration.json")
        try:
            # Ensure config directory exists
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.mapper.calibration_data is None:
                logger.error("No calibration data to save")
                return False
            
            import json
            data = {
                'center_x': self.mapper.calibration_data.center_x,
                'center_y': self.mapper.calibration_data.center_y,
                'inner_bull_radius': self.mapper.calibration_data.inner_bull_radius,
                'outer_bull_radius': self.mapper.calibration_data.outer_bull_radius,
                'inner_single_radius': self.mapper.calibration_data.inner_single_radius,
                'triple_radius': self.mapper.calibration_data.triple_radius,
                'outer_single_radius': self.mapper.calibration_data.outer_single_radius,
                'double_radius': self.mapper.calibration_data.double_radius
            }
            
            with open(calibration_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Calibration saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False


def main():
    """Application entry point."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Create and run application
    app = BullSightApp(sys.argv)
    
    try:
        exit_code = app.exec()
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Cleanup
        app.stop_camera()
        logger.info("Application shutdown")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
