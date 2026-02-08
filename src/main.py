"""
BullSight Dart Scoring System - Main Application.

Entry point for the dart scoring system with full hardware integration,
computer vision, and touch UI.

Author: Mario Neuhauser
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import QApplication, QStackedWidget
from PySide6.QtCore import Qt

from src.ui.start_screen import StartScreen
from src.ui.player_management_screen import PlayerManagementScreen
from src.ui.game_mode_screen import GameModeScreen
from src.ui.live_score_screen import LiveScoreScreen
from src.ui.calibration_screen import CalibrationScreen
from src.ui.settings_screen import SettingsScreen
from src.ui.ml_demo_screen import MLDemoScreen
from src.ui.stereo_calibration_screen import StereoCalibrationWizard
from src.vision.dual_camera_manager import DualCameraManager, check_available_cameras
from src.vision.dart_detector import DartDetector
from src.calibration.board_mapper import BoardMapper, CalibrationData
from src.game.game_engine import GameEngine
from src.config.settings_manager import SettingsManager


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
        camera (DualCameraManager): Dual camera hardware manager for stereo vision
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
        self.camera: Optional[DualCameraManager] = None
        
        # Initialize dart detector with ML support if available
        # Try to enable ML automatically if ultralytics is installed
        use_ml = os.environ.get('BULLSIGHT_USE_ML', 'auto')
        
        # Auto-detect ML availability
        if use_ml == 'auto':
            try:
                import ultralytics
                use_ml = True
                logger.info("Ultralytics detected - ML detection enabled automatically")
            except ImportError:
                use_ml = False
                logger.info("Ultralytics not found - using classical CV")
        else:
            use_ml = use_ml == '1'
        
        ml_model_path = os.environ.get('BULLSIGHT_ML_MODEL', None)
        ml_confidence = float(os.environ.get('BULLSIGHT_ML_CONFIDENCE', '0.5'))
        
        if use_ml:
            logger.info("ML dart detection enabled")
            if ml_model_path:
                logger.info(f"Using custom model: {ml_model_path}")
        
        self.detector = DartDetector(
            use_ml=use_ml,
            ml_model_path=ml_model_path,
            ml_confidence=ml_confidence
        )
        
        self.mapper = BoardMapper()
        self.game: Optional[GameEngine] = None
        self.settings = SettingsManager()
        
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
        self.screens["ml_demo"] = MLDemoScreen(self)
        self.screens["stereo_calibration"] = StereoCalibrationWizard(self)
        
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
        Initialize and start dual camera system.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            Displays error dialog if less than 2 cameras are available
        """
        try:
            if self.camera is None:
                # Check available cameras
                available_cameras = check_available_cameras()
                
                if len(available_cameras) < 2:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(
                        None,
                        "Dual Camera Required",
                        f"❌ BullSight requires 2 USB cameras for stereo vision.\n\n"
                        f"Found: {len(available_cameras)} camera(s)\n\n"
                        f"Please connect a second USB camera and restart the application."
                    )
                    logger.error(f"Insufficient cameras: Found {len(available_cameras)}, need 2")
                    return False
                
                # Initialize dual camera manager
                logger.info(f"Initializing dual camera system (indices: {available_cameras[0]}, {available_cameras[1]})...")
                self.camera = DualCameraManager(
                    camera_index_left=available_cameras[0],
                    camera_index_right=available_cameras[1],
                    resolution=(1280, 720)
                )
            
            if not self.camera.start():
                logger.error("Failed to start cameras")
                return False
                
            logger.info("✅ Dual camera system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cameras: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "Camera Error",
                f"Failed to initialize cameras:\n\n{str(e)}"
            )
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
                
                calibration = CalibrationData.from_dict(data)
                self.mapper.set_calibration(calibration)
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
            
            if self.mapper.calibration is None:
                logger.error("No calibration data to save")
                return False
            
            import json
            data = self.mapper.calibration.to_dict()
            
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
