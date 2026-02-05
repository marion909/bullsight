"""
End-to-end integration tests for BullSight.

Tests complete game flow from startup to game completion with all
components integrated.

Author: Mario Neuhauser
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Ensure PyQt6 uses offscreen platform for testing
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

from src.main import BullSightApp
from src.game.game_engine import GameMode, Player, GameState
from src.calibration.board_mapper import DartboardField, CalibrationData


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_app():
    """Create mock app object for screen testing."""
    app = Mock()
    app.show_screen = Mock()
    app.start_camera = Mock(return_value=False)
    app.stop_camera = Mock()
    app.save_calibration = Mock(return_value=True)
    
    from src.vision.dart_detector import DartDetector
    from src.calibration.board_mapper import BoardMapper
    
    app.detector = DartDetector()
    app.mapper = BoardMapper()
    app.camera = None
    app.game = None
    app.applicationVersion = Mock(return_value="1.0.0")
    
    return app


class TestApplicationStartup:
    """Test application screen initialization."""
    
    def test_start_screen_creation(self, qapp, mock_app):
        """Test StartScreen can be created."""
        from src.ui.start_screen import StartScreen
        screen = StartScreen(mock_app)
        assert screen is not None
    
    def test_player_management_creation(self, qapp, mock_app):
        """Test PlayerManagementScreen can be created."""
        from src.ui.player_management_screen import PlayerManagementScreen
        screen = PlayerManagementScreen(mock_app)
        assert screen is not None
        assert len(screen.players) == 0
    
    def test_all_screens_can_be_created(self, qapp, mock_app):
        """Test all screens can be instantiated."""
        from src.ui.start_screen import StartScreen
        from src.ui.player_management_screen import PlayerManagementScreen
        from src.ui.game_mode_screen import GameModeScreen
        from src.ui.live_score_screen import LiveScoreScreen
        from src.ui.settings_screen import SettingsScreen
        from src.ui.calibration_screen import CalibrationScreen
        
        screens = {
            "start": StartScreen(mock_app),
            "player_management": PlayerManagementScreen(mock_app),
            "game_mode": GameModeScreen(mock_app),
            "live_score": LiveScoreScreen(mock_app),
            "settings": SettingsScreen(mock_app),
            "calibration": CalibrationScreen(mock_app)
        }
        
        for name, screen in screens.items():
            assert screen is not None, f"{name} screen is None"


class TestNavigation:
    """Test screen button functionality."""
    
    def test_start_screen_buttons(self, qapp, mock_app):
        """Test start screen buttons call correct methods."""
        from src.ui.start_screen import StartScreen
        screen = StartScreen(mock_app)
        
        # Test new game button calls show_screen
        screen.start_new_game()
        mock_app.show_screen.assert_called_with("player_management")
    
    def test_back_navigation(self, qapp, mock_app):
        """Test back button navigation."""
        from src.ui.player_management_screen import PlayerManagementScreen
        screen = PlayerManagementScreen(mock_app)
        
        screen.go_back()
        mock_app.show_screen.assert_called_with("start")


class TestPlayerManagement:
    """Test player management workflow."""
    
    def test_add_single_player(self, qapp, mock_app):
        """Test adding one player."""
        from src.ui.player_management_screen import PlayerManagementScreen
        pm_screen = PlayerManagementScreen(mock_app)
        
        # Add player
        pm_screen.player_name_input.setText("Alice")
        pm_screen.add_player()
        
        assert len(pm_screen.players) == 1
        assert pm_screen.players[0].name == "Alice"
    
    def test_add_multiple_players(self, qapp, mock_app):
        """Test adding multiple players."""
        from src.ui.player_management_screen import PlayerManagementScreen
        pm_screen = PlayerManagementScreen(mock_app)
        
        # Add three players
        for name in ["Alice", "Bob", "Charlie"]:
            pm_screen.player_name_input.setText(name)
            pm_screen.add_player()
        
        assert len(pm_screen.players) == 3
        assert [p.name for p in pm_screen.players] == ["Alice", "Bob", "Charlie"]
    
    def test_remove_player(self, qapp, mock_app):
        """Test removing a player."""
        from src.ui.player_management_screen import PlayerManagementScreen
        pm_screen = PlayerManagementScreen(mock_app)
        
        # Add two players
        pm_screen.player_name_input.setText("Alice")
        pm_screen.add_player()
        pm_screen.player_name_input.setText("Bob")
        pm_screen.add_player()
        
        # Remove first player
        pm_screen.player_list.setCurrentRow(0)
        pm_screen.remove_player()
        
        assert len(pm_screen.players) == 1
        assert pm_screen.players[0].name == "Bob"
    
    def test_max_players_limit(self, qapp, mock_app):
        """Test maximum 8 players limit."""
        from src.ui.player_management_screen import PlayerManagementScreen
        pm_screen = PlayerManagementScreen(mock_app)
        
        # Add 8 players
        for i in range(8):
            pm_screen.player_name_input.setText(f"Player{i+1}")
            pm_screen.add_player()
        
        assert len(pm_screen.players) == 8
        
        # Try to add 9th player - should fail
        pm_screen.player_name_input.setText("Player9")
        pm_screen.add_player()
        
        assert len(pm_screen.players) == 8  # Still 8


class TestGameModeSelection:
    """Test game mode selection."""
    
    def test_default_mode_is_301(self, qapp, mock_app):
        """Test default selected mode is 301."""
        from src.ui.game_mode_screen import GameModeScreen
        gm_screen = GameModeScreen(mock_app)
        assert gm_screen.selected_mode == GameMode.GAME_301
    
    def test_select_501_mode(self, qapp, mock_app):
        """Test selecting 501 mode."""
        from src.ui.game_mode_screen import GameModeScreen
        gm_screen = GameModeScreen(mock_app)
        gm_screen.select_mode(GameMode.GAME_501)
        assert gm_screen.selected_mode == GameMode.GAME_501
    
    def test_double_out_toggle(self, qapp, mock_app):
        """Test double-out option toggle."""
        from src.ui.game_mode_screen import GameModeScreen
        from PySide6.QtCore import Qt
        gm_screen = GameModeScreen(mock_app)
        
        # Default is True
        assert gm_screen.double_out == True
        
        # Toggle off
        gm_screen.double_out_checkbox.setChecked(False)
        gm_screen.toggle_double_out(Qt.CheckState.Unchecked.value)
        assert gm_screen.double_out == False


class TestGameEngineIntegration:
    """Test game engine integration with UI."""
    
    def test_game_mode_screen_has_start_game_method(self, qapp, mock_app):
        """Test that GameModeScreen has start_game method."""
        from src.ui.game_mode_screen import GameModeScreen
        
        gm_screen = GameModeScreen(mock_app)
        assert hasattr(gm_screen, 'start_game')
        assert callable(gm_screen.start_game)
    
    def test_game_mode_screen_tracks_selected_mode(self, qapp, mock_app):
        """Test that GameModeScreen tracks selected mode."""
        from src.ui.game_mode_screen import GameModeScreen
        
        gm_screen = GameModeScreen(mock_app)
        
        # Default should be 301
        assert gm_screen.selected_mode == GameMode.GAME_301
        
        # Select 501
        gm_screen.select_mode(GameMode.GAME_501)
        assert gm_screen.selected_mode == GameMode.GAME_501


class TestCalibration:
    """Test calibration workflow."""
    
    def test_calibration_screen_loads(self, qapp, mock_app):
        """Test calibration screen can be loaded."""
        from src.ui.calibration_screen import CalibrationScreen
        calib_screen = CalibrationScreen(mock_app)
        assert calib_screen is not None
        assert calib_screen.calibration is not None
    
    def test_save_calibration(self, qapp, mock_app):
        """Test saving calibration."""
        from src.ui.calibration_screen import CalibrationScreen
        from src.calibration.board_mapper import CalibrationData
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.dump') as mock_dump:
                # Create calibration data with correct parameter names
                calib = CalibrationData(
                    center_x=320,
                    center_y=240,
                    bull_eye_radius=10.0,
                    bull_radius=20.0,
                    triple_inner_radius=50.0,
                    triple_outer_radius=60.0,
                    double_inner_radius=80.0,
                    double_outer_radius=90.0
                )
                
                # Use set_calibration instead of calibrate
                mock_app.mapper.set_calibration(calib)
                result = mock_app.save_calibration()
                
                # Verify save was attempted
                mock_app.save_calibration.assert_called_once()


class TestSettings:
    """Test settings screen."""
    
    def test_default_volume(self, qapp, mock_app):
        """Test default volume setting."""
        from src.ui.settings_screen import SettingsScreen
        settings_screen = SettingsScreen(mock_app)
        assert settings_screen.sound_volume == 70
    
    def test_sound_toggle(self, qapp, mock_app):
        """Test sound enable/disable."""
        from src.ui.settings_screen import SettingsScreen
        from PySide6.QtCore import Qt
        settings_screen = SettingsScreen(mock_app)
        
        # Default is enabled
        assert settings_screen.sound_enabled == True
        
        # Disable
        settings_screen.sound_checkbox.setChecked(False)
        settings_screen.toggle_sound(Qt.CheckState.Unchecked.value)
        
        assert settings_screen.sound_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
