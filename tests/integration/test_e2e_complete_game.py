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
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from src.main import BullSightApp
from src.game.game_engine import GameMode, Player, GameState
from src.calibration.board_mapper import DartboardField, CalibrationData


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for testing."""
    # Set offscreen platform for headless testing
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit - might affect other tests


class TestApplicationStartup:
    """Test application initialization and startup."""
    
    def test_app_creation(self, qapp):
        """Test BullSightApp can be instantiated."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            assert app is not None
            assert app.applicationName() == "BullSight"
            assert app.applicationVersion() == "1.0.0"
            assert app.stack is not None
            assert len(app.screens) == 6
    
    def test_initial_screen_is_start(self, qapp):
        """Test application starts on start screen."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            current_screen = app.stack.currentWidget()
            assert current_screen == app.screens["start"]
    
    def test_all_screens_exist(self, qapp):
        """Test all required screens are created."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            required_screens = [
                "start", "player_management", "game_mode",
                "live_score", "calibration", "settings"
            ]
            
            for screen_name in required_screens:
                assert screen_name in app.screens
                assert app.screens[screen_name] is not None


class TestNavigation:
    """Test screen navigation."""
    
    def test_navigate_to_player_management(self, qapp):
        """Test navigation from start to player management."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            app.show_screen("player_management")
            assert app.stack.currentWidget() == app.screens["player_management"]
    
    def test_navigate_to_calibration(self, qapp):
        """Test navigation to calibration screen."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            app.show_screen("calibration")
            assert app.stack.currentWidget() == app.screens["calibration"]
    
    def test_navigate_to_settings(self, qapp):
        """Test navigation to settings screen."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            app.show_screen("settings")
            assert app.stack.currentWidget() == app.screens["settings"]


class TestPlayerManagement:
    """Test player management workflow."""
    
    def test_add_single_player(self, qapp):
        """Test adding one player."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            pm_screen = app.screens["player_management"]
            
            # Add player
            pm_screen.player_name_input.setText("Alice")
            pm_screen.add_player()
            
            assert len(pm_screen.players) == 1
            assert pm_screen.players[0].name == "Alice"
    
    def test_add_multiple_players(self, qapp):
        """Test adding multiple players."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            pm_screen = app.screens["player_management"]
            
            # Add three players
            for name in ["Alice", "Bob", "Charlie"]:
                pm_screen.player_name_input.setText(name)
                pm_screen.add_player()
            
            assert len(pm_screen.players) == 3
            assert [p.name for p in pm_screen.players] == ["Alice", "Bob", "Charlie"]
    
    def test_remove_player(self, qapp):
        """Test removing a player."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            pm_screen = app.screens["player_management"]
            
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
    
    def test_max_players_limit(self, qapp):
        """Test maximum 8 players limit."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            pm_screen = app.screens["player_management"]
            
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
    
    def test_default_mode_is_301(self, qapp):
        """Test default selected mode is 301."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            gm_screen = app.screens["game_mode"]
            
            assert gm_screen.selected_mode == GameMode.GAME_301
    
    def test_select_501_mode(self, qapp):
        """Test selecting 501 mode."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            gm_screen = app.screens["game_mode"]
            
            gm_screen.select_mode(GameMode.GAME_501)
            assert gm_screen.selected_mode == GameMode.GAME_501
    
    def test_double_out_toggle(self, qapp):
        """Test double-out option toggle."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            gm_screen = app.screens["game_mode"]
            
            # Default is True
            assert gm_screen.double_out == True
            
            # Toggle off
            gm_screen.double_out_checkbox.setChecked(False)
            gm_screen.toggle_double_out(Qt.CheckState.Unchecked.value)
            assert gm_screen.double_out == False


class TestGameEngineIntegration:
    """Test game engine integration with UI."""
    
    def test_start_301_game(self, qapp):
        """Test starting a 301 game."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            # Setup players
            pm_screen = app.screens["player_management"]
            pm_screen.player_name_input.setText("Alice")
            pm_screen.add_player()
            pm_screen.player_name_input.setText("Bob")
            pm_screen.add_player()
            
            # Select mode and start
            gm_screen = app.screens["game_mode"]
            gm_screen.select_mode(GameMode.GAME_301)
            gm_screen.start_game()
            
            assert app.game is not None
            assert app.game.mode == GameMode.GAME_301
            assert len(app.game.players) == 2
            assert app.game.state == GameState.IN_PROGRESS
    
    def test_game_initial_scores(self, qapp):
        """Test initial scores for countdown game."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            # Setup players
            pm_screen = app.screens["player_management"]
            pm_screen.player_name_input.setText("Alice")
            pm_screen.add_player()
            
            # Start 301 game
            gm_screen = app.screens["game_mode"]
            gm_screen.select_mode(GameMode.GAME_301)
            gm_screen.start_game()
            
            assert app.game.players[0].score == 301


class TestCalibration:
    """Test calibration workflow."""
    
    def test_calibration_screen_loads(self, qapp):
        """Test calibration screen can be loaded."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            
            calib_screen = app.screens["calibration"]
            assert calib_screen is not None
            assert calib_screen.calibration is not None
    
    def test_save_calibration(self, qapp):
        """Test saving calibration."""
        with patch('src.main.Path.mkdir'):
            with patch('builtins.open', create=True) as mock_open:
                with patch('json.dump') as mock_dump:
                    app = BullSightApp([])
                    
                    # Create calibration data
                    calib = CalibrationData(
                        center_x=320,
                        center_y=240,
                        inner_bull_radius=10.0,
                        outer_bull_radius=20.0,
                        inner_single_radius=50.0,
                        triple_radius=60.0,
                        outer_single_radius=80.0,
                        double_radius=90.0
                    )
                    
                    app.mapper.calibrate(calib)
                    result = app.save_calibration()
                    
                    # Verify save was attempted
                    assert result == True


class TestSettings:
    """Test settings screen."""
    
    def test_default_volume(self, qapp):
        """Test default volume setting."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            settings_screen = app.screens["settings"]
            
            assert settings_screen.sound_volume == 70
    
    def test_sound_toggle(self, qapp):
        """Test sound enable/disable."""
        with patch('src.main.Path.mkdir'):
            app = BullSightApp([])
            settings_screen = app.screens["settings"]
            
            # Default is enabled
            assert settings_screen.sound_enabled == True
            
            # Disable
            settings_screen.sound_checkbox.setChecked(False)
            settings_screen.toggle_sound(Qt.CheckState.Unchecked.value)
            
            assert settings_screen.sound_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
