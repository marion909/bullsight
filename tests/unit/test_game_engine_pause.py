"""
Additional game engine tests for pause/resume functionality.

Author: Mario Neuhauser
"""

import pytest

from src.game.game_engine import GameEngine, GameMode, GameState, Player
from src.calibration.board_mapper import DartboardField


class TestPauseResume:
    """Test game pause and resume functionality."""
    
    def test_pause_game_from_in_progress(self):
        """Test pausing game from in progress state."""
        players = [Player("Alice", 1)]
        engine = GameEngine(GameMode.GAME_301, players)
        engine.start_game()
        
        engine.pause_game()
        assert engine.state == GameState.PAUSED
    
    def test_pause_game_from_waiting_for_dart(self):
        """Test pausing game from waiting for dart state."""
        players = [Player("Alice", 1)]
        engine = GameEngine(GameMode.GAME_301, players)
        engine.start_game()
        
        # Record one dart to transition to waiting state
        field = DartboardField(20, "single", 20, 1)
        engine.record_dart(field)
        
        engine.pause_game()
        assert engine.state == GameState.PAUSED
    
    def test_resume_game(self):
        """Test resuming paused game."""
        players = [Player("Alice", 1)]
        engine = GameEngine(GameMode.GAME_301, players)
        engine.start_game()
        
        engine.pause_game()
        assert engine.state == GameState.PAUSED
        
        engine.resume_game()
        assert engine.state == GameState.WAITING_FOR_DART
    
    def test_pause_when_not_pausable(self):
        """Test pause has no effect when game not in progress."""
        players = [Player("Alice", 1)]
        engine = GameEngine(GameMode.GAME_301, players)
        
        # Not started yet
        engine.pause_game()
        assert engine.state == GameState.NOT_STARTED
    
    def test_resume_when_not_paused(self):
        """Test resume has no effect when not paused."""
        players = [Player("Alice", 1)]
        engine = GameEngine(GameMode.GAME_301, players)
        engine.start_game()
        
        engine.resume_game()
        # Should remain in waiting state, not affect it
        assert engine.state == GameState.WAITING_FOR_DART


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
