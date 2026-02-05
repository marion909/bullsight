"""
Additional edge case tests for game engine.

Covers remaining untested scenarios for 100% coverage.

Author: Mario Neuhauser
"""

import pytest

from src.game.game_engine import GameEngine, GameMode, GameState, Player
from src.calibration.board_mapper import DartboardField


class TestGameEngineEdgeCases:
    """Edge case tests for GameEngine."""
    
    def test_complete_round_with_no_current_round(self):
        """Test complete_round when current_round is None."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.GAME_301, [player])
        game.start_game()
        
        # Force current_round to None
        game.current_round = None
        game.complete_round()  # Should not crash
    
    def test_cricket_mode_initialization(self):
        """Test Cricket mode initialization."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.CRICKET, [player])
        game.start_game()
        
        assert game.players[0].score == 0
        assert game.state == GameState.WAITING_FOR_DART
    
    def test_around_the_clock_mode(self):
        """Test Around the Clock mode."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.AROUND_THE_CLOCK, [player])
        game.start_game()
        
        assert game.players[0].score == 0
        
        # Throw some darts
        field = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        game.record_dart(field)
        game.record_dart(field)
        game.record_dart(field)
        
        # Score should remain 0 (needs specialized logic)
        assert game.players[0].score == 0
    
    def test_training_mode_never_ends(self):
        """Test training mode has no win condition."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.TRAINING, [player])
        game.start_game()
        
        # Even with high score, should not win
        player.score = 9999
        assert game.check_win_condition(player) is False
    
    def test_game_summary_with_no_times(self):
        """Test game summary before game starts."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.GAME_301, [player])
        
        summary = game.get_game_summary()
        
        assert summary["start_time"] is None
        assert summary["end_time"] is None
    
    def test_game_summary_after_game_over(self):
        """Test game summary after game ends."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.GAME_301, [player])
        game.start_game()
        
        # Simulate quick win
        player.score = 0
        game.current_round.is_checkout = True
        game.end_game(player)
        
        summary = game.get_game_summary()
        
        assert summary["winner"] == "Test"
        assert summary["end_time"] is not None
    
    def test_record_dart_no_active_round(self):
        """Test recording dart when no active round."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.GAME_301, [player])
        game.start_game()
        
        # Force no active round
        game.current_round = None
        
        field = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        with pytest.raises(RuntimeError, match="No active round"):
            game.record_dart(field)
    
    def test_countdown_with_config_options(self):
        """Test countdown with configuration options."""
        player = Player(name="Test", id=1)
        config = {"double_out": False}  # Disable double-out
        game = GameEngine(GameMode.GAME_301, [player], config=config)
        game.start_game()
        
        # Set up for finish
        player.score = 20
        
        # Finish with single 20 (should work without double-out)
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        complete = game.record_dart(field)
        
        # With double_out=False, should be valid
        assert complete is True
        assert player.score == 0
        assert game.rounds[0].is_checkout is True
    
    def test_statistics_bulls_eye(self):
        """Test bull's eye statistics tracking."""
        player = Player(name="Test", id=1)
        game = GameEngine(GameMode.TRAINING, [player])
        game.start_game()
        
        field = DartboardField(segment=25, zone="bull_eye", score=50, multiplier=1)
        game.record_dart(field)
        
        assert player.statistics["bulls_hit"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
