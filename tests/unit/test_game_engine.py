"""
Unit tests for game engine core functionality.

Tests player management, game flow, state transitions, and scoring.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
from datetime import datetime

from src.game.game_engine import (
    GameEngine, GameMode, GameState, Player,
    DartThrow, Round
)
from src.calibration.board_mapper import DartboardField


class TestPlayer:
    """Tests for Player dataclass."""
    
    def test_player_creation(self):
        """Test player initialization."""
        player = Player(name="Alice", id=1)
        assert player.name == "Alice"
        assert player.id == 1
        assert player.score == 0
        assert player.darts_thrown == 0
        assert player.rounds_played == 0
    
    def test_player_statistics_initialization(self):
        """Test player statistics are initialized."""
        player = Player(name="Bob", id=2)
        assert "total_score" in player.statistics
        assert "highest_round" in player.statistics
        assert "bulls_hit" in player.statistics
        assert player.statistics["triples_hit"] == 0
    
    def test_player_with_custom_statistics(self):
        """Test player with pre-existing statistics."""
        stats = {"total_score": 100, "custom_stat": 42}
        player = Player(name="Charlie", id=3, statistics=stats)
        assert player.statistics["total_score"] == 100
        assert player.statistics["custom_stat"] == 42


class TestDartThrow:
    """Tests for DartThrow dataclass."""
    
    def test_dart_throw_creation(self):
        """Test dart throw initialization."""
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        dart = DartThrow(
            player_id=1,
            field=field,
            round_number=1,
            dart_number=1
        )
        assert dart.player_id == 1
        assert dart.field.score == 60
        assert dart.round_number == 1
        assert dart.dart_number == 1
        assert isinstance(dart.timestamp, datetime)


class TestRound:
    """Tests for Round dataclass."""
    
    def test_round_creation(self):
        """Test round initialization."""
        round_obj = Round(player_id=1)
        assert round_obj.player_id == 1
        assert len(round_obj.darts) == 0
        assert round_obj.round_score == 0
        assert round_obj.is_bust is False
        assert round_obj.is_checkout is False


class TestGameEngine:
    """Tests for GameEngine class."""
    
    @pytest.fixture
    def players(self):
        """Create test players."""
        return [
            Player(name="Alice", id=1),
            Player(name="Bob", id=2)
        ]
    
    @pytest.fixture
    def game_301(self, players):
        """Create 301 game."""
        return GameEngine(GameMode.GAME_301, players)
    
    def test_initialization_single_player(self):
        """Test game initialization with one player."""
        player = Player(name="Solo", id=1)
        game = GameEngine(GameMode.GAME_501, [player])
        assert game.mode == GameMode.GAME_501
        assert len(game.players) == 1
        assert game.state == GameState.NOT_STARTED
    
    def test_initialization_multiple_players(self, players):
        """Test game initialization with multiple players."""
        game = GameEngine(GameMode.GAME_301, players)
        assert len(game.players) == 2
        assert game.current_player_index == 0
    
    def test_initialization_no_players(self):
        """Test error when no players provided."""
        with pytest.raises(ValueError, match="At least one player required"):
            GameEngine(GameMode.GAME_301, [])
    
    def test_initialization_too_many_players(self):
        """Test error when too many players."""
        players = [Player(name=f"P{i}", id=i) for i in range(9)]
        with pytest.raises(ValueError, match="Maximum 8 players"):
            GameEngine(GameMode.GAME_301, players)
    
    def test_start_game_301(self, game_301):
        """Test starting 301 game."""
        game_301.start_game()
        
        assert game_301.state == GameState.WAITING_FOR_DART
        assert game_301.players[0].score == 301
        assert game_301.players[1].score == 301
        assert game_301.game_start_time is not None
        assert game_301.current_round is not None
    
    def test_start_game_501(self, players):
        """Test starting 501 game."""
        game = GameEngine(GameMode.GAME_501, players)
        game.start_game()
        
        assert game.players[0].score == 501
        assert game.players[1].score == 501
    
    def test_start_game_training(self, players):
        """Test starting training mode."""
        game = GameEngine(GameMode.TRAINING, players)
        game.start_game()
        
        assert game.players[0].score == 0
        assert game.players[1].score == 0
    
    def test_start_game_already_started(self, game_301):
        """Test error when starting already-started game."""
        game_301.start_game()
        
        with pytest.raises(RuntimeError, match="already started"):
            game_301.start_game()
    
    def test_get_current_player(self, game_301):
        """Test getting current player."""
        game_301.start_game()
        player = game_301.get_current_player()
        assert player.name == "Alice"
        assert player.id == 1
    
    def test_record_dart_single(self, game_301):
        """Test recording a single dart."""
        game_301.start_game()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        complete = game_301.record_dart(field)
        
        assert complete is False
        assert len(game_301.current_round.darts) == 1
        assert game_301.players[0].darts_thrown == 1
    
    def test_record_dart_complete_round(self, game_301):
        """Test completing a round with 3 darts."""
        game_301.start_game()
        
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        
        game_301.record_dart(field)
        game_301.record_dart(field)
        complete = game_301.record_dart(field)
        
        assert complete is True
        assert len(game_301.rounds) == 1
        assert game_301.rounds[0].round_score == 60
        assert game_301.players[0].score == 241  # 301 - 60
    
    def test_record_dart_wrong_state(self, game_301):
        """Test error when recording dart in wrong state."""
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        
        with pytest.raises(RuntimeError, match="Cannot record dart"):
            game_301.record_dart(field)
    
    def test_update_player_statistics_bull(self, game_301):
        """Test statistics update for bull hit."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=25, zone="bull", score=25, multiplier=1)
        game_301.update_player_statistics(player, field)
        
        assert player.statistics["bulls_hit"] == 1
        assert player.statistics["total_score"] == 25
    
    def test_update_player_statistics_triple(self, game_301):
        """Test statistics update for triple hit."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.update_player_statistics(player, field)
        
        assert player.statistics["triples_hit"] == 1
    
    def test_update_player_statistics_double(self, game_301):
        """Test statistics update for double hit."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="double", score=40, multiplier=2)
        game_301.update_player_statistics(player, field)
        
        assert player.statistics["doubles_hit"] == 1
    
    def test_countdown_round_simple(self, game_301):
        """Test simple countdown scoring."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert player.score == 241  # 301 - 60
    
    def test_countdown_bust_below_zero(self, game_301):
        """Test bust when score goes below zero."""
        game_301.start_game()
        player = game_301.get_current_player()
        player.score = 50  # Set up for bust
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        # Bust: 50 - 180 = -130
        assert player.score == 50  # Score unchanged
        assert game_301.rounds[0].is_bust is True
    
    def test_countdown_bust_exactly_one(self, game_301):
        """Test bust when score is exactly 1."""
        game_301.start_game()
        player = game_301.get_current_player()
        player.score = 3  # Will leave 1 after single 2
        
        field = DartboardField(segment=1, zone="double", score=2, multiplier=2)
        game_301.record_dart(field)
        field2 = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        game_301.record_dart(field2)
        game_301.record_dart(field2)
        
        # Bust: 3 - 4 would be -1, but after first dart it's 1 (impossible)
        assert player.score == 3
        assert game_301.rounds[0].is_bust is True
    
    def test_countdown_double_out_requirement(self, game_301):
        """Test valid double-out checkout."""
        game_301.start_game()
        player = game_301.get_current_player()
        player.score = 40
        
        # Finish with double 20 (valid checkout)
        field = DartboardField(segment=20, zone="double", score=40, multiplier=2)
        complete = game_301.record_dart(field)
        
        # Should complete immediately and finish the game
        assert complete is True
        assert player.score == 0
        assert game_301.rounds[0].is_checkout is True
        assert game_301.state == GameState.GAME_OVER
        assert game_301.winner == player
    
    def test_countdown_invalid_double_out(self, game_301):
        """Test invalid checkout with non-double."""
        game_301.start_game()
        player = game_301.get_current_player()
        player.score = 20
        
        # Try to finish with single 20 (should bust)
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        complete = game_301.record_dart(field)
        
        # Should complete immediately but be a bust
        assert complete is True
        assert player.score == 20  # Score unchanged
        assert game_301.rounds[0].is_bust is True
    
    def test_training_mode_scoring(self, players):
        """Test training mode accumulates score."""
        game = GameEngine(GameMode.TRAINING, players)
        game.start_game()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game.record_dart(field)
        game.record_dart(field)
        game.record_dart(field)
        
        assert game.players[0].score == 180
    
    def test_check_win_condition_301_not_won(self, game_301):
        """Test win condition not met."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        assert game_301.check_win_condition(player) is False
    
    def test_check_win_condition_301_won(self, game_301):
        """Test win condition met for 301."""
        game_301.start_game()
        player = game_301.get_current_player()
        player.score = 0
        game_301.current_round.is_checkout = True
        
        assert game_301.check_win_condition(player) is True
    
    def test_end_game(self, game_301):
        """Test ending the game."""
        game_301.start_game()
        winner = game_301.players[0]
        
        game_301.end_game(winner)
        
        assert game_301.winner == winner
        assert game_301.state == GameState.GAME_OVER
        assert game_301.game_end_time is not None
    
    def test_next_player(self, game_301):
        """Test advancing to next player."""
        game_301.start_game()
        
        assert game_301.current_player_index == 0
        
        field = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        # Should advance to player 2
        assert game_301.current_player_index == 1
        assert game_301.get_current_player().name == "Bob"
    
    def test_player_rotation(self, game_301):
        """Test player rotation wraps around."""
        game_301.start_game()
        
        field = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        
        # Player 1
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        # Player 2
        assert game_301.current_player_index == 1
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        # Should wrap back to Player 1
        assert game_301.current_player_index == 0
    
    def test_round_statistics_ton(self, game_301):
        """Test ton (100+) statistics tracking."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        field2 = DartboardField(segment=1, zone="single", score=1, multiplier=1)
        game_301.record_dart(field2)
        
        assert player.statistics["tons"] == 1
        assert player.statistics["highest_round"] == 121
    
    def test_round_statistics_ton_eighty(self, game_301):
        """Test 180 statistics tracking."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert player.statistics["ton_eighties"] == 1
    
    def test_get_game_summary(self, game_301):
        """Test game summary generation."""
        game_301.start_game()
        
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        summary = game_301.get_game_summary()
        
        assert summary["mode"] == "301"
        assert summary["state"] == "waiting_for_dart"
        assert len(summary["players"]) == 2
        assert summary["players"][0]["name"] == "Alice"
        assert summary["rounds_count"] == 1
        assert summary["winner"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.game", "--cov-report=term-missing"])
