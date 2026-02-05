# 🎮 Phase 4 – Game Engine & Logic

**Dependencies:** [Phase 3 – Calibration & Mapping](phase3-calibration.md) ✅  
**Next Phase:** [Phase 5 – UI & Polish](phase5-ui-polish.md)

---

## 🎯 Phase Goals

- Implement complete dart game rule engine
- Support multiple game modes (301/501/701, Cricket, Around the Clock)
- Build multiplayer state management
- Create statistics and scoring system
- Achieve 100% test coverage for all game logic

---

## 📋 Prerequisites

### From Phase 3
- ✅ DartboardField mapping functional
- ✅ Score calculation working
- ✅ Calibration persistence tested

### Phase 4 Requirements
- Clear understanding of dart rules
- State machine for game flow
- Persistent player data

---

## 🎲 Game Modes Overview

### 301/501/701 (Countdown Games)

**Rules:**
- Start with 301/501/701 points
- Subtract each dart score
- First to exactly 0 wins
- **Bust:** Score goes below 0 or exactly 1 → round score reset
- **Double-Out:** Must finish on a double
- **Master-Out:** Must finish on double or bull's eye
- **Double-In:** Must hit double to start scoring

### Cricket

**Rules:**
- Target numbers: 15-20 and Bull
- Hit each number 3 times to "close" it
- Score points on closed numbers if opponent hasn't closed
- First to close all numbers with most points wins

### Around the Clock

**Rules:**
- Hit segments 1-20 in order
- Optional: finish on bull's eye
- First to complete sequence wins

---

## 🔧 Implementation Tasks

### 4.1 Core Game Engine

**Task:** Base game state management

**File:** `src/game/game_engine.py`

```python
"""
Core game engine for dart scoring system.

Manages game state, player turns, scoring, and rule enforcement
across multiple game modes.

Author: Mario Neuhauser
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.calibration.board_mapper import DartboardField


logger = logging.getLogger(__name__)


class GameMode(Enum):
    """Available game modes."""
    GAME_301 = "301"
    GAME_501 = "501"
    GAME_701 = "701"
    CRICKET = "cricket"
    AROUND_THE_CLOCK = "around_the_clock"
    TRAINING = "training"


class GameState(Enum):
    """Game state machine states."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_DART = "waiting_for_dart"
    ROUND_COMPLETE = "round_complete"
    GAME_OVER = "game_over"
    PAUSED = "paused"


@dataclass
class Player:
    """
    Represents a player in the game.
    
    Attributes:
        name (str): Player name
        id (int): Unique player identifier
        score (int): Current score (meaning varies by game mode)
        darts_thrown (int): Total darts thrown
        rounds_played (int): Total rounds played
        statistics (Dict): Player statistics
    """
    name: str
    id: int
    score: int = 0
    darts_thrown: int = 0
    rounds_played: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default statistics."""
        if not self.statistics:
            self.statistics = {
                "total_score": 0,
                "highest_round": 0,
                "checkout_attempts": 0,
                "successful_checkouts": 0,
                "tons": 0,  # Rounds >= 100 points
                "ton_eighties": 0,  # Rounds >= 180 points
                "bulls_hit": 0,
                "triples_hit": 0,
                "doubles_hit": 0
            }


@dataclass
class DartThrow:
    """
    Represents a single dart throw result.
    
    Attributes:
        player_id (int): ID of player who threw
        field (DartboardField): Field hit
        round_number (int): Round number
        dart_number (int): Dart number in round (1-3)
        timestamp (datetime): When dart was thrown
    """
    player_id: int
    field: DartboardField
    round_number: int
    dart_number: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Round:
    """
    Represents a single round (3 darts).
    
    Attributes:
        player_id (int): Player who threw
        darts (List[DartThrow]): Up to 3 dart throws
        round_score (int): Total score for round
        is_bust (bool): Whether round was a bust
        is_checkout (bool): Whether round finished the game
    """
    player_id: int
    darts: List[DartThrow] = field(default_factory=list)
    round_score: int = 0
    is_bust: bool = False
    is_checkout: bool = False


class GameEngine:
    """
    Core game engine managing state, turns, and scoring.
    
    Handles game flow, rule enforcement, and state transitions
    for all supported game modes.
    
    Attributes:
        mode (GameMode): Current game mode
        players (List[Player]): Players in game
        state (GameState): Current game state
        current_player_index (int): Index of active player
        rounds (List[Round]): All completed rounds
        current_round (Round): Current round in progress
    """
    
    def __init__(
        self,
        mode: GameMode,
        players: List[Player],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize game engine.
        
        Args:
            mode: Game mode to play
            players: List of players (1-8 supported)
            config: Optional game configuration
        """
        if not players:
            raise ValueError("At least one player required")
        if len(players) > 8:
            raise ValueError("Maximum 8 players supported")
        
        self.mode = mode
        self.players = players
        self.config = config or {}
        self.state = GameState.NOT_STARTED
        self.current_player_index = 0
        self.rounds: List[Round] = []
        self.current_round: Optional[Round] = None
        self.winner: Optional[Player] = None
        self.game_start_time: Optional[datetime] = None
        self.game_end_time: Optional[datetime] = None
        
        logger.info(f"Game initialized: {mode.value} with {len(players)} players")
    
    def start_game(self) -> None:
        """
        Start the game.
        
        Initializes starting scores based on game mode and
        transitions to IN_PROGRESS state.
        """
        if self.state != GameState.NOT_STARTED:
            raise RuntimeError("Game already started")
        
        # Initialize player scores based on mode
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            starting_score = int(self.mode.value)
            for player in self.players:
                player.score = starting_score
        else:
            for player in self.players:
                player.score = 0
        
        self.state = GameState.IN_PROGRESS
        self.game_start_time = datetime.now()
        self.start_new_round()
        
        logger.info(f"Game started at {self.game_start_time}")
    
    def start_new_round(self) -> None:
        """Start a new round for current player."""
        current_player = self.get_current_player()
        self.current_round = Round(player_id=current_player.id)
        self.state = GameState.WAITING_FOR_DART
        
        logger.debug(f"New round started for {current_player.name}")
    
    def get_current_player(self) -> Player:
        """
        Get currently active player.
        
        Returns:
            Current Player object
        """
        return self.players[self.current_player_index]
    
    def record_dart(self, field: DartboardField) -> bool:
        """
        Record a dart throw for current player.
        
        Args:
            field: Dartboard field hit
            
        Returns:
            True if round is complete, False if more darts remain
            
        Raises:
            RuntimeError: If game not in correct state
        """
        if self.state != GameState.WAITING_FOR_DART:
            raise RuntimeError(f"Cannot record dart in state: {self.state}")
        
        if self.current_round is None:
            raise RuntimeError("No active round")
        
        player = self.get_current_player()
        dart_number = len(self.current_round.darts) + 1
        round_number = len([r for r in self.rounds if r.player_id == player.id]) + 1
        
        # Create dart throw
        dart_throw = DartThrow(
            player_id=player.id,
            field=field,
            round_number=round_number,
            dart_number=dart_number
        )
        
        self.current_round.darts.append(dart_throw)
        player.darts_thrown += 1
        
        # Update statistics
        self.update_player_statistics(player, field)
        
        logger.info(f"{player.name} threw dart {dart_number}: {field}")
        
        # Check if round is complete (3 darts or game-specific conditions)
        if dart_number >= 3:
            self.complete_round()
            return True
        
        return False
    
    def update_player_statistics(self, player: Player, field: DartboardField) -> None:
        """
        Update player statistics based on dart throw.
        
        Args:
            player: Player who threw
            field: Field hit
        """
        stats = player.statistics
        
        stats["total_score"] += field.score
        
        if field.zone == "bull_eye" or field.zone == "bull":
            stats["bulls_hit"] += 1
        elif field.zone == "triple":
            stats["triples_hit"] += 1
        elif field.zone == "double":
            stats["doubles_hit"] += 1
    
    def complete_round(self) -> None:
        """
        Complete current round and process scoring.
        
        Applies game-specific rules and checks for win conditions.
        """
        if self.current_round is None:
            return
        
        player = self.get_current_player()
        
        # Calculate round score
        round_score = sum(dart.field.score for dart in self.current_round.darts)
        self.current_round.round_score = round_score
        
        # Apply game-specific rules
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            self.process_countdown_round(player, round_score)
        elif self.mode == GameMode.CRICKET:
            self.process_cricket_round(player)
        elif self.mode == GameMode.AROUND_THE_CLOCK:
            self.process_around_clock_round(player)
        else:  # Training mode
            player.score += round_score
        
        # Update player statistics
        player.rounds_played += 1
        if round_score > player.statistics["highest_round"]:
            player.statistics["highest_round"] = round_score
        
        if round_score >= 100:
            player.statistics["tons"] += 1
        if round_score >= 180:
            player.statistics["ton_eighties"] += 1
        
        # Store round
        self.rounds.append(self.current_round)
        
        # Check win condition
        if self.check_win_condition(player):
            self.end_game(player)
        else:
            self.state = GameState.ROUND_COMPLETE
            self.next_player()
    
    def process_countdown_round(self, player: Player, round_score: int) -> None:
        """
        Process round for countdown games (301/501/701).
        
        Handles bust logic, double-out, and checkout rules.
        
        Args:
            player: Player who completed round
            round_score: Total score for round
        """
        config = self.config
        double_out = config.get("double_out", True)
        double_in = config.get("double_in", False)
        master_out = config.get("master_out", False)
        
        # Check double-in requirement
        if double_in and player.score == int(self.mode.value):
            # Player hasn't scored yet, need double to start
            has_double = any(
                dart.field.zone == "double"
                for dart in self.current_round.darts
            )
            if not has_double:
                self.current_round.is_bust = True
                logger.info(f"{player.name} bust: Double-in required")
                return
        
        new_score = player.score - round_score
        
        # Check bust conditions
        is_bust = False
        
        if new_score < 0:
            is_bust = True
            logger.info(f"{player.name} bust: Score went below 0")
        elif new_score == 1:
            is_bust = True
            logger.info(f"{player.name} bust: Cannot finish on 1")
        elif new_score == 0:
            # Potential checkout
            last_dart = self.current_round.darts[-1]
            
            if double_out and last_dart.field.zone != "double":
                is_bust = True
                logger.info(f"{player.name} bust: Double-out required")
            elif master_out and last_dart.field.zone not in ["double", "bull_eye"]:
                is_bust = True
                logger.info(f"{player.name} bust: Master-out required")
            else:
                # Valid checkout!
                self.current_round.is_checkout = True
                player.score = 0
                player.statistics["successful_checkouts"] += 1
                logger.info(f"{player.name} checkout! Game won!")
                return
        
        if is_bust:
            self.current_round.is_bust = True
            # Score doesn't change on bust
        else:
            player.score = new_score
    
    def process_cricket_round(self, player: Player) -> None:
        """
        Process round for Cricket game mode.
        
        Args:
            player: Player who completed round
        """
        # Cricket implementation would track hits on 15-20 and bull
        # Simplified for now
        player.score += sum(dart.field.score for dart in self.current_round.darts)
    
    def process_around_clock_round(self, player: Player) -> None:
        """
        Process round for Around the Clock mode.
        
        Args:
            player: Player who completed round
        """
        # Track progress through segments 1-20
        # Simplified for now
        for dart in self.current_round.darts:
            if dart.field.segment == player.score + 1:
                player.score += 1
    
    def check_win_condition(self, player: Player) -> bool:
        """
        Check if player has won the game.
        
        Args:
            player: Player to check
            
        Returns:
            True if player won, False otherwise
        """
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            return player.score == 0 and self.current_round.is_checkout
        elif self.mode == GameMode.AROUND_THE_CLOCK:
            return player.score >= 20
        # Add other mode win conditions
        return False
    
    def next_player(self) -> None:
        """Advance to next player and start their round."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.start_new_round()
    
    def end_game(self, winner: Player) -> None:
        """
        End the game with a winner.
        
        Args:
            winner: Player who won
        """
        self.winner = winner
        self.state = GameState.GAME_OVER
        self.game_end_time = datetime.now()
        
        duration = (self.game_end_time - self.game_start_time).total_seconds()
        logger.info(f"Game over! Winner: {winner.name} ({duration:.1f}s)")
    
    def pause_game(self) -> None:
        """Pause the game."""
        if self.state == GameState.IN_PROGRESS or self.state == GameState.WAITING_FOR_DART:
            self.state = GameState.PAUSED
            logger.info("Game paused")
    
    def resume_game(self) -> None:
        """Resume a paused game."""
        if self.state == GameState.PAUSED:
            self.state = GameState.WAITING_FOR_DART
            logger.info("Game resumed")
    
    def get_game_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive game summary.
        
        Returns:
            Dictionary with game statistics and state
        """
        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "players": [
                {
                    "name": p.name,
                    "score": p.score,
                    "darts_thrown": p.darts_thrown,
                    "rounds_played": p.rounds_played,
                    "average": (p.statistics["total_score"] / p.darts_thrown) if p.darts_thrown > 0 else 0,
                    "statistics": p.statistics
                }
                for p in self.players
            ],
            "current_player": self.get_current_player().name if self.state != GameState.GAME_OVER else None,
            "winner": self.winner.name if self.winner else None,
            "total_rounds": len(self.rounds),
            "duration_seconds": (
                (self.game_end_time - self.game_start_time).total_seconds()
                if self.game_end_time and self.game_start_time
                else None
            )
        }
```

**Expected Outcome:** Fully functional game engine with state management

---

### 4.2 Score Calculator Module

**Task:** Specialized scoring logic for each game mode

**File:** `src/game/score_calculator.py`

```python
"""
Score calculation utilities for different game modes.

Provides specialized scoring logic, checkout hints, and
optimal finish calculations.

Author: Mario Neuhauser
"""

from typing import List, Optional, Tuple
from src.calibration.board_mapper import DartboardField


class ScoreCalculator:
    """
    Advanced score calculation and game strategy utilities.
    
    Provides checkout hints, optimal finishes, and score validation
    for competitive dart gameplay.
    """
    
    # Possible checkout scores with recommended finish
    CHECKOUT_TABLE = {
        170: [(20, "triple"), (20, "triple"), (25, "bull_eye")],
        167: [(20, "triple"), (19, "triple"), (25, "bull_eye")],
        164: [(20, "triple"), (18, "triple"), (25, "bull_eye")],
        # ... (complete checkout table would be extensive)
        50: [(25, "bull_eye")],
        40: [(20, "double")],
        32: [(16, "double")],
        # Add common checkouts
    }
    
    @staticmethod
    def calculate_average(total_score: int, darts_thrown: int) -> float:
        """
        Calculate player's average score per dart.
        
        Args:
            total_score: Total points scored
            darts_thrown: Total darts thrown
            
        Returns:
            Average score per dart
        """
        if darts_thrown == 0:
            return 0.0
        return total_score / darts_thrown
    
    @staticmethod
    def calculate_three_dart_average(total_score: int, darts_thrown: int) -> float:
        """
        Calculate average per 3-dart round.
        
        Args:
            total_score: Total points scored
            darts_thrown: Total darts thrown
            
        Returns:
            Average per 3-dart round
        """
        return ScoreCalculator.calculate_average(total_score, darts_thrown) * 3
    
    @staticmethod
    def is_valid_checkout(remaining: int) -> bool:
        """
        Check if a score is theoretically checkable.
        
        Args:
            remaining: Points remaining
            
        Returns:
            True if checkout is possible
        """
        # Cannot checkout on 1 (no double-0.5)
        # Cannot checkout above 170 (max: T20 + T20 + Bull)
        if remaining == 1:
            return False
        if remaining > 170:
            return False
        return True
    
    @staticmethod
    def suggest_checkout(remaining: int) -> Optional[List[Tuple[int, str]]]:
        """
        Suggest optimal checkout combination.
        
        Args:
            remaining: Points remaining
            
        Returns:
            List of (segment, zone) tuples for optimal finish, or None
        """
        return ScoreCalculator.CHECKOUT_TABLE.get(remaining)
    
    @staticmethod
    def calculate_outs_percentage(
        checkout_attempts: int,
        successful_checkouts: int
    ) -> float:
        """
        Calculate checkout success percentage.
        
        Args:
            checkout_attempts: Number of checkout attempts
            successful_checkouts: Number of successful checkouts
            
        Returns:
            Percentage (0-100)
        """
        if checkout_attempts == 0:
            return 0.0
        return (successful_checkouts / checkout_attempts) * 100
    
    @staticmethod
    def is_bogey_number(score: int) -> bool:
        """
        Check if score is a "bogey number" (difficult to checkout).
        
        Bogey numbers: 169, 168, 166, 165, 163, 162, 159
        
        Args:
            score: Score to check
            
        Returns:
            True if bogey number
        """
        bogey_numbers = [169, 168, 166, 165, 163, 162, 159]
        return score in bogey_numbers
    
    @staticmethod
    def maximum_possible_score(darts_remaining: int) -> int:
        """
        Calculate maximum possible score with remaining darts.
        
        Args:
            darts_remaining: Number of darts left
            
        Returns:
            Maximum achievable score
        """
        # Triple 20 is highest single dart score
        return darts_remaining * 60


def calculate_required_average(current_score: int, darts_remaining: int, target: int = 0) -> float:
    """
    Calculate required average to finish.
    
    Args:
        current_score: Current player score
        darts_remaining: Darts left in match
        target: Target score (0 for countdown games)
        
    Returns:
        Required average per dart
    """
    points_needed = current_score - target
    if darts_remaining == 0:
        return float('inf')
    return points_needed / darts_remaining
```

**Expected Outcome:** Complete scoring utilities with checkout logic

---

### 4.3 Unit Tests for Game Engine

**Task:** Comprehensive game logic tests

**File:** `tests/unit/test_game_engine.py`

```python
"""
Unit tests for game engine.

Tests all game modes, state transitions, scoring, and rule enforcement.
Coverage Target: 100%

Author: Mario Neuhauser
"""

import pytest
from datetime import datetime

from src.game.game_engine import (
    GameEngine, GameMode, GameState, Player, DartThrow, Round
)
from src.calibration.board_mapper import DartboardField


@pytest.fixture
def players():
    """Create test players."""
    return [
        Player(name="Alice", id=1),
        Player(name="Bob", id=2)
    ]


@pytest.fixture
def game_301(players):
    """Create 301 game."""
    return GameEngine(GameMode.GAME_301, players, {"double_out": True})


class TestPlayer:
    """Tests for Player dataclass."""
    
    def test_player_creation(self):
        """Test player initialization."""
        player = Player(name="Test", id=1)
        assert player.name == "Test"
        assert player.score == 0
        assert player.darts_thrown == 0
        assert "total_score" in player.statistics
    
    def test_player_statistics_initialized(self):
        """Test default statistics are created."""
        player = Player(name="Test", id=1)
        assert player.statistics["tons"] == 0
        assert player.statistics["triples_hit"] == 0


class TestGameEngine:
    """Tests for GameEngine class."""
    
    def test_initialization(self, players):
        """Test game engine initialization."""
        game = GameEngine(GameMode.GAME_501, players)
        assert game.mode == GameMode.GAME_501
        assert len(game.players) == 2
        assert game.state == GameState.NOT_STARTED
    
    def test_initialization_no_players(self):
        """Test error with no players."""
        with pytest.raises(ValueError, match="At least one player"):
            GameEngine(GameMode.GAME_301, [])
    
    def test_initialization_too_many_players(self):
        """Test error with too many players."""
        too_many = [Player(f"Player{i}", i) for i in range(10)]
        with pytest.raises(ValueError, match="Maximum 8 players"):
            GameEngine(GameMode.GAME_301, too_many)
    
    def test_start_game_301(self, game_301):
        """Test starting 301 game."""
        game_301.start_game()
        
        assert game_301.state == GameState.WAITING_FOR_DART
        assert all(p.score == 301 for p in game_301.players)
        assert game_301.game_start_time is not None
    
    def test_start_game_already_started(self, game_301):
        """Test error when starting already started game."""
        game_301.start_game()
        with pytest.raises(RuntimeError, match="already started"):
            game_301.start_game()
    
    def test_get_current_player(self, game_301):
        """Test getting current player."""
        game_301.start_game()
        player = game_301.get_current_player()
        assert player.name == "Alice"
    
    def test_record_dart(self, game_301):
        """Test recording a dart throw."""
        game_301.start_game()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        is_complete = game_301.record_dart(field)
        
        assert not is_complete  # First dart of round
        assert len(game_301.current_round.darts) == 1
        assert game_301.players[0].darts_thrown == 1
    
    def test_record_three_darts_completes_round(self, game_301):
        """Test that 3 darts completes a round."""
        game_301.start_game()
        
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        
        game_301.record_dart(field)
        game_301.record_dart(field)
        is_complete = game_301.record_dart(field)
        
        assert is_complete
        assert game_301.players[0].score == 301 - 60
    
    def test_record_dart_invalid_state(self, game_301):
        """Test error when recording dart in wrong state."""
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        
        with pytest.raises(RuntimeError, match="Cannot record dart"):
            game_301.record_dart(field)
    
    def test_update_statistics_triple(self, game_301):
        """Test statistics update for triple."""
        game_301.start_game()
        player = game_301.get_current_player()
        
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        
        assert player.statistics["triples_hit"] == 1
        assert player.statistics["total_score"] == 60
    
    def test_countdown_scoring(self, game_301):
        """Test countdown game scoring."""
        game_301.start_game()
        
        # Throw 60 points
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert game_301.players[0].score == 301 - 180
    
    def test_bust_below_zero(self, game_301):
        """Test bust when going below zero."""
        game_301.start_game()
        game_301.players[0].score = 20  # Set low score
        
        # Throw 60 points (would go to -40)
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert game_301.current_round.is_bust
        assert game_301.players[0].score == 20  # Score unchanged
    
    def test_bust_on_one(self, game_301):
        """Test bust when landing on exactly 1."""
        game_301.start_game()
        game_301.players[0].score = 21
        
        # Throw 20 (would leave 1)
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert game_301.current_round.is_bust
        assert game_301.players[0].score == 21
    
    def test_valid_checkout_double_out(self, game_301):
        """Test valid checkout with double-out."""
        game_301.start_game()
        game_301.players[0].score = 40
        
        # Throw double 20 (40 points)
        field = DartboardField(segment=20, zone="double", score=40, multiplier=2)
        game_301.record_dart(field)
        game_301.record_dart(DartboardField(0, "miss", 0, 0))
        game_301.record_dart(DartboardField(0, "miss", 0, 0))
        
        assert game_301.current_round.is_checkout
        assert game_301.players[0].score == 0
        assert game_301.state == GameState.GAME_OVER
        assert game_301.winner == game_301.players[0]
    
    def test_invalid_checkout_no_double(self, game_301):
        """Test bust when trying to finish without double."""
        game_301.start_game()
        game_301.players[0].score = 20
        
        # Throw single 20 (not double)
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(DartboardField(0, "miss", 0, 0))
        game_301.record_dart(DartboardField(0, "miss", 0, 0))
        
        assert game_301.current_round.is_bust
        assert game_301.players[0].score == 20
    
    def test_player_rotation(self, game_301):
        """Test players alternate turns."""
        game_301.start_game()
        
        assert game_301.get_current_player().name == "Alice"
        
        # Complete Alice's round
        field = DartboardField(segment=20, zone="single", score=20, multiplier=1)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        # Should be Bob's turn
        assert game_301.get_current_player().name == "Bob"
    
    def test_pause_resume(self, game_301):
        """Test pausing and resuming game."""
        game_301.start_game()
        
        game_301.pause_game()
        assert game_301.state == GameState.PAUSED
        
        game_301.resume_game()
        assert game_301.state == GameState.WAITING_FOR_DART
    
    def test_game_summary(self, game_301):
        """Test game summary generation."""
        game_301.start_game()
        
        summary = game_301.get_game_summary()
        
        assert summary["mode"] == "301"
        assert summary["state"] == "waiting_for_dart"
        assert len(summary["players"]) == 2
        assert summary["current_player"] == "Alice"
    
    def test_statistics_tracking(self, game_301):
        """Test comprehensive statistics tracking."""
        game_301.start_game()
        player = game_301.players[0]
        
        # Throw 180 (3x triple 20)
        field = DartboardField(segment=20, zone="triple", score=60, multiplier=3)
        game_301.record_dart(field)
        game_301.record_dart(field)
        game_301.record_dart(field)
        
        assert player.statistics["ton_eighties"] == 1
        assert player.statistics["tons"] == 1
        assert player.statistics["highest_round"] == 180


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.game", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
pytest tests/unit/test_game_engine.py -v --cov=src.game --cov-report=term-missing
```

**Expected Coverage:** 100%

---

## ⚠️ Phase 4 Risks & Mitigations

### Risk: Complex Rule Edge Cases

**Symptoms:** Unexpected game state, incorrect scoring

**Mitigation:**
1. Comprehensive unit tests for all rule combinations
2. State machine validation
3. Edge case documentation
4. Real-world play testing

---

## ✅ Phase 4 Completion Checklist

### Code Implementation
- [ ] `src/game/game_engine.py` complete (100% coverage)
- [ ] `src/game/score_calculator.py` complete
- [ ] All game modes implemented
- [ ] State machine validated

### Testing
- [ ] Unit tests pass (100% coverage)
- [ ] All game modes tested
- [ ] Bust logic tested (all scenarios)
- [ ] Checkout logic tested
- [ ] Multiplayer rotation tested
- [ ] Statistics tracking tested

### Game Modes
- [ ] 301/501/701 fully functional
- [ ] Double-out working
- [ ] Master-out working
- [ ] Double-in working
- [ ] Cricket implemented
- [ ] Around the Clock implemented

---

## 📊 Expected Coverage Report

```
src/game/game_engine.py              567      0   100%
src/game/score_calculator.py         143      0   100%
tests/unit/test_game_engine.py       489      0   100%
------------------------------------------------------------
TOTAL                               1199      0   100%
```

---

## 🔗 Next Steps

Once Phase 4 is complete with 100% coverage:

**Proceed to:** [Phase 5 – UI & Polish](phase5-ui-polish.md)

**Phase 5 Requirements:**
- ✅ Game engine functional
- ✅ All game modes working
- ✅ Statistics calculated

---

**Phase Status:** 🔴 Not Started  
**Estimated Duration:** 2 weeks  
**Test Coverage:** 0% → Target: 100%  
**Dependencies:** Phase 3 ✅
