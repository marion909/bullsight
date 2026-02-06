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
        name: Player name
        id: Unique player identifier
        score: Current score (meaning varies by game mode)
        darts_thrown: Total darts thrown
        rounds_played: Total rounds played
        statistics: Player statistics dictionary
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
        player_id: ID of player who threw
        field: Field hit
        round_number: Round number
        dart_number: Dart number in round (1-3)
        timestamp: When dart was thrown
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
        player_id: Player who threw
        darts: Up to 3 dart throws
        round_score: Total score for round
        is_bust: Whether round was a bust
        is_checkout: Whether round finished the game
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
        mode: Current game mode
        players: Players in game
        state: Current game state
        current_player_index: Index of active player
        rounds: All completed rounds
        current_round: Current round in progress
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
            
        Raises:
            ValueError: If player count invalid
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
        
        Raises:
            RuntimeError: If game already started
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
        
        # For countdown games, check if this dart finishes the game
        early_finish = False
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            new_score = player.score - field.score
            if new_score == 0:
                # Potential checkout - complete round immediately
                early_finish = True
        
        # Check if round is complete (3 darts or early finish)
        if dart_number >= 3 or early_finish:
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
    
    def is_master_field(self, field: DartboardField) -> bool:
        """
        Check if field is a master (Bull or Double).
        
        Args:
            field: Field to check
            
        Returns:
            True if field is Bull's Eye, Bull, or any Double
        """
        return (field.zone == "bull_eye" or 
                field.zone == "bull" or 
                field.zone == "double")
    
    def complete_round(self) -> None:
        """
        Complete current round and process scoring.
        
        Applies game-specific rules and checks for win conditions.
        """
        if self.current_round is None:
            return
        
        player = self.get_current_player()
        
        # Check if game has started for this player (for In-rules)
        player_started = player.score > 0 or len([r for r in self.rounds if r.player_id == player.id]) > 0
        
        # Check Double In / Master In requirements for first scoring round
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            if not player_started and len(self.current_round.darts) > 0:
                double_in = self.config.get("double_in", False)
                master_in = self.config.get("master_in", False)
                
                if double_in or master_in:
                    # Check first scoring dart
                    valid_start = False
                    for dart in self.current_round.darts:
                        if dart.field.score > 0:  # Found first scoring dart
                            if master_in:
                                valid_start = self.is_master_field(dart.field)
                            elif double_in:
                                valid_start = dart.field.multiplier == 2
                            break
                    
                    if not valid_start:
                        # Invalid start - no score this round
                        self.current_round.is_bust = True
                        logger.info(f"{player.name} must start with {'Master' if master_in else 'Double'}!")
                        # Store round but don't score
                        self.rounds.append(self.current_round)
                        self.state = GameState.ROUND_COMPLETE
                        self.next_player()
                        return
        
        # Calculate round score
        round_score = sum(dart.field.score for dart in self.current_round.darts)
        self.current_round.round_score = round_score
        
        # Apply game-specific rules via strategy pattern
        self.apply_game_rules(player, round_score)
        
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
    
    def apply_game_rules(self, player: Player, round_score: int) -> None:
        """
        Apply game-specific scoring rules.
        
        Args:
            player: Player who completed round
            round_score: Total score for round
        """
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            self.process_countdown_round(player, round_score)
        elif self.mode == GameMode.TRAINING:
            player.score += round_score
        # Cricket and Around-the-Clock require field-by-field processing
        # and are handled separately via specialized classes
    
    def process_countdown_round(self, player: Player, round_score: int) -> None:
        """
        Process round for countdown games (301/501/701).
        
        Handles bust logic, double-out, master-out, and checkout rules.
        
        Args:
            player: Player who completed round
            round_score: Total score for round
        """
        config = self.config
        double_out = config.get("double_out", True)
        master_out = config.get("master_out", False)
        
        new_score = player.score - round_score
        
        # Check for bust conditions
        if new_score < 0 or new_score == 1:
            # Bust: score goes negative or exactly 1 (impossible to finish)
            self.current_round.is_bust = True
            logger.info(f"{player.name} bust! Score remains {player.score}")
            return
        
        # Check double-out or master-out requirement
        if new_score == 0:
            last_dart = self.current_round.darts[-1]
            
            if master_out:
                # Must finish on Bull or Double
                if not self.is_master_field(last_dart.field):
                    self.current_round.is_bust = True
                    logger.info(f"{player.name} must finish on Master (Bull or Double)! Score remains {player.score}")
                    return
            elif double_out:
                # Must finish on double
                if last_dart.field.multiplier != 2:
                    self.current_round.is_bust = True
                    logger.info(f"{player.name} must finish on double! Score remains {player.score}")
                    return
            
            # Valid checkout
            self.current_round.is_checkout = True
            player.statistics["successful_checkouts"] += 1
        
        # Valid score
        player.score = new_score
        logger.info(f"{player.name} score now {new_score}")
    
    def check_win_condition(self, player: Player) -> bool:
        """
        Check if player has won the game.
        
        Args:
            player: Player to check
            
        Returns:
            True if player has won
        """
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            return player.score == 0 and self.current_round.is_checkout
        elif self.mode == GameMode.TRAINING:
            return False  # Training never ends
        
        return False
    
    def undo_round(self) -> None:
        """
        Undo the current round (for score correction).
        
        Reverts all changes made in the current round including:
        - Player score
        - Darts thrown count
        - Round statistics
        """
        if self.current_round is None or len(self.current_round.darts) == 0:
            logger.warning("No round to undo")
            return
        
        player = self.get_current_player()
        
        # Revert player score
        if self.mode in [GameMode.GAME_301, GameMode.GAME_501, GameMode.GAME_701]:
            # For countdown games, add back the score that was subtracted
            round_score = sum(dart.field.score for dart in self.current_round.darts)
            player.score += round_score
            logger.info(f"Reverted {player.name} score to {player.score}")
        
        # Revert darts thrown count
        player.darts_thrown -= len(self.current_round.darts)
        
        # Revert statistics
        for dart in self.current_round.darts:
            field = dart.field
            stats = player.statistics
            
            stats["total_score"] -= field.score
            
            if field.zone == "bull_eye" or field.zone == "bull":
                stats["bulls_hit"] = max(0, stats["bulls_hit"] - 1)
            elif field.zone == "triple":
                stats["triples_hit"] = max(0, stats["triples_hit"] - 1)
            elif field.zone == "double":
                stats["doubles_hit"] = max(0, stats["doubles_hit"] - 1)
        
        # Clear current round
        self.current_round = Round(player_id=player.id, round_number=player.rounds_played)
        
        logger.info(f"Undone round for {player.name}")
    
    def end_game(self, winner: Player) -> None:
        """
        End the game with a winner.
        
        Args:
            winner: Player who won
        """
        self.winner = winner
        self.state = GameState.GAME_OVER
        self.game_end_time = datetime.now()
        
        duration = (self.game_end_time - self.game_start_time).total_seconds() if self.game_start_time else 0
        
        logger.info(f"Game over! {winner.name} wins! Duration: {duration:.1f}s")
    
    def pause_game(self) -> None:
        """Pause the game."""
        if self.state in [GameState.IN_PROGRESS, GameState.WAITING_FOR_DART]:
            self.state = GameState.PAUSED
            logger.info("Game paused")
    
    def resume_game(self) -> None:
        """Resume the game from pause."""
        if self.state == GameState.PAUSED:
            self.state = GameState.WAITING_FOR_DART
            logger.info("Game resumed")
    
    def next_player(self) -> None:
        """Advance to next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.start_new_round()
    
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
                    "statistics": p.statistics
                }
                for p in self.players
            ],
            "winner": self.winner.name if self.winner else None,
            "rounds_count": len(self.rounds),
            "start_time": self.game_start_time.isoformat() if self.game_start_time else None,
            "end_time": self.game_end_time.isoformat() if self.game_end_time else None
        }
