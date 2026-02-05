"""
Live score screen for active dart games.

Displays current game state, player scores, and processes dart throws
with real-time visual and audio feedback.

Author: Mario Neuhauser
"""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from typing import TYPE_CHECKING

from src.game.game_engine import GameState
from src.calibration.board_mapper import DartboardField

if TYPE_CHECKING:
    from src.main import BullSightApp

logger = logging.getLogger(__name__)


class LiveScoreScreen(QWidget):
    """
    Live game screen with scoring and dart detection.
    
    Features:
    - Real-time score display
    - Dart detection and processing
    - Audio feedback
    - Visual animations
    - Statistics overlay
    
    Attributes:
        app: Main application instance
        detection_timer: Timer for dart detection polling
    """
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize live score screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        
        # Detection timer
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.check_for_dart)
        
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup live score UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Top bar: Game info and controls
        top_bar = self.create_top_bar()
        layout.addWidget(top_bar)
        
        # Player scores area
        self.scores_layout = QVBoxLayout()
        layout.addLayout(self.scores_layout)
        
        # Current throw area
        self.throw_area = self.create_throw_area()
        layout.addWidget(self.throw_area)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.next_player_btn = QPushButton("Next Player (Manual)")
        self.next_player_btn.setMinimumHeight(80)
        self.next_player_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
            }
        """)
        self.next_player_btn.clicked.connect(self.manual_next_player)
        button_layout.addWidget(self.next_player_btn)
        
        pause_btn = QPushButton("Pause Game")
        pause_btn.setMinimumHeight(80)
        pause_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
        """)
        pause_btn.clicked.connect(self.pause_game)
        button_layout.addWidget(pause_btn)
        
        quit_btn = QPushButton("End Game")
        quit_btn.setMinimumHeight(80)
        quit_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
        """)
        quit_btn.clicked.connect(self.confirm_quit)
        button_layout.addWidget(quit_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_top_bar(self) -> QFrame:
        """Create top information bar."""
        bar = QFrame()
        bar.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout()
        
        self.game_mode_label = QLabel("Game: ---")
        self.game_mode_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.game_mode_label)
        
        layout.addStretch()
        
        self.round_label = QLabel("Round: 1")
        self.round_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.round_label)
        
        bar.setLayout(layout)
        return bar
    
    def create_throw_area(self) -> QFrame:
        """Create current throw display area."""
        area = QFrame()
        area.setFrameStyle(QFrame.Shape.Box)
        area.setStyleSheet("background-color: #263238; border-radius: 10px;")
        area.setMinimumHeight(200)
        
        layout = QVBoxLayout()
        
        self.current_player_label = QLabel("Current Player: ---")
        self.current_player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_player_label.setStyleSheet("font-size: 32px; color: white; font-weight: bold;")
        layout.addWidget(self.current_player_label)
        
        self.dart_display = QLabel("Waiting for dart...")
        self.dart_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dart_display.setStyleSheet("font-size: 48px; color: #4CAF50;")
        layout.addWidget(self.dart_display)
        
        self.darts_remaining_label = QLabel("Darts: 0 / 3")
        self.darts_remaining_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.darts_remaining_label.setStyleSheet("font-size: 24px; color: #FFD600;")
        layout.addWidget(self.darts_remaining_label)
        
        area.setLayout(layout)
        return area
    
    def start_game(self) -> None:
        """Start live game session."""
        if self.app.game is None:
            logger.error("No game instance to start")
            return
        
        # Update UI with game info
        self.game_mode_label.setText(f"Game: {self.app.game.mode.value}")
        self.update_scores()
        self.update_current_player()
        
        # Start dart detection
        if self.app.start_camera():
            # Capture reference image
            reference = self.app.camera.capture()
            if reference is not None:
                self.app.detector.set_reference_image(reference)
                
                # Start detection timer (check every 500ms)
                self.detection_timer.start(500)
                logger.info("Dart detection started")
        else:
            logger.warning("Failed to start camera, manual mode only")
    
    def check_for_dart(self) -> None:
        """Check for dart in current frame."""
        if not self.app.camera or self.app.game.state != GameState.WAITING_FOR_DART:
            return
        
        try:
            # Capture current image
            current_image = self.app.camera.capture()
            
            if current_image is None:
                return
            
            # Detect dart
            dart_coord = self.app.detector.detect_dart(current_image)
            
            if dart_coord:
                # Map to field
                field = self.app.mapper.map_coordinate(dart_coord.x, dart_coord.y)
                
                # Process dart
                self.process_dart(field)
                
                # Update reference image for next dart
                self.app.detector.set_reference_image(current_image)
                
        except Exception as e:
            logger.error(f"Dart detection error: {e}")
    
    def process_dart(self, field: DartboardField) -> None:
        """
        Process detected dart throw.
        
        Args:
            field: Detected dartboard field
        """
        # Play sound
        self.play_sound_for_field(field)
        
        # Display dart
        self.dart_display.setText(self.format_field(field))
        
        # Record in game engine
        is_complete = self.app.game.record_dart(field)
        
        # Update UI
        self.update_scores()
        
        darts_thrown = len(self.app.game.current_round.darts)
        self.darts_remaining_label.setText(f"Darts: {darts_thrown} / 3")
        
        if is_complete:
            self.handle_round_complete()
    
    def format_field(self, field: DartboardField) -> str:
        """Format field for display."""
        if field.zone == "miss":
            return "MISS"
        elif field.zone == "bull_eye":
            return "BULL'S EYE! (50)"
        elif field.zone == "bull":
            return f"BULL! (25)"
        elif field.multiplier == 3:
            return f"TRIPLE {field.segment}! ({field.score})"
        elif field.multiplier == 2:
            return f"DOUBLE {field.segment}! ({field.score})"
        else:
            return f"{field.segment} ({field.score})"
    
    def play_sound_for_field(self, field: DartboardField) -> None:
        """Play appropriate sound for field."""
        try:
            import pygame
            
            # Simple beep sounds based on field type
            if field.zone == "bull_eye" or field.zone == "bull":
                # High pitch for bull
                self.play_beep(880, 0.2)
            elif field.zone == "triple":
                # Medium-high pitch for triple
                self.play_beep(660, 0.15)
            elif field.zone == "double":
                # Medium pitch for double
                self.play_beep(523, 0.15)
            else:
                # Low pitch for single
                self.play_beep(440, 0.1)
        except Exception as e:
            logger.warning(f"Could not play sound: {e}")
    
    def play_beep(self, frequency: int, duration: float) -> None:
        """Play a simple beep sound."""
        try:
            import pygame
            import numpy as np
            
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t) * 0.3
            wave = (wave * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_wave = np.column_stack((wave, wave))
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
        except:
            pass
    
    def handle_round_complete(self) -> None:
        """Handle round completion."""
        if self.app.game.current_round.is_bust:
            self.dart_display.setText("BUST!")
            self.dart_display.setStyleSheet("font-size: 48px; color: #f44336;")
            self.play_beep(220, 0.3)
        elif self.app.game.current_round.is_checkout:
            self.play_beep(1047, 0.5)
            self.handle_game_over()
            return
        
        # Update round number
        current_player = self.app.game.get_current_player()
        self.round_label.setText(f"Round: {current_player.rounds_played + 1}")
        
        # Reset for next player
        QTimer.singleShot(2000, self.prepare_next_player)
    
    def prepare_next_player(self) -> None:
        """Prepare UI for next player."""
        self.update_current_player()
        self.dart_display.setText("Waiting for dart...")
        self.dart_display.setStyleSheet("font-size: 48px; color: #4CAF50;")
        self.darts_remaining_label.setText("Darts: 0 / 3")
    
    def update_scores(self) -> None:
        """Update player score display."""
        # Clear existing scores
        while self.scores_layout.count():
            item = self.scores_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add updated scores
        for player in self.app.game.players:
            score_widget = self.create_player_score_widget(player)
            self.scores_layout.addWidget(score_widget)
    
    def create_player_score_widget(self, player) -> QFrame:
        """Create score widget for player."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Panel)
        frame.setMinimumHeight(100)
        
        # Highlight current player
        if player.id == self.app.game.get_current_player().id:
            frame.setStyleSheet("background-color: #1976D2; color: white;")
        
        layout = QHBoxLayout()
        
        name = QLabel(player.name)
        name.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(name)
        
        layout.addStretch()
        
        score = QLabel(str(player.score))
        score.setStyleSheet("font-size: 36px; font-weight: bold;")
        layout.addWidget(score)
        
        frame.setLayout(layout)
        return frame
    
    def update_current_player(self) -> None:
        """Update current player display."""
        player = self.app.game.get_current_player()
        self.current_player_label.setText(f"Current Player: {player.name}")
    
    def manual_next_player(self) -> None:
        """Manually advance to next player (backup)."""
        # Complete current round with misses if needed
        while len(self.app.game.current_round.darts) < 3:
            field = DartboardField(0, "miss", 0, 0)
            self.app.game.record_dart(field)
        
        self.prepare_next_player()
    
    def pause_game(self) -> None:
        """Pause the game."""
        self.app.game.pause_game()
        self.detection_timer.stop()
        
        reply = QMessageBox.information(
            self,
            "Game Paused",
            "Game is paused. Click OK to resume.",
            QMessageBox.StandardButton.Ok
        )
        
        if reply == QMessageBox.StandardButton.Ok:
            self.app.game.resume_game()
            self.detection_timer.start(500)
    
    def handle_game_over(self) -> None:
        """Handle game completion."""
        self.detection_timer.stop()
        winner = self.app.game.winner
        
        self.dart_display.setText(f"🏆 {winner.name} WINS! 🏆")
        self.dart_display.setStyleSheet("font-size: 42px; color: #FFD700;")
        
        # Show statistics
        stats = self.app.game.get_game_summary()
        stats_text = f"""
        Winner: {winner.name}
        Final Score: {winner.score}
        Rounds Played: {winner.rounds_played}
        Total Darts: {winner.darts_thrown}
        """
        
        QMessageBox.information(
            self,
            "Game Over",
            stats_text,
            QMessageBox.StandardButton.Ok
        )
        
        # Return to start screen
        self.app.show_screen("start")
    
    def confirm_quit(self) -> None:
        """Confirm and quit game."""
        reply = QMessageBox.question(
            self,
            "End Game",
            "Are you sure you want to end the game?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.detection_timer.stop()
            self.app.stop_camera()
            self.app.show_screen("start")
