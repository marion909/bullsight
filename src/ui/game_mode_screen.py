"""
Game mode selection screen for BullSight application.

Allows selecting game mode and configuration before starting.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import TYPE_CHECKING

from src.game.game_engine import GameMode, GameEngine

if TYPE_CHECKING:
    from src.main import BullSightApp


class GameModeScreen(QWidget):
    """
    Game mode selection screen.
    
    Features:
    - Select game mode (301/501/701/Cricket/ATCC/Training)
    - Toggle double-out requirement
    - Start game with selected settings
    
    Attributes:
        app: Main application instance
        selected_mode: Currently selected game mode
        double_out: Double-out requirement enabled
    """
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize game mode screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        self.selected_mode = GameMode.GAME_301
        self.double_out = False
        self.double_in = False
        self.master_in = False
        self.master_out = False
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup game mode selection UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Select Game Mode")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Game mode buttons grid
        grid = QGridLayout()
        grid.setSpacing(15)
        
        modes = [
            (GameMode.GAME_301, "301", 0, 0),
            (GameMode.GAME_501, "501", 0, 1),
            (GameMode.GAME_701, "701", 0, 2),
            (GameMode.CRICKET, "Cricket", 1, 0),
            (GameMode.AROUND_THE_CLOCK, "Around the Clock", 1, 1),
            (GameMode.TRAINING, "Training", 1, 2),
        ]
        
        self.mode_buttons = {}
        for mode, text, row, col in modes:
            btn = QPushButton(text)
            btn.setMinimumSize(200, 100)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px;
                    background-color: #424242;
                    color: white;
                    border-radius: 10px;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background-color: #2196F3;
                }
                QPushButton:hover {
                    background-color: #616161;
                }
                QPushButton:checked:hover {
                    background-color: #1976D2;
                }
            """)
            btn.clicked.connect(lambda checked, m=mode: self.select_mode(m))
            grid.addWidget(btn, row, col)
            self.mode_buttons[mode] = btn
        
        # Set default selection
        self.mode_buttons[GameMode.GAME_301].setChecked(True)
        
        layout.addLayout(grid)
        
        # Options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(10)
        
        options_label = QLabel("Options")
        options_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        options_layout.addWidget(options_label)
        
        # In-Options (mutually exclusive)
        in_label = QLabel("Game Start:")
        in_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px;")
        options_layout.addWidget(in_label)
        
        self.double_in_checkbox = QCheckBox("Double In")
        self.double_in_checkbox.setChecked(False)
        self.double_in_checkbox.setStyleSheet("font-size: 18px; margin-left: 20px;")
        self.double_in_checkbox.stateChanged.connect(self.toggle_double_in)
        options_layout.addWidget(self.double_in_checkbox)
        
        self.master_in_checkbox = QCheckBox("Master In (Bull or Double)")
        self.master_in_checkbox.setChecked(False)
        self.master_in_checkbox.setStyleSheet("font-size: 18px; margin-left: 20px;")
        self.master_in_checkbox.stateChanged.connect(self.toggle_master_in)
        options_layout.addWidget(self.master_in_checkbox)
        
        # Out-Options
        out_label = QLabel("Game Finish:")
        out_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px;")
        options_layout.addWidget(out_label)
        
        self.double_out_checkbox = QCheckBox("Double Out")
        self.double_out_checkbox.setChecked(True)
        self.double_out_checkbox.setStyleSheet("font-size: 18px; margin-left: 20px;")
        self.double_out_checkbox.stateChanged.connect(self.toggle_double_out)
        options_layout.addWidget(self.double_out_checkbox)
        
        self.master_out_checkbox = QCheckBox("Master Out (Bull or Double)")
        self.master_out_checkbox.setChecked(False)
        self.master_out_checkbox.setStyleSheet("font-size: 18px; margin-left: 20px;")
        self.master_out_checkbox.stateChanged.connect(self.toggle_master_out)
        options_layout.addWidget(self.master_out_checkbox)
        
        layout.addLayout(options_layout)
        
        layout.addStretch()
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        back_btn = QPushButton("Back")
        back_btn.setMinimumHeight(80)
        back_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #757575;
                color: white;
                border-radius: 10px;
            }
        """)
        back_btn.clicked.connect(self.go_back)
        button_layout.addWidget(back_btn)
        
        start_btn = QPushButton("Start Game")
        start_btn.setMinimumHeight(80)
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_btn.clicked.connect(self.start_game)
        button_layout.addWidget(start_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def select_mode(self, mode: GameMode) -> None:
        """
        Select a game mode.
        
        Args:
            mode: Selected game mode
        """
        self.selected_mode = mode
        
        # Update button states
        for m, btn in self.mode_buttons.items():
            btn.setChecked(m == mode)
    
    def toggle_double_out(self, state: int) -> None:
        """
        Toggle double-out requirement.
        
        Args:
            state: Checkbox state
        """
        self.double_out = (state == Qt.CheckState.Checked.value)
    
    def toggle_double_in(self, state: int) -> None:
        """
        Toggle double-in requirement.
        Mutually exclusive with master-in.
        
        Args:
            state: Checkbox state
        """
        if state == Qt.CheckState.Checked.value:
            self.double_in = True
            self.master_in = False
            self.master_in_checkbox.blockSignals(True)
            self.master_in_checkbox.setChecked(False)
            self.master_in_checkbox.blockSignals(False)
        else:
            self.double_in = False
    
    def toggle_master_in(self, state: int) -> None:
        """
        Toggle master-in requirement.
        Mutually exclusive with double-in.
        
        Args:
            state: Checkbox state
        """
        if state == Qt.CheckState.Checked.value:
            self.master_in = True
            self.double_in = False
            self.double_in_checkbox.blockSignals(True)
            self.double_in_checkbox.setChecked(False)
            self.double_in_checkbox.blockSignals(False)
        else:
            self.master_in = False
    
    def toggle_master_out(self, state: int) -> None:
        """
        Toggle master-out requirement.
        
        Args:
            state: Checkbox state
        """
        self.master_out = (state == Qt.CheckState.Checked.value)
    
    def start_game(self) -> None:
        """Start game with selected settings."""
        # Get players from player management screen
        player_screen = self.app.screens["player_management"]
        players = player_screen.get_players()
        
        if not players:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Players",
                "No players configured. Please go back and add players."
            )
            return
        
        # Create game engine
        config = {
            "double_out": self.double_out,
            "double_in": self.double_in,
            "master_in": self.master_in,
            "master_out": self.master_out
        }
        self.app.game = GameEngine(self.selected_mode, players, config)
        self.app.game.start_game()
        
        # Navigate to live score screen
        live_screen = self.app.screens["live_score"]
        live_screen.start_game()
        self.app.show_screen("live_score")
    
    def go_back(self) -> None:
        """Return to player management screen."""
        self.app.show_screen("player_management")
