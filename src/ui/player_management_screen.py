"""
Player management screen for BullSight application.

Allows adding and removing players (1-8) before starting a game.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QListWidget, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import TYPE_CHECKING, List

from src.game.game_engine import Player

if TYPE_CHECKING:
    from src.main import BullSightApp


class PlayerManagementScreen(QWidget):
    """
    Player management screen for game setup.
    
    Features:
    - Add players (up to 8)
    - Remove players
    - Name input with touch keyboard
    - Validation (1-8 players required)
    
    Attributes:
        app: Main application instance
        players: List of players for the game
    """
    
    def __init__(self, app: 'BullSightApp'):
        """
        Initialize player management screen.
        
        Args:
            app: Main application instance
        """
        super().__init__()
        self.app = app
        self.players: List[Player] = []
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Setup player management UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Add Players")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Player input
        input_layout = QHBoxLayout()
        
        self.player_name_input = QLineEdit()
        self.player_name_input.setPlaceholderText("Enter player name...")
        self.player_name_input.setMinimumHeight(60)
        self.player_name_input.setStyleSheet("font-size: 20px; padding: 10px;")
        self.player_name_input.returnPressed.connect(self.add_player)
        input_layout.addWidget(self.player_name_input)
        
        add_btn = QPushButton("Add")
        add_btn.setMinimumSize(120, 60)
        add_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_btn.clicked.connect(self.add_player)
        input_layout.addWidget(add_btn)
        
        layout.addLayout(input_layout)
        
        # Player list
        self.player_list = QListWidget()
        self.player_list.setMinimumHeight(300)
        self.player_list.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.player_list)
        
        # Remove button
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setMinimumHeight(60)
        remove_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        remove_btn.clicked.connect(self.remove_player)
        layout.addWidget(remove_btn)
        
        # Info label
        self.info_label = QLabel("0 players (minimum 1, maximum 8)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(self.info_label)
        
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
        
        continue_btn = QPushButton("Continue")
        continue_btn.setMinimumHeight(80)
        continue_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        continue_btn.clicked.connect(self.continue_to_game_mode)
        button_layout.addWidget(continue_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_player(self) -> None:
        """Add a player to the list."""
        name = self.player_name_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a player name.")
            return
        
        if len(self.players) >= 8:
            QMessageBox.warning(self, "Maximum Players", "Maximum 8 players allowed.")
            return
        
        # Check for duplicate names
        if any(p.name == name for p in self.players):
            QMessageBox.warning(self, "Duplicate Name", f"Player '{name}' already exists.")
            return
        
        # Add player
        player = Player(name=name, id=len(self.players) + 1)
        self.players.append(player)
        self.player_list.addItem(f"{len(self.players)}. {name}")
        self.player_name_input.clear()
        self.update_info_label()
    
    def remove_player(self) -> None:
        """Remove selected player from the list."""
        current_row = self.player_list.currentRow()
        if current_row >= 0:
            self.players.pop(current_row)
            self.player_list.takeItem(current_row)
            
            # Re-number players
            self.player_list.clear()
            for i, player in enumerate(self.players):
                player.id = i + 1
                self.player_list.addItem(f"{i + 1}. {player.name}")
            
            self.update_info_label()
    
    def update_info_label(self) -> None:
        """Update player count info label."""
        count = len(self.players)
        self.info_label.setText(f"{count} players (minimum 1, maximum 8)")
    
    def continue_to_game_mode(self) -> None:
        """Continue to game mode selection."""
        if len(self.players) < 1:
            QMessageBox.warning(
                self,
                "No Players",
                "Please add at least 1 player to continue."
            )
            return
        
        # Navigate to game mode screen
        self.app.show_screen("game_mode")
    
    def go_back(self) -> None:
        """Return to start screen."""
        # Clear players when going back
        self.players.clear()
        self.player_list.clear()
        self.update_info_label()
        self.app.show_screen("start")
    
    def get_players(self) -> List[Player]:
        """
        Get the list of players.
        
        Returns:
            List of Player objects
        """
        return self.players.copy()
