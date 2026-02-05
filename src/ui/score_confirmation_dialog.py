"""
Score confirmation and correction dialog.

Allows players to confirm or correct their 3-dart score after a round.

Author: Mario Neuhauser
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGridLayout, QFrame, QScrollArea, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import List, Optional

from src.calibration.board_mapper import DartboardField


class ScoreConfirmationDialog(QDialog):
    """
    Dialog for confirming or correcting 3-dart score.
    
    Shows current detected throws and allows:
    - Confirm: Accept detected score
    - Correct: Manually select 3 throws from all possible fields
    """
    
    def __init__(self, detected_throws: List[DartboardField], parent=None):
        """
        Initialize dialog.
        
        Args:
            detected_throws: List of detected dart throws (up to 3)
            parent: Parent widget
        """
        super().__init__(parent)
        self.detected_throws = detected_throws
        self.corrected_throws: Optional[List[DartboardField]] = None
        self.selected_fields: List[DartboardField] = []
        
        self.setWindowTitle("Confirm Score")
        self.setModal(True)
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("Round Complete - Confirm Score")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Detected throws display
        detected_frame = self.create_detected_throws_display()
        layout.addWidget(detected_frame)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        confirm_btn = QPushButton("✓ Confirm Score")
        confirm_btn.setMinimumHeight(80)
        confirm_btn.setStyleSheet("""
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
        confirm_btn.clicked.connect(self.confirm_score)
        action_layout.addWidget(confirm_btn)
        
        correct_btn = QPushButton("✎ Correct Score")
        correct_btn.setMinimumHeight(80)
        correct_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        correct_btn.clicked.connect(self.show_correction_ui)
        action_layout.addWidget(correct_btn)
        
        layout.addLayout(action_layout)
        
        self.setLayout(layout)
    
    def create_detected_throws_display(self) -> QFrame:
        """Create display for detected throws."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        frame.setStyleSheet("background-color: #f5f5f5; padding: 20px;")
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Detected Throws:")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Throws display
        throws_layout = QHBoxLayout()
        total_score = 0
        
        for i, throw in enumerate(self.detected_throws, 1):
            throw_widget = self.create_throw_widget(i, throw)
            throws_layout.addWidget(throw_widget)
            total_score += throw.score
        
        # Fill empty slots
        for i in range(len(self.detected_throws), 3):
            empty_widget = self.create_empty_throw_widget(i + 1)
            throws_layout.addWidget(empty_widget)
        
        layout.addLayout(throws_layout)
        
        # Total score
        total_label = QLabel(f"Total: {total_score} points")
        total_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(total_label)
        
        frame.setLayout(layout)
        return frame
    
    def create_throw_widget(self, number: int, field: DartboardField) -> QFrame:
        """Create widget for single throw."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        frame.setMinimumWidth(200)
        
        layout = QVBoxLayout()
        
        # Dart number
        number_label = QLabel(f"Dart {number}")
        number_label.setStyleSheet("font-size: 14px; color: #666;")
        number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(number_label)
        
        # Field description
        field_text = self.format_field(field)
        field_label = QLabel(field_text)
        field_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        field_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        field_label.setWordWrap(True)
        layout.addWidget(field_label)
        
        # Score
        score_label = QLabel(f"{field.score} pts")
        score_label.setStyleSheet("font-size: 18px; color: #2196F3;")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(score_label)
        
        frame.setLayout(layout)
        return frame
    
    def create_empty_throw_widget(self, number: int) -> QFrame:
        """Create widget for empty throw slot."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #eeeeee;
                border: 3px dashed #999;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        frame.setMinimumWidth(200)
        
        layout = QVBoxLayout()
        
        number_label = QLabel(f"Dart {number}")
        number_label.setStyleSheet("font-size: 14px; color: #999;")
        number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(number_label)
        
        empty_label = QLabel("Not thrown")
        empty_label.setStyleSheet("font-size: 16px; color: #999;")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_label)
        
        frame.setLayout(layout)
        return frame
    
    def format_field(self, field: DartboardField) -> str:
        """Format field for display."""
        if field.zone == "miss":
            return "MISS"
        elif field.zone == "bull_eye":
            return "BULL'S EYE"
        elif field.zone == "bull":
            return "BULL"
        elif field.multiplier == 3:
            return f"T{field.segment}"
        elif field.multiplier == 2:
            return f"D{field.segment}"
        else:
            return f"{field.segment}"
    
    def confirm_score(self):
        """Confirm detected score."""
        self.corrected_throws = None
        self.accept()
    
    def show_correction_ui(self):
        """Show UI for manual score correction."""
        # Clear current layout
        self.setLayout(QVBoxLayout())
        layout = self.layout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("Select 3 Throws")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Selected throws display
        self.selected_display = QLabel("Selected: 0 / 3")
        self.selected_display.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        self.selected_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.selected_display)
        
        # Field selector (NO ScrollArea - all buttons visible)
        field_widget = QWidget()
        field_layout = QGridLayout(field_widget)
        field_layout.setSpacing(5)
        field_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create all possible fields
        fields = self.generate_all_fields()
        
        row = 0
        col = 0
        for field in fields:
            btn = self.create_field_button(field)
            field_layout.addWidget(btn, row, col)
            
            col += 1
            if col >= 6:  # 6 buttons per row
                col = 0
                row += 1
        
        layout.addWidget(field_widget)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.setMinimumHeight(60)
        clear_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #9E9E9E;
                color: white;
                border-radius: 5px;
            }
        """)
        clear_btn.clicked.connect(self.clear_selection)
        action_layout.addWidget(clear_btn)
        
        self.submit_btn = QPushButton("Submit Correction")
        self.submit_btn.setMinimumHeight(60)
        self.submit_btn.setEnabled(False)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.submit_btn.clicked.connect(self.submit_correction)
        action_layout.addWidget(self.submit_btn)
        
        layout.addLayout(action_layout)
    
    def generate_all_fields(self) -> List[DartboardField]:
        """Generate all possible dartboard fields."""
        fields = []
        
        # Miss
        fields.append(DartboardField(segment=0, multiplier=0, score=0, zone="miss"))
        
        # Bull's Eye
        fields.append(DartboardField(segment=25, multiplier=2, score=50, zone="bull_eye"))
        
        # Bull
        fields.append(DartboardField(segment=25, multiplier=1, score=25, zone="bull"))
        
        # All segments
        segments = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        for segment in segments:
            # Single
            fields.append(DartboardField(segment=segment, multiplier=1, score=segment, zone="single"))
            # Double
            fields.append(DartboardField(segment=segment, multiplier=2, score=segment*2, zone="double"))
            # Triple
            fields.append(DartboardField(segment=segment, multiplier=3, score=segment*3, zone="triple"))
        
        return fields
    
    def create_field_button(self, field: DartboardField) -> QPushButton:
        """Create button for field selection."""
        text = self.format_field_button_text(field)
        
        btn = QPushButton(text)
        btn.setMinimumHeight(60)
        btn.setMinimumWidth(100)
        
        # Color coding
        if field.zone == "miss":
            color = "#f44336"
        elif field.zone == "bull_eye":
            color = "#FFD700"
        elif field.zone == "bull":
            color = "#FFA500"
        elif field.zone == "triple":
            color = "#4CAF50"
        elif field.zone == "double":
            color = "#2196F3"
        else:
            color = "#9E9E9E"
        
        btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 16px;
                font-weight: bold;
                background-color: {color};
                color: white;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """)
        
        btn.clicked.connect(lambda: self.select_field(field))
        
        return btn
    
    def format_field_button_text(self, field: DartboardField) -> str:
        """Format field for button text."""
        if field.zone == "miss":
            return "MISS\n0"
        elif field.zone == "bull_eye":
            return "Bull's Eye\n50"
        elif field.zone == "bull":
            return "Bull\n25"
        elif field.multiplier == 3:
            return f"T{field.segment}\n{field.score}"
        elif field.multiplier == 2:
            return f"D{field.segment}\n{field.score}"
        else:
            return f"{field.segment}\n{field.score}"
    
    def select_field(self, field: DartboardField):
        """Handle field selection."""
        if len(self.selected_fields) < 3:
            self.selected_fields.append(field)
            
            # Update display
            total_score = sum(f.score for f in self.selected_fields)
            
            # Show selected fields
            selected_text = " + ".join(
                self.format_field_short(f) for f in self.selected_fields
            )
            self.selected_display.setText(
                f"Selected: {len(self.selected_fields)} / 3 | {selected_text} = {total_score} pts"
            )
            
            # Enable submit if 3 selected
            if len(self.selected_fields) == 3:
                self.submit_btn.setEnabled(True)
    
    def format_field_short(self, field: DartboardField) -> str:
        """Format field name for short display."""
        if field.zone == "miss":
            return "MISS"
        elif field.zone == "bull_eye":
            return "BE"
        elif field.zone == "bull":
            return "B"
        elif field.multiplier == 3:
            return f"T{field.segment}"
        elif field.multiplier == 2:
            return f"D{field.segment}"
        else:
            return f"{field.segment}"
    
    def clear_selection(self):
        """Clear all selected fields."""
        self.selected_fields.clear()
        self.selected_display.setText("Selected: 0 / 3")
        self.submit_btn.setEnabled(False)
    
    def submit_correction(self):
        """Submit corrected throws."""
        if len(self.selected_fields) == 3:
            self.corrected_throws = self.selected_fields
            self.accept()
    
    def get_final_throws(self) -> List[DartboardField]:
        """Get final throws (detected or corrected)."""
        if self.corrected_throws is not None:
            return self.corrected_throws
        return self.detected_throws
