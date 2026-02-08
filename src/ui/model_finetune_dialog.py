"""
Model finetuning dialog for ML Demo Screen.

Allows users to collect training data and finetune the ML model
directly from the UI without command-line tools.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging
import threading

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QDoubleSpinBox, QProgressBar,
    QGroupBox, QTextEdit, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    """Background thread for model training."""
    
    progress = Signal(str)  # Message updates
    finished = Signal(bool, str)  # (success, message)
    
    def __init__(self, model_path: str, dataset_dir: str, 
                 epochs: int, batch_size: int):
        super().__init__()
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
    
    def run(self):
        """Train model in background thread."""
        try:
            from ultralytics import YOLO
            
            self.progress.emit("📥 Loading model...")
            model = YOLO(self.model_path)
            
            dataset_yaml = Path(self.dataset_dir) / 'data.yaml'
            if not dataset_yaml.exists():
                self.finished.emit(False, f"Dataset not found: {dataset_yaml}")
                return
            
            self.progress.emit(f"🚀 Starting training ({self.epochs} epochs)...")
            
            results = model.train(
                data=str(dataset_yaml),
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=640,
                device='cpu',
                patience=self.epochs // 3,
                save=True,
                project='bullsight_training',
                name='finetuned_model',
                verbose=False,
            )
            
            self.progress.emit("✅ Training complete!")
            self.progress.emit(f"📊 mAP50: {results.metrics.get('metrics/mAP50(B)', 0):.3f}")
            
            # Copy best model
            best_pt = Path('bullsight_training/finetuned_model/weights/best.pt')
            if best_pt.exists():
                import shutil
                target = Path('models/deepdarts_finetuned.pt')
                shutil.copy(best_pt, target)
                self.progress.emit(f"📦 Model saved to: {target}")
                self.finished.emit(True, "Training successful! Restart to use new model.")
            else:
                self.finished.emit(False, "Training failed - no best.pt generated")
        
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class ModelFinetuneDialog(QDialog):
    """Dialog for finetuning ML model with collected dart images."""
    
    model_updated = Signal(str)  # Signal when model is trained
    
    def __init__(self, parent=None):
        """Initialize finetuning dialog."""
        super().__init__(parent)
        
        # Setup finetuning dataset with correct paths
        try:
            import sys
            from pathlib import Path
            setup_script = Path(__file__).parent.parent.parent / "setup_finetuning_dataset.py"
            if setup_script.exists():
                exec(open(str(setup_script)).read())
        except Exception as e:
            print(f"Warning: Could not auto-setup dataset: {e}")
        
        self.dataset_dir = Path('training_data/finetuning_data')
        self.training_worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("🤖 Finetune Dart Detection Model")
        self.setGeometry(100, 100, 600, 700)
        
        layout = QVBoxLayout()
        
        # ========== INFO ==========
        info_label = QLabel("Finetune the ML model with your real dart images")
        info_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(info_label)
        
        # ========== DATASET SECTION ==========
        dataset_group = QGroupBox("📊 Dataset")
        dataset_layout = QVBoxLayout()
        
        dataset_status = QLabel("Status: Ready to collect")
        self.dataset_status_label = dataset_status
        dataset_layout.addWidget(dataset_status)
        
        dataset_info = QLabel(
            "Current data in training_data/finetuning_data/\n"
            "Images with annotations (YOLO format) will be used for training."
        )
        dataset_info.setStyleSheet("color: #666; font-size: 10px;")
        dataset_layout.addWidget(dataset_info)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # ========== TRAINING CONFIG ==========
        config_group = QGroupBox("⚙️ Training Configuration")
        config_layout = QVBoxLayout()
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 200)
        self.epochs_spin.setValue(30)
        self.epochs_spin.setSuffix(" (more = better accuracy, longer training)")
        epochs_layout.addWidget(self.epochs_spin)
        config_layout.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setSuffix(" (lower = less memory, slower)")
        batch_layout.addWidget(self.batch_spin)
        config_layout.addLayout(batch_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # ========== TRAINING LOG ==========
        log_group = QGroupBox("📝 Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # ========== PROGRESS ==========
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # ========== BUTTONS ==========
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("▶️ Start Training")
        self.train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_button)
        
        cancel_button = QPushButton("Close")
        cancel_button.clicked.connect(self.accept)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch
        layout.addStretch()
        
        self.setLayout(layout)
    
    def start_training(self):
        """Start model training with collected images."""
        
        # Check if dataset exists
        if not self.dataset_dir.exists():
            self.log("❌ Dataset directory not found")
            self.log("   Next time:")
            self.log("   1. ML Detection Demo → Press SPACE to capture (50-100 times)")
            self.log("   2. Run: python quick_label_darts.py")
            self.log("   3. Return here to train")
            return
        
        # Check if images exist
        train_images = list((self.dataset_dir / 'images' / 'train').glob('*')) if (self.dataset_dir / 'images' / 'train').exists() else []
        
        if not train_images:
            self.log("❌ No training images found in training_data/finetuning_data/images/train/")
            self.log("")
            self.log("📸 To collect training images:")
            self.log("   1. Go to ML Detection Demo")
            self.log("   2. Press SPACE repeatedly (50-100 times)")
            self.log("   3. Capture frames with 1, 2, 3 darts in different positions")
            return
        
        # Check if labels exist
        train_labels = list((self.dataset_dir / 'labels' / 'train').glob('*.txt')) if (self.dataset_dir / 'labels' / 'train').exists() else []
        
        if not train_labels or all(p.stat().st_size == 0 for p in train_labels):
            self.log("⚠️  No annotations found")
            self.log("")
            self.log("🏷️  To label your images:")
            self.log("   1. Run: python quick_label_darts.py")
            self.log("   2. Draw bounding boxes around darts")
            self.log("   3. Press SPACE to save and continue (takes ~5-10 min for 200 images)")
            self.log("   4. Return here to train")
            return
        
        self.log(f"📊 Found {len(train_images)} training images")
        
        # Disable button during training
        self.train_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        
        self.log(f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}")
        self.log("")
        
        # Start training in background
        model_path = 'models/deepdarts_trained.pt'
        if not Path(model_path).exists():
            model_path = 'yolov8n.pt'
        
        self.training_worker = TrainingWorker(
            model_path, 
            str(self.dataset_dir),
            epochs,
            batch_size
        )
        
        self.training_worker.progress.connect(self.log)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.start()
    
    def log(self, message: str):
        """Add message to training log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def on_training_finished(self, success: bool, message: str):
        """Handle training completion."""
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        
        self.log("")
        if success:
            self.log("✅ " + message)
            self.model_updated.emit('models/deepdarts_finetuned.pt')
        else:
            self.log("❌ " + message)
        
        self.log("")
        self.log("💡 Restart BullSight to use the new model")
