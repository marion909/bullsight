#!/usr/bin/env python3
"""
Train YOLOv8 model for dart detection using DartsVision dataset.
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

def main():
    project_root = Path(__file__).parent
    dataset_path = project_root / "DartsVision_repo" / "deepdarts_yolo_data" / "data.yaml"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║         Training Dart Detection Model (YOLOv8)                 ║
║                                                                ║
║  Dataset: DartsVision (16,000+ dartboard images)              ║
║  Classes: dart, cal_top, cal_right, cal_bottom, cal_left      ║
║  Model: YOLOv8 Nano (fast training)                           ║
║  Epochs: 100 (with early stopping)                            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Load base model
    print("📥 Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
    # Train
    print("🚀 Starting training...")
    print(f"   Data: {dataset_path}")
    
    results = model.train(
        data=str(dataset_path),
        epochs=50,  # Reduced for CPU training
        imgsz=640,
        batch=16,  # Smaller batch for CPU
        patience=15,  # Earlier stopping
        device='cpu',  # CPU Mode (GPU not available)
        save=True,
        project="bullsight_training",
        name="dart_detector_v1",
        verbose=False,
        plots=False,  # Skip plots to speed up training
        cos_lr=False,
        close_mosaic=5,
    )
    
    print()
    print("✅ Training complete!")
    print()
    print(f"📊 Results saved to: bullsight_training/dart_detector_v1")
    print(f"🎯 Best model: bullsight_training/dart_detector_v1/weights/best.pt")
    print()
    
    # Copy to models directory
    best_model = Path("bullsight_training/dart_detector_v1/weights/best.pt")
    if best_model.exists():
        import shutil
        target = project_root / "models" / "deepdarts_trained.pt"
        shutil.copy(best_model, target)
        print(f"📦 Copied to: {target}")
        print()
        print("✨ Model ready to use! Restart BullSight to activate.")
        return 0
    else:
        print("❌ Training failed - no best.pt generated")
        return 1

if __name__ == "__main__":
    sys.exit(main())
