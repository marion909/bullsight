#!/usr/bin/env python3
"""
Generate synthetic dartboard training dataset and train YOLO model.
Creates minimal training data to get a working model quickly.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def create_synthetic_dartboard(width=640, height=640, num_darts=None):
    """Create a synthetic dartboard image with random dart positions."""
    
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw dartboard rings
    center_x, center_y = width // 2, height // 2
    ring_radii = [50, 100, 150, 200, 250]  # Various ring sizes
    
    # Draw circles
    for radius in ring_radii:
        x0 = center_x - radius
        y0 = center_y - radius
        x1 = center_x + radius
        y1 = center_y + radius
        draw.ellipse([x0, y0, x1, y1], outline='black', width=2)
    
    # Draw dartboard sections
    for angle in range(0, 360, 18):  # 20 sections
        rad = np.radians(angle)
        x1 = center_x + 250 * np.cos(rad)
        y1 = center_y + 250 * np.sin(rad)
        draw.line([(center_x, center_y), (int(x1), int(y1))], fill='black', width=1)
    
    # Add random darts
    if num_darts is None:
        num_darts = random.randint(1, 3)
    
    dart_annotations = []
    for _ in range(num_darts):
        # Random position within dartboard area
        dart_x = center_x + random.randint(-250, 250)
        dart_y = center_y + random.randint(-250, 250)
        
        # Draw dart
        dart_size = 4
        dart_bbox = [dart_x - dart_size, dart_y - dart_size, 
                     dart_x + dart_size, dart_y + dart_size]
        draw.ellipse(dart_bbox, fill='red', outline='darkred')
        
        # Convert to YOLO format (normalized coordinates)
        # YOLO format: class_id center_x center_y width height (all normalized 0-1)
        norm_x = dart_x / width
        norm_y = dart_y / height
        norm_w = (dart_size * 2) / width
        norm_h = (dart_size * 2) / height
        
        dart_annotations.append(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    return img, dart_annotations

def create_synthetic_dataset(output_dir="synthetic_dart_dataset", num_images=100):
    """Create a minimal synthetic dartboard dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating synthetic dartboard dataset...")
    print(f"   Output: {output_dir}")
    
    # Generate images
    split_counts = {'train': int(num_images * 0.7), 
                   'val': int(num_images * 0.2),
                   'test': int(num_images * 0.1)}
    
    for split, count in split_counts.items():
        print(f"   {split}: {count} images...", end='', flush=True)
        for i in range(count):
            # Create image
            img, annotations = create_synthetic_dartboard()
            
            # Save image
            img_name = f"dartboard_{split}_{i:04d}.jpg"
            img_path = output_path / 'images' / split / img_name
            img.save(img_path, 'JPEG', quality=85)
            
            # Save annotations (YOLO format)
            label_name = f"dartboard_{split}_{i:04d}.txt"
            label_path = output_path / 'labels' / split / label_name
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(ann + '\n')
        
        print(" ✓")
    
    # Create data.yaml
    data_yaml = f"""path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: dart
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    print(f"✅ Dataset created: {yaml_path}")
    return str(yaml_path)

def train_model(data_yaml):
    """Train YOLO on the dataset."""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         Training Dart Detection Model (YOLOv8-nano)            ║")
    print("║         Using Synthetic Dartboard Dataset                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    model = YOLO('yolov8n.pt')
    
    print("🚀 Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=30,  # Quick training
        imgsz=640,
        batch=8,
        patience=10,
        device='cpu',
        save=True,
        project="bullsight_training",
        name="dart_detector_v1",
        verbose=False,
        plots=False,
    )
    
    print("✅ Training complete!")
    return True

def main():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         BullSight Dart Model Generator                         ║")
    print("║                                                                ║")
    print("║  This script creates synthetic training data and trains a      ║")
    print("║  YOLOv8 model for dart detection.                             ║")
    print("║                                                                ║")
    print("║  ⚠️  This is a fast demo model. For best results, train with   ║")
    print("║  real dartboard photos using the full training guide.         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create synthetic dataset
    data_yaml = create_synthetic_dataset(
        output_dir="synthetic_dart_dataset",
        num_images=100  # Quick dataset
    )
    
    # Train model
    success = train_model(data_yaml)
    
    if success:
        # Copy to models directory
        import shutil
        try:
            src = Path("bullsight_training/dart_detector_v1/weights/best.pt")
            if src.exists():
                dst = Path("models/deepdarts_trained.pt")
                shutil.copy(src, dst)
                print(f"📦 Copied to: {dst}")
                print()
                print("✨ Model ready! Restart BullSight to activate:")
                print("   python -m src.main")
                return 0
        except Exception as e:
            print(f"⚠️  Could not copy model: {e}")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
