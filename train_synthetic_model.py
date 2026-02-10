#!/usr/bin/env python3
"""
Create a quick dart detection model using synthetic training data.
This bypasses the need for the full DartsVision image dataset.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import random

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def create_synthetic_dartboard_dataset(output_dir: str, num_images: int = 100):
    """
    Generate synthetic dartboard images with annotated dart positions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_imgs = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_imgs = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    train_imgs.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_imgs.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    img_size = 640
    
    print(f"📝 Generating {num_images} synthetic dartboard images...")
    
    for i in range(num_images):
        # Create blank image with dartboard colors
        img = Image.new('RGB', (img_size, img_size), color=(200, 100, 50))
        draw = ImageDraw.Draw(img)
        
        # Draw dartboard rings (concentric circles)
        center = (img_size // 2, img_size // 2)
        colors = [(255, 200, 0), (0, 0, 0), (255, 200, 0), (0, 0, 0)]
        
        for ring_idx, radius in enumerate([280, 260, 200, 180, 100, 80, 30, 10]):
            color = colors[ring_idx % len(colors)]
            draw.ellipse([
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ], outline=color, width=3)
        
        # Draw wedge lines for dartboard sections
        for angle in range(0, 360, 18):
            rad = np.radians(angle)
            x1 = center[0] + 280 * np.cos(rad)
            y1 = center[1] + 280 * np.sin(rad)
            draw.line([center, (x1, y1)], fill=(100, 100, 100), width=1)
        
        # Generate 1-3 random dart positions
        num_darts = random.randint(1, 3)
        labels_list = []
        
        for _ in range(num_darts):
            # Random dart position within dartboard bounds
            dart_x = random.randint(center[0] - 250, center[0] + 250)
            dart_y = random.randint(center[1] - 250, center[1] + 250)
            
            # Draw dart (small circle)
            dart_size = random.randint(8, 15)
            draw.ellipse([
                dart_x - dart_size, dart_y - dart_size,
                dart_x + dart_size, dart_y + dart_size
            ], fill=(255, 0, 0), outline=(200, 0, 0), width=2)
            
            # YOLO format: class x_center y_center width height (normalized 0-1)
            x_norm = dart_x / img_size
            y_norm = dart_y / img_size
            w_norm = (dart_size * 2) / img_size
            h_norm = (dart_size * 2) / img_size
            
            # Class 0 = dart
            labels_list.append(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Determine train/val split
        is_train = random.random() < 0.8
        split = "train" if is_train else "val"
        
        # Save image
        img_name = f"dartboard_{split}_{i:04d}.jpg"
        img_path = train_imgs / img_name if is_train else val_imgs / img_name
        img.save(img_path, quality=95)
        
        # Save labels
        label_name = f"dartboard_{split}_{i:04d}.txt"
        label_path = train_labels / label_name if is_train else val_labels / label_name
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels_list))
        
        if (i + 1) % 20 == 0:
            print(f"  ✓ Generated {i+1}/{num_images} images")
    
    print(f"✅ Dataset created in {output_dir}")
    return output_dir


def create_dataset_yaml(dataset_dir: Path):
    """Create YOLO data.yaml file."""
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val

nc: 1
names:
  0: dart
"""
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path


def train_synthetic_model():
    """Train model on synthetic data."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║     Training Dart Detection Model (Synthetic Data)             ║
║                                                                ║
║  This creates a quick test model using generated dartboard    ║
║  images. For production, train with real dartboard photos.    ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Create synthetic dataset
    dataset_dir = create_synthetic_dartboard_dataset(
        "synthetic_dart_dataset",
        num_images=200  # Small number for quick training
    )
    
    # Create data.yaml
    yaml_path = create_dataset_yaml(dataset_dir)
    print(f"\n📄 Created {yaml_path}")
    
    # Load YOLOv8n model
    print("\n📥 Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
    # Train on synthetic data
    print("\n🚀 Starting training on synthetic data...")
    print("   (This is just a proof-of-concept. Train with real images for production.)\n")
    
    results = model.train(
        data=str(yaml_path),
        epochs=30,  # Quick training
        imgsz=640,
        batch=16,
        device='cpu',  # CPU mode
        patience=10,
        save=True,
        project='bullsight_training',
        name='dart_detector_synthetic',
        verbose=False,
        plots=False,
    )
    
    print("\n✅ Training complete!")
    
    # Copy best model
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        target = Path("models/deepdarts_trained.pt")
        target.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, target)
        print(f"\n📦 Model copied to: {target}")
        print(f"\n🎉 Ready to use! Restart BullSight to activate.")
        return 0
    else:
        print("❌ Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(train_synthetic_model())
