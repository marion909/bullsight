#!/usr/bin/env python3
"""
Train YOLO model on real annotated dart images.
Requires images to be annotated with LabelImg in YOLO format.
"""

import os
import sys
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

def prepare_dataset_structure(images_dir="training_data/raw_images", 
                             output_dir="training_data/yolo_dataset"):
    """Convert raw images into YOLO dataset structure."""
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    if not images_path.exists():
        print(f"❌ Images not found: {images_dir}")
        print("Run collect_training_images.py first")
        return False
    
    # Create YOLO structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get image and label files - search in main dir AND subdirs (left/, right/)
    image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))
    image_files += sorted(images_path.glob("*/*.jpg")) + sorted(images_path.glob("*/*.png"))
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        print(f"   Looked in: {images_dir}/*.jpg, {images_dir}/*/*.jpg, etc.")
        return False
    
    print(f"📊 Found {len(image_files)} images")
    print("   Splitting: 70% train, 20% val, 10% test")
    
    # Split indices - handle small datasets
    total = len(image_files)
    
    # For small datasets (< 10 images), use simple split
    if total < 10:
        print(f"   ⚠️  Small dataset ({total} images) - using 80% train, 20% val (no test split)")
        train_count = int(total * 0.8)
        val_count = total - train_count
        test_count = 0
    else:
        print("   Splitting: 70% train, 20% val, 10% test")
        train_count = int(total * 0.7)
        val_count = int(total * 0.2)
        test_count = total - train_count - val_count
    
    # Copy images and labels
    for i, img_file in enumerate(image_files):
        label_file = img_file.with_suffix('.txt')
        
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
        
        # Copy image
        shutil.copy(img_file, output_path / 'images' / split / img_file.name)
        
        # Copy label if exists
        if label_file.exists():
            shutil.copy(label_file, output_path / 'labels' / split / label_file.name)
        else:
            # Create empty label (no darts detected)
            (output_path / 'labels' / split / label_file.name).touch()
    
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
    yaml_path.write_text(data_yaml)
    
    print(f"✓ Dataset prepared: {output_dir}")
    return True

def train_real_model(dataset_dir="training_data/yolo_dataset", epochs=50):
    """Train YOLO model on real annotated dart images."""
    
    dataset_path = Path(dataset_dir) / 'data.yaml'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run: python train_real_model.py --prepare")
        return
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║      Training on Real Dart Images                              ║
║                                                                ║
║  Dataset: {dataset_dir}                   
║  Epochs: {epochs} (early stopping at 15 epochs no improvement)  
║  Model: YOLOv8 Nano (CPU mode)                                ║
║  Expected time: 1-3 hours                                     ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
    print("📚 Starting training...")
    results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=15,
        device='cpu',
        save=True,
        project='bullsight_training',
        name='real_dart_detector',
        verbose=False,
        plots=False,
    )
    
    print("\n✅ Training complete!")
    print(f"📊 Results: bullsight_training/real_dart_detector/")
    
    # Copy best model
    best_model = Path('bullsight_training/real_dart_detector/weights/best.pt')
    if best_model.exists():
        target = Path('models/deepdarts_real.pt')
        shutil.copy(best_model, target)
        print(f"📦 Copied to: {target}")
        print(f"\n✨ Restart BullSight to use new model!")
        return True
    
    return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO on real dart images')
    parser.add_argument('--prepare', action='store_true', 
                       help='Prepare dataset structure from raw images')
    parser.add_argument('--train', action='store_true', 
                       help='Train model on dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--images', type=str, default='training_data/raw_images',
                       help='Directory with raw images')
    parser.add_argument('--dataset', type=str, default='training_data/yolo_dataset',
                       help='Output YOLO dataset directory')
    
    args = parser.parse_args()
    
    if not args.prepare and not args.train:
        args.prepare = True
        args.train = True
    
    if args.prepare:
        print("📁 Preparing dataset structure...")
        if prepare_dataset_structure(args.images, args.dataset):
            print("✓ Dataset ready")
        else:
            return 1
    
    if args.train:
        train_real_model(args.dataset, args.epochs)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
