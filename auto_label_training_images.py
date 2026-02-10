#!/usr/bin/env python3
"""
Auto-generate YOLO annotations for training images using existing dart detector.
This allows training without manual annotation using LabelImg.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_annotations():
    """
    Auto-generate YOLO format annotations from training images.
    Uses the current dart detector to find darts and create label files.
    """
    
    # Look in training_data/raw_images (the folder where collect_training_images saves files)
    images_dir = Path("training_data/raw_images")
    
    if not images_dir.exists():
        logger.error(f"❌ Training data folder not found: {images_dir}")
        return False
    
    # Get all images from main dir AND subdirs (left/, right/)
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    image_files += sorted(images_dir.glob("*/*.jpg")) + sorted(images_dir.glob("*/*.png"))
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if not image_files:
        logger.error("❌ No training images found in:")
        logger.error(f"   {images_dir}")
        logger.error(f"   {images_dir}/left/")
        logger.error(f"   {images_dir}/right/")
        return False
    
    logger.info(f"🔍 Auto-labeling {len(image_files)} training images...")
    logger.info(f"   From: {images_dir}")
    logger.info("   Using current dart detector")
    print()
    
    # Import detector
    try:
        from src.vision.ml_dart_detector import MLDartDetector
    except ImportError:
        logger.error("Failed to import MLDartDetector")
        return False
    
    detector = MLDartDetector()
    
    if not detector.is_available():
        logger.error("❌ ML detector not available")
        return False
    
    # Set lower confidence threshold to catch more darts
    detector.confidence_threshold = 0.3
    
    successful = 0
    skipped = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"  Skipped: {img_path.name} (read failed)")
                skipped += 1
                continue
            
            height, width = image.shape[:2]
            
            # Detect darts
            detections = detector.detect_multiple(image, max_darts=10)
            
            if not detections:
                # Create empty label file (no darts detected) - save next to image
                label_path = img_path.with_suffix(".txt")
                label_path.touch()
                logger.info(f"  [{i:3d}/{len(image_files)}] {img_path.name}: No darts (empty label)")
            else:
                # Create YOLO format annotations
                # Format: class_id center_x center_y width height (normalized 0-1)
                label_lines = []
                
                for x, y, conf, class_name in detections:
                    # Assume dart is a small point - create bounding box
                    # Dart size estimate: ~30 pixels (adjust if needed)
                    dart_size = 20
                    
                    x_min = max(0, x - dart_size)
                    y_min = max(0, y - dart_size)
                    x_max = min(width, x + dart_size)
                    y_max = min(height, y + dart_size)
                    
                    # Normalize to 0-1
                    center_x = (x_min + x_max) / 2.0 / width
                    center_y = (y_min + y_max) / 2.0 / height
                    w = (x_max - x_min) / width
                    h = (y_max - y_min) / height
                    
                    # YOLO format: class_id center_x center_y width height
                    label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")
                
                # Write label file next to image
                label_path = img_path.with_suffix(".txt")
                label_path.write_text("\n".join(label_lines))
                
                logger.info(f"  [{i:3d}/{len(image_files)}] {img_path.name}: {len(detections)} darts detected")
                successful += 1
        
        except Exception as e:
            logger.error(f"  Error processing {img_path.name}: {e}")
            skipped += 1
    
    print()
    logger.info(f"✅ Auto-labeling complete!")
    logger.info(f"   ✓ Annotated: {successful}")
    logger.info(f"   ⊘ Skipped: {skipped}")
    logger.info(f"   📁 Labels saved to: {labels_dir}")
    print()
    logger.info("📊 Dataset ready for training!")
    logger.info("   Next: Use 'Finetune Model' button in ML Demo → Start Training")
    
    return True

if __name__ == "__main__":
    success = generate_annotations()
    exit(0 if success else 1)
