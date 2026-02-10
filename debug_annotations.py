#!/usr/bin/env python3
"""Visualize bounding boxes to debug annotation quality."""

import cv2
from pathlib import Path
import numpy as np

print('🔍 Überprüfen: Sitzen die Bounding Boxes auf den Darts?')
print()

images_dir = Path('training_data/finetuning_data/images/train')
labels_dir = Path('training_data/finetuning_data/labels/train')

# Take first 5 images
for img_path in sorted(images_dir.glob('*.jpg'))[:5]:
    label_path = labels_dir / (img_path.stem + '.txt')
    
    if not label_path.exists():
        print(f'❌ {img_path.name}: Keine Labels gefunden')
        continue
    
    image = cv2.imread(str(img_path))
    if image is None:
        continue
        
    h, w = image.shape[:2]
    
    # Draw annotations
    img_copy = image.copy()
    label_content = label_path.read_text().strip()
    
    box_count = 0
    for line in label_content.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
            
        cls_id, cx, cy, box_w, box_h = map(float, parts)
        
        # Convert from normalized to pixel coords
        x_min = int((cx - box_w/2) * w)
        y_min = int((cy - box_h/2) * h)
        x_max = int((cx + box_w/2) * w)
        y_max = int((cy + box_h/2) * h)
        
        # Clamp to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w-1, x_max)
        y_max = min(h-1, y_max)
        
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        box_size_percent = (box_w * 100, box_h * 100)
        print(f'   Box {box_count+1}: center=({cx:.2f}, {cy:.2f}) size={box_size_percent[0]:.1f}% x {box_size_percent[1]:.1f}%')
        box_count += 1
    
    # Save visualization
    out_path = f'debug_label_{img_path.stem}.jpg'
    cv2.imwrite(out_path, img_copy)
    
    print(f'✓ {img_path.name}: {box_count} boxes')
    print(f'  └─ Saved: {out_path}')
    print()

print('✅ Visualisierungen erstellt!')
print('   Bitte öffnen Sie die debug_label_*.jpg Dateien')
print('   und kontrollieren ob grüne Rechtecke auf Darts sitzen')
