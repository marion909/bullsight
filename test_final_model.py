#!/usr/bin/env python3
"""Test the newly trained dart detector model."""

from src.vision.ml_dart_detector import MLDartDetector
import cv2
from pathlib import Path
import random

print('🎯 Testing FINAL trained model (30 epochs, smart annotations)...')
print()

# Load newly trained model
detector = MLDartDetector(model_path='models/deepdarts_trained.pt')
print('✅ Model loaded from: models/deepdarts_trained.pt')
print()

# Test on 20 random training images
images_dir = Path('training_data/finetuning_data/images/train')
test_images = random.sample(sorted(images_dir.glob('*.jpg')), min(20, 201))

print('Detection Results on 20 Random Images:')
print('━' * 70)

total_darts = 0
detected_count = 0
all_confidences = []

for i, img_path in enumerate(test_images, 1):
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    
    dets = detector.detect_multiple(image, max_darts=10)
    total_darts += len(dets)
    if len(dets) > 0:
        detected_count += 1
    
    status = '✅' if len(dets) > 0 else '❓'
    print(f'[{i:2d}] {status} {img_path.name[:45]:45s} → {len(dets)} dart{"s" if len(dets) != 1 else ""}', end='')
    
    if dets:
        confs = [c for x, y, c, _ in dets]
        all_confidences.extend(confs)
        avg_conf = sum(confs) / len(confs)
        print(f', avg_conf: {avg_conf:.2f}')
    else:
        print()

print('━' * 70)
print()
print('📊 Final Results:')
print(f'   ✅ Images with detections:    {detected_count}/20 ({detected_count/20*100:.0f}%)')
print(f'   🎯 Total darts detected:       {total_darts}')

if all_confidences:
    avg_all_conf = sum(all_confidences) / len(all_confidences)
    min_conf = min(all_confidences)
    max_conf = max(all_confidences)
    print(f'   📈 Average confidence:        {avg_all_conf:.3f}')
    print(f'   🔝 Confidence range:          {min_conf:.3f} - {max_conf:.3f}')

print()
if detected_count >= 18:
    print('✨ EXCELLENT RESULTS! Model is production-ready! 🚀')
elif detected_count >= 15:
    print('✅ VERY GOOD! Model works reliably!')
else:
    print('⚠️  Model needs more training or data')
