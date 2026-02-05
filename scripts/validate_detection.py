"""
Validate detection quality against test dataset.

Tests current detection parameters against all test images
and provides detailed accuracy report.

Author: Mario Neuhauser
"""

import sys
sys.path.insert(0, '.')

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from src.vision.dart_detector import DartDetector


def load_test_data(json_path: str = "config/test_images.json") -> List[dict]:
    """Load test images data from JSON."""
    path = Path(json_path)
    if not path.exists():
        print(f"❌ Test data not found: {json_path}")
        print("   Use Calibration → 'Capture Test Image' to create test data")
        return []
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data.get('test_images', [])


def validate_detection(
    detector: DartDetector,
    test_images: List[dict]
) -> Tuple[int, int, float, List[dict]]:
    """
    Validate detection against test images.
    
    Args:
        detector: Configured DartDetector
        test_images: List of test image data
        
    Returns:
        Tuple of (successful, total, avg_error, detailed_results)
    """
    successful = 0
    total = len(test_images)
    total_error = 0.0
    results = []
    
    for test_data in test_images:
        img_path = Path(test_data['image_path'])
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️  Could not load: {img_path}")
            continue
        
        # Detect dart
        result = detector.detect_dart(frame)
        
        # Calculate error
        expected_x = test_data['dart_x']
        expected_y = test_data['dart_y']
        
        if result:
            distance_error = np.sqrt(
                (result.x - expected_x) ** 2 +
                (result.y - expected_y) ** 2
            )
            total_error += distance_error
            
            # Consider successful if within 50 pixels
            is_success = distance_error < 50
            if is_success:
                successful += 1
            
            results.append({
                'image': img_path.name,
                'detected': True,
                'expected': (expected_x, expected_y),
                'actual': (result.x, result.y),
                'error': distance_error,
                'success': is_success,
                'segment': test_data.get('segment', 'Unknown'),
                'score': test_data.get('score', 0)
            })
        else:
            results.append({
                'image': img_path.name,
                'detected': False,
                'expected': (expected_x, expected_y),
                'actual': None,
                'error': None,
                'success': False,
                'segment': test_data.get('segment', 'Unknown'),
                'score': test_data.get('score', 0)
            })
    
    avg_error = total_error / successful if successful > 0 else 0
    
    return successful, total, avg_error, results


def print_validation_report(
    successful: int,
    total: int,
    avg_error: float,
    results: List[dict],
    detector: DartDetector
):
    """Print detailed validation report."""
    print("\n" + "=" * 60)
    print("🎯 Detection Validation Report")
    print("=" * 60)
    
    # Current parameters
    print("\n📊 Current Parameters:")
    print(f"   threshold_value      = {detector.threshold_value}")
    print(f"   min_contour_area     = {detector.min_contour_area}")
    print(f"   max_contour_area     = {detector.max_contour_area}")
    print(f"   blur_kernel_size     = {detector.blur_kernel_size}")
    
    # Overall metrics
    detection_rate = successful / total * 100 if total > 0 else 0
    print(f"\n📈 Overall Metrics:")
    print(f"   Test Images:         {total}")
    print(f"   Successful:          {successful}")
    print(f"   Failed:              {total - successful}")
    print(f"   Detection Rate:      {detection_rate:.1f}%")
    print(f"   Avg Position Error:  {avg_error:.1f} pixels")
    
    # Quality rating
    if detection_rate >= 95:
        rating = "🟢 EXCELLENT"
    elif detection_rate >= 85:
        rating = "🟡 GOOD"
    elif detection_rate >= 70:
        rating = "🟠 ACCEPTABLE"
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
    
    print(f"   Quality Rating:      {rating}")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    print(f"{'Image':<25} {'Segment':<8} {'Detected':<10} {'Error (px)':<12} {'Status'}")
    print("-" * 60)
    
    for r in results:
        status_icon = "✅" if r['success'] else ("❌" if not r['detected'] else "⚠️ ")
        detected_str = "Yes" if r['detected'] else "No"
        error_str = f"{r['error']:.1f}" if r['error'] is not None else "N/A"
        
        print(f"{r['image']:<25} {r['segment']:<8} {detected_str:<10} {error_str:<12} {status_icon}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if detection_rate < 85:
        print("   - Try running 'python scripts/optimize_parameters.py'")
        print("   - Capture more test images in different conditions")
        print("   - Check lighting consistency")
    elif avg_error > 30:
        print("   - Position accuracy could be improved")
        print("   - Consider fine-tuning threshold and area parameters")
    else:
        print("   - Detection quality is good!")
        print("   - Capture more test images to validate further")


def main():
    """Run validation."""
    print("🎯 BullSight Detection Validation")
    print("=" * 60)
    
    # Load test data
    test_images = load_test_data()
    if not test_images:
        return
    
    print(f"✅ Loaded {len(test_images)} test images")
    
    # Load reference image
    ref_path = Path("config/reference_board.jpg")
    if not ref_path.exists():
        print(f"❌ Reference image not found: {ref_path}")
        print("   Use Calibration → 'Capture Reference Image' first")
        return
    
    reference = cv2.imread(str(ref_path))
    print(f"✅ Loaded reference image")
    
    # Initialize detector with current parameters
    detector = DartDetector(
        min_contour_area=100,
        max_contour_area=5000,
        blur_kernel_size=5,
        threshold_value=30
    )
    detector.set_reference_image(reference)
    
    print(f"✅ Detector initialized")
    print(f"\n🔍 Validating detection...")
    
    # Run validation
    successful, total, avg_error, results = validate_detection(detector, test_images)
    
    # Print report
    print_validation_report(successful, total, avg_error, results, detector)


if __name__ == "__main__":
    main()
