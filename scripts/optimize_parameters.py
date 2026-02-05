"""
Automatic parameter optimization using test images with ground truth.

Uses captured test images with known dart positions to find optimal
detection parameters through grid search.

Author: Mario Neuhauser
"""

import sys
sys.path.insert(0, '.')

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.vision.dart_detector import DartDetector


@dataclass
class TestImage:
    """Test image with ground truth data."""
    image_path: str
    dart_x: int
    dart_y: int
    segment: str
    score: int
    description: str


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    threshold: int
    min_area: int
    max_area: int
    blur_kernel: int
    accuracy: float
    avg_distance_error: float
    detection_rate: float


class ParameterOptimizer:
    """
    Optimizes detection parameters using test images.
    
    Uses grid search to find parameter combination that maximizes
    detection accuracy on test dataset.
    """
    
    def __init__(self, test_data_path: str = "config/test_images.json"):
        """
        Initialize optimizer.
        
        Args:
            test_data_path: Path to test images JSON file
        """
        self.test_data_path = Path(test_data_path)
        self.test_images: List[TestImage] = []
        self.reference_image: Optional[np.ndarray] = None
        
    def load_test_data(self) -> bool:
        """
        Load test images and ground truth data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.test_data_path.exists():
            print(f"❌ Test data file not found: {self.test_data_path}")
            print("   Use Calibration → 'Capture Test Image' to create test data")
            return False
        
        try:
            with open(self.test_data_path, 'r') as f:
                data = json.load(f)
            
            self.test_images = [
                TestImage(**item) for item in data.get('test_images', [])
            ]
            
            print(f"✅ Loaded {len(self.test_images)} test images")
            return len(self.test_images) > 0
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def load_reference_image(self, reference_path: str = "config/reference_board.jpg") -> bool:
        """
        Load reference image for detection.
        
        Args:
            reference_path: Path to reference image
            
        Returns:
            True if successful, False otherwise
        """
        ref_path = Path(reference_path)
        if not ref_path.exists():
            print(f"❌ Reference image not found: {ref_path}")
            print("   Use Calibration → 'Capture Reference Image' first")
            return False
        
        self.reference_image = cv2.imread(str(ref_path))
        if self.reference_image is None:
            print(f"❌ Could not load reference image")
            return False
        
        print(f"✅ Loaded reference image: {ref_path}")
        return True
    
    def evaluate_parameters(
        self,
        threshold: int,
        min_area: int,
        max_area: int,
        blur_kernel: int
    ) -> OptimizationResult:
        """
        Evaluate detection accuracy with given parameters.
        
        Args:
            threshold: Threshold value
            min_area: Minimum contour area
            max_area: Maximum contour area
            blur_kernel: Blur kernel size
            
        Returns:
            Optimization result with accuracy metrics
        """
        detector = DartDetector(
            min_contour_area=min_area,
            max_contour_area=max_area,
            blur_kernel_size=blur_kernel,
            threshold_value=threshold
        )
        detector.set_reference_image(self.reference_image)
        
        total_tests = len(self.test_images)
        successful_detections = 0
        total_distance_error = 0.0
        
        for test_img in self.test_images:
            # Load test image
            img_path = Path(test_img.image_path)
            if not img_path.exists():
                continue
            
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Detect dart
            result = detector.detect_dart(frame)
            
            if result:
                # Calculate distance error
                distance_error = np.sqrt(
                    (result.x - test_img.dart_x) ** 2 +
                    (result.y - test_img.dart_y) ** 2
                )
                total_distance_error += distance_error
                
                # Consider detection successful if within 50 pixels
                if distance_error < 50:
                    successful_detections += 1
        
        detection_rate = successful_detections / total_tests if total_tests > 0 else 0
        avg_distance = total_distance_error / successful_detections if successful_detections > 0 else float('inf')
        
        # Accuracy combines detection rate and position accuracy
        accuracy = detection_rate * (1.0 - min(avg_distance / 100.0, 1.0))
        
        return OptimizationResult(
            threshold=threshold,
            min_area=min_area,
            max_area=max_area,
            blur_kernel=blur_kernel,
            accuracy=accuracy,
            avg_distance_error=avg_distance,
            detection_rate=detection_rate
        )
    
    def grid_search(
        self,
        threshold_range: Tuple[int, int, int] = (20, 45, 5),
        min_area_range: Tuple[int, int, int] = (80, 250, 30),
        max_area_range: Tuple[int, int, int] = (3000, 7000, 1000),
        blur_kernel_range: Tuple[int, int, int] = (3, 9, 2)
    ) -> List[OptimizationResult]:
        """
        Perform grid search over parameter space.
        
        Args:
            threshold_range: (start, stop, step) for threshold
            min_area_range: (start, stop, step) for min area
            max_area_range: (start, stop, step) for max area
            blur_kernel_range: (start, stop, step) for blur kernel
            
        Returns:
            List of results sorted by accuracy (best first)
        """
        results = []
        
        # Generate parameter combinations
        thresholds = range(*threshold_range)
        min_areas = range(*min_area_range)
        max_areas = range(*max_area_range)
        blur_kernels = range(blur_kernel_range[0], blur_kernel_range[1], blur_kernel_range[2])
        # Ensure odd kernel sizes
        blur_kernels = [k if k % 2 == 1 else k + 1 for k in blur_kernels]
        
        total_combinations = len(thresholds) * len(min_areas) * len(max_areas) * len(blur_kernels)
        
        print(f"\n🔍 Grid Search: Testing {total_combinations} parameter combinations...")
        print(f"   Threshold: {threshold_range}")
        print(f"   Min Area: {min_area_range}")
        print(f"   Max Area: {max_area_range}")
        print(f"   Blur Kernel: {blur_kernel_range}")
        print()
        
        count = 0
        for threshold in thresholds:
            for min_area in min_areas:
                for max_area in max_areas:
                    for blur_kernel in blur_kernels:
                        count += 1
                        
                        # Skip invalid combinations
                        if min_area >= max_area:
                            continue
                        
                        result = self.evaluate_parameters(
                            threshold, min_area, max_area, blur_kernel
                        )
                        results.append(result)
                        
                        # Progress indicator
                        if count % 10 == 0:
                            print(f"   Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")
        
        # Sort by accuracy (best first)
        results.sort(key=lambda r: r.accuracy, reverse=True)
        
        return results


def main():
    """Run parameter optimization."""
    print("🎯 BullSight Parameter Optimization")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Load data
    if not optimizer.load_reference_image():
        return
    
    if not optimizer.load_test_data():
        return
    
    print(f"\n📊 Test Dataset: {len(optimizer.test_images)} images")
    
    # Run grid search
    results = optimizer.grid_search(
        threshold_range=(25, 40, 5),      # Test: 25, 30, 35
        min_area_range=(100, 200, 50),    # Test: 100, 150
        max_area_range=(4000, 6000, 1000), # Test: 4000, 5000
        blur_kernel_range=(5, 7, 2)       # Test: 5, 7
    )
    
    # Display top 10 results
    print("\n" + "=" * 50)
    print("🏆 Top 10 Parameter Combinations")
    print("=" * 50)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n#{i}")
        print(f"   Accuracy:       {result.accuracy:.2%}")
        print(f"   Detection Rate: {result.detection_rate:.2%}")
        print(f"   Avg Distance:   {result.avg_distance_error:.1f} px")
        print(f"   Parameters:")
        print(f"      threshold_value={result.threshold}")
        print(f"      min_contour_area={result.min_area}")
        print(f"      max_contour_area={result.max_area}")
        print(f"      blur_kernel_size={result.blur_kernel}")
    
    # Save best parameters
    if results:
        best = results[0]
        output = {
            "optimized_parameters": {
                "threshold_value": best.threshold,
                "min_contour_area": best.min_area,
                "max_contour_area": best.max_area,
                "blur_kernel_size": best.blur_kernel
            },
            "metrics": {
                "accuracy": best.accuracy,
                "detection_rate": best.detection_rate,
                "avg_distance_error": best.avg_distance_error
            }
        }
        
        output_path = Path("config/optimized_parameters.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Best parameters saved to: {output_path}")
        print("\nYou can now update your DartDetector initialization with these values!")


if __name__ == "__main__":
    main()
