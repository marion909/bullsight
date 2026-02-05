#!/usr/bin/env python3
"""
Test dart detection with different parameter sets.
Helps find optimal parameters for your setup.

Author: Mario Neuhauser
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import time


# Parameter sets to test
TEST_PARAMETERS = [
    {
        "name": "Default",
        "threshold": 30,
        "min_area": 100,
        "max_area": 5000,
        "blur": 5
    },
    {
        "name": "High Sensitivity",
        "threshold": 25,
        "min_area": 80,
        "max_area": 5000,
        "blur": 5
    },
    {
        "name": "Low Sensitivity",
        "threshold": 40,
        "min_area": 150,
        "max_area": 4000,
        "blur": 7
    },
    {
        "name": "Bright Environment",
        "threshold": 35,
        "min_area": 100,
        "max_area": 5000,
        "blur": 5
    },
    {
        "name": "Dark Environment",
        "threshold": 25,
        "min_area": 100,
        "max_area": 5000,
        "blur": 7
    }
]


def main():
    """Run detection tests with different parameters."""
    print("=" * 60)
    print("🎯 BullSight Detection Parameter Testing")
    print("=" * 60)
    print()
    
    # Check for reference image
    ref_path = Path("config/reference_board.jpg")
    if not ref_path.exists():
        print("❌ No reference image found!")
        print("   Please run: python scripts/capture_reference.py")
        return 1
    
    print("This will test different parameter sets.")
    print()
    print("Instructions:")
    print("  1. Throw a dart at the dartboard")
    print("  2. Wait for the countdown")
    print("  3. Keep still during capture")
    print("  4. Review results for each parameter set")
    print()
    
    input("Press ENTER when you have thrown a dart...")
    print()
    
    # Initialize components
    print("📸 Initializing camera...")
    camera = CameraManager()
    
    try:
        camera.start()
        print("✅ Camera started")
        print()
        
        # Wait for stabilization
        print("⏳ Waiting for camera stabilization (3 seconds)...")
        time.sleep(3)
        
        # Capture test frame
        print("📸 Capturing test image...")
        test_frame = camera.capture_frame()
        
        if test_frame is None:
            print("❌ Failed to capture frame!")
            return 1
        
        print("✅ Test image captured")
        print()
        
        # Load reference for all detectors
        print("📂 Loading reference image...")
        reference_detector = DartDetector()
        reference_detector.load_reference_from_file(str(ref_path))
        reference_image = reference_detector.reference_image
        print("✅ Reference loaded")
        print()
        
        # Test each parameter set
        print("=" * 60)
        print("Testing Parameter Sets")
        print("=" * 60)
        print()
        
        results = []
        
        for i, params in enumerate(TEST_PARAMETERS, 1):
            print(f"Test {i}/{len(TEST_PARAMETERS)}: {params['name']}")
            print("-" * 40)
            
            # Create detector with test parameters
            detector = DartDetector(
                min_contour_area=params["min_area"],
                max_contour_area=params["max_area"],
                blur_kernel_size=params["blur"],
                threshold_value=params["threshold"]
            )
            detector.set_reference_image(reference_image)
            
            # Print parameters
            print(f"  Threshold:    {params['threshold']}")
            print(f"  Min Area:     {params['min_area']}")
            print(f"  Max Area:     {params['max_area']}")
            print(f"  Blur Kernel:  {params['blur']}")
            print()
            
            # Detect dart
            result = detector.detect_dart(test_frame)
            
            # Print result
            if result:
                print(f"  ✅ Dart detected!")
                print(f"     Position:   ({result.x}, {result.y})")
                print(f"     Confidence: {result.confidence:.2f}")
                print(f"     Area:       {result.contour_area:.0f} pixels")
                
                results.append({
                    "name": params["name"],
                    "detected": True,
                    "position": (result.x, result.y),
                    "confidence": result.confidence,
                    "area": result.contour_area
                })
            else:
                print(f"  ❌ No dart detected")
                results.append({
                    "name": params["name"],
                    "detected": False
                })
            
            print()
        
        # Summary
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print()
        
        detected_count = sum(1 for r in results if r["detected"])
        print(f"Detection Rate: {detected_count}/{len(TEST_PARAMETERS)} ({detected_count/len(TEST_PARAMETERS)*100:.0f}%)")
        print()
        
        if detected_count > 0:
            print("Successful Detections:")
            for r in results:
                if r["detected"]:
                    print(f"  • {r['name']:<20} → ({r['position'][0]:4d}, {r['position'][1]:4d})  "
                          f"Conf: {r['confidence']:.2f}  Area: {r['area']:.0f}")
            print()
            
            # Find most consistent detection
            if detected_count >= 2:
                positions = [r["position"] for r in results if r["detected"]]
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                print(f"Average Position: ({avg_x:.0f}, {avg_y:.0f})")
                
                # Find closest to average
                min_dist = float('inf')
                best_param = None
                for r in results:
                    if r["detected"]:
                        dist = ((r["position"][0] - avg_x)**2 + (r["position"][1] - avg_y)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_param = r["name"]
                
                print(f"Most Consistent: {best_param}")
                print()
        
        print("Recommendations:")
        if detected_count == 0:
            print("  ⚠️  No detections! Try:")
            print("     - Check if dart is actually in frame")
            print("     - Improve lighting")
            print("     - Recapture reference image")
            print("     - Manually adjust parameters with live_tuning.py")
        elif detected_count < len(TEST_PARAMETERS) // 2:
            print("  ⚠️  Low detection rate. Try:")
            print("     - Use 'High Sensitivity' parameters")
            print("     - Check lighting conditions")
            print("     - Recapture reference image in current lighting")
        else:
            print("  ✅ Good detection rate!")
            print(f"     - Use '{results[0]['name']}' or similar parameters")
            print("     - Fine-tune with: python scripts/live_tuning.py")
        
        print()
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        print()
        print("🧹 Cleaning up...")
        camera.stop()
        print("✅ Done")


if __name__ == "__main__":
    exit(main())
