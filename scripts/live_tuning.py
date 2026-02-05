#!/usr/bin/env python3
"""
Interactive live tuning tool for dart detection parameters.
Adjust parameters in real-time and see the effects immediately.

Author: Mario Neuhauser
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import cv2
import numpy as np


def main():
    """Run live tuning interface."""
    print("=" * 60)
    print("🎯 BullSight Live Detection Tuning")
    print("=" * 60)
    print()
    print("This tool lets you adjust detection parameters in real-time.")
    print()
    print("Controls:")
    print("  - Adjust trackbars to change parameters")
    print("  - Press 's' to save current parameters")
    print("  - Press 'r' to reset to defaults")
    print("  - Press 'q' to quit")
    print()
    
    # Check for reference image
    ref_path = Path("config/reference_board.jpg")
    if not ref_path.exists():
        print("❌ No reference image found!")
        print("   Please run: python scripts/capture_reference.py")
        return 1
    
    # Initialize components
    print("📸 Initializing...")
    camera = CameraManager()
    detector = DartDetector()
    
    try:
        # Load reference image
        detector.load_reference_from_file(str(ref_path))
        print(f"✅ Loaded reference from: {ref_path}")
        
        # Start camera
        camera.start()
        print("✅ Camera started")
        print()
        print("Opening tuning window...")
        
        # Create window and trackbars
        window_name = "BullSight Live Tuning"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Default values
        defaults = {
            "threshold": 30,
            "min_area": 100,
            "max_area": 5000,
            "blur_kernel": 5
        }
        
        # Create trackbars
        cv2.createTrackbar("Threshold", window_name, defaults["threshold"], 100, lambda x: None)
        cv2.createTrackbar("Min Area", window_name, defaults["min_area"], 1000, lambda x: None)
        cv2.createTrackbar("Max Area", window_name, defaults["max_area"], 10000, lambda x: None)
        cv2.createTrackbar("Blur Kernel", window_name, defaults["blur_kernel"], 15, lambda x: None)
        
        print("✅ Tuning interface ready!")
        print()
        print("Throw a dart and adjust parameters to optimize detection.")
        print()
        
        while True:
            # Get current parameters from trackbars
            threshold = cv2.getTrackbarPos("Threshold", window_name)
            min_area = cv2.getTrackbarPos("Min Area", window_name)
            max_area = cv2.getTrackbarPos("Max Area", window_name)
            blur_kernel = cv2.getTrackbarPos("Blur Kernel", window_name)
            
            # Ensure blur kernel is odd
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            blur_kernel = max(3, blur_kernel)  # Minimum 3
            
            # Update detector parameters
            detector.threshold_value = threshold
            detector.min_contour_area = min_area
            detector.max_contour_area = max_area
            detector.blur_kernel_size = blur_kernel
            
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            # Detect dart
            result = detector.detect_dart(frame)
            
            # Visualize
            vis_frame = detector.visualize_detection(frame, result)
            
            # Add parameter info overlay
            info_y = 30
            line_height = 35
            
            # Background for text
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, 5), (400, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
            
            # Parameters text
            cv2.putText(vis_frame, f"Threshold: {threshold}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += line_height
            
            cv2.putText(vis_frame, f"Min Area: {min_area}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += line_height
            
            cv2.putText(vis_frame, f"Max Area: {max_area}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += line_height
            
            cv2.putText(vis_frame, f"Blur: {blur_kernel}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += line_height
            
            # Detection result
            if result:
                cv2.putText(vis_frame, f"Dart: ({result.x}, {result.y})", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                info_y += line_height
                cv2.putText(vis_frame, f"Conf: {result.confidence:.2f}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(vis_frame, "No dart detected", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow(window_name, vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                print("\n📝 Saving parameters...")
                print(f"   threshold_value = {threshold}")
                print(f"   min_contour_area = {min_area}")
                print(f"   max_contour_area = {max_area}")
                print(f"   blur_kernel_size = {blur_kernel}")
                print("\nUpdate these in your code:")
                print(f"detector = DartDetector(")
                print(f"    min_contour_area={min_area},")
                print(f"    max_contour_area={max_area},")
                print(f"    blur_kernel_size={blur_kernel},")
                print(f"    threshold_value={threshold}")
                print(f")")
                print()
            elif key == ord('r'):
                print("\n🔄 Resetting to defaults...")
                cv2.setTrackbarPos("Threshold", window_name, defaults["threshold"])
                cv2.setTrackbarPos("Min Area", window_name, defaults["min_area"])
                cv2.setTrackbarPos("Max Area", window_name, defaults["max_area"])
                cv2.setTrackbarPos("Blur Kernel", window_name, defaults["blur_kernel"])
                print("✅ Reset complete")
                print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        print("✅ Done")


if __name__ == "__main__":
    exit(main())
