#!/usr/bin/env python3
"""
Script to capture reference image for dart detection.
Run this with an empty dartboard to create the reference image.

Author: Mario Neuhauser
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vision.camera_manager import CameraManager
from src.vision.dart_detector import DartDetector
import time


def main():
    """Capture reference image for dart detection."""
    print("=" * 60)
    print("🎯 BullSight Reference Image Capture")
    print("=" * 60)
    print()
    print("This will capture a reference image of your dartboard.")
    print()
    print("IMPORTANT:")
    print("  ✅ Remove ALL darts from the board")
    print("  ✅ Ensure good, even lighting")
    print("  ✅ Make sure camera is focused")
    print("  ✅ Keep hands away from the board")
    print()
    
    input("Press ENTER when ready to capture...")
    print()
    
    # Initialize components
    print("📸 Initializing camera...")
    camera = CameraManager(resolution=(1280, 720), enable_autofocus=True)
    detector = DartDetector()
    
    try:
        # Start camera
        camera.start()
        print("✅ Camera started")
        
        # Trigger autofocus and wait
        print("🔍 Focusing camera...")
        camera.trigger_autofocus()
        time.sleep(3)
        print("✅ Focus complete")
        
        # Capture multiple frames and use the middle one
        print("📸 Capturing frames...")
        frames = []
        for i in range(10):
            frame = camera.capture_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.1)
        
        if not frames:
            print("❌ Failed to capture frames!")
            return 1
        
        print(f"✅ Captured {len(frames)} frames")
        
        # Use middle frame (most stable)
        reference_frame = frames[len(frames) // 2]
        
        # Set as reference and save
        print("💾 Saving reference image...")
        detector.set_reference_image(reference_frame)
        
        # Create config directory if it doesn't exist
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        output_path = config_dir / "reference_board.jpg"
        detector.save_reference_to_file(str(output_path))
        
        print(f"✅ Reference image saved to: {output_path}")
        print()
        print("=" * 60)
        print("Success! Reference image captured.")
        print()
        print("Next steps:")
        print("  1. Test detection: python scripts/test_detection.py")
        print("  2. Tune parameters: python scripts/live_tuning.py")
        print("  3. Start application: ./run.sh")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    finally:
        # Clean up
        print()
        print("🧹 Cleaning up...")
        camera.stop()
        print("✅ Camera stopped")


if __name__ == "__main__":
    exit(main())
