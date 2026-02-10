#!/usr/bin/env python3
"""
Collect real dartboard images from DUAL STEREO CAMERAS for training.
This script captures synchronized images from both cameras.
"""

import cv2
from pathlib import Path
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.dual_camera_manager import DualCameraManager

def collect_stereo_dart_images(output_dir="training_data/raw_images", num_images=50):
    """Capture stereo dartboard images from dual camera system."""
    
    output_left = Path(output_dir) / "left"
    output_right = Path(output_dir) / "right"
    output_left.mkdir(parents=True, exist_ok=True)
    output_right.mkdir(parents=True, exist_ok=True)
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║        Stereo Dart Image Collection for Training               ║
║                                                                ║
║  Instructions:                                                 ║
║  1. Point BOTH cameras at dartboard                           ║
║  2. Throw 1-3 darts                                           ║
║  3. Press SPACE to capture BOTH images simultaneously         ║
║  4. Press 'q' to quit                                         ║
║                                                                ║
║  📊 Collect 50-100 stereo pairs with darts in different       ║
║  positions for best training results                          ║
║                                                                ║
║  Both LEFT and RIGHT images will be saved together            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize dual camera system
    print("📹 Initializing dual camera system...")
    camera_mgr = DualCameraManager()
    
    if not camera_mgr.start():
        print("❌ Dual camera system failed to initialize")
        return
    
    print(f"✅ Dual camera initialized")
    print(f"📐 Resolution: {camera_mgr.resolution[0]}x{camera_mgr.resolution[1]}\n")
    
    captured = 0
    
    print(f"📷 Ready for capture. Waiting for input...\n")
    
    while captured < num_images:
        # Capture from both cameras
        stereo_frame = camera_mgr.capture_stereo()
        
        if stereo_frame is None:
            print("❌ Failed to capture frames")
            break
        
        left_frame = stereo_frame.left
        right_frame = stereo_frame.right
        
        if left_frame is None or right_frame is None:
            print("❌ One or both frames are None")
            continue
        
        # Create side-by-side display
        display = cv2.hconcat([left_frame, right_frame])
        display_frame = display.copy()
        
        # Add info text
        cv2.putText(display_frame, f"Stereo Pair: {captured}/{num_images}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "👈 LEFT", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "RIGHT 👉", 
                   (camera_mgr.resolution[0] + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=capture, Q=quit", 
                   (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Stereo Dart Image Collector", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar - capture both
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Save LEFT image
            left_filename = f"dart_image_{captured:04d}_{timestamp}_LEFT.jpg"
            left_filepath = output_left / left_filename
            cv2.imwrite(str(left_filepath), left_frame)
            
            # Save RIGHT image
            right_filename = f"dart_image_{captured:04d}_{timestamp}_RIGHT.jpg"
            right_filepath = output_right / right_filename
            cv2.imwrite(str(right_filepath), right_frame)
            
            print(f"✅ Captured stereo pair {captured+1}/{num_images}")
            print(f"   📁 LEFT:  {left_filename}")
            print(f"   📁 RIGHT: {right_filename}\n")
            captured += 1
        
        elif key == ord('q'):  # Quit
            break
    
    camera_mgr.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"✅ Collection complete!")
    print(f"{'='*60}")
    print(f"📊 Total stereo pairs captured: {captured}")
    print(f"📁 LEFT images saved to:  {output_left}")
    print(f"📁 RIGHT images saved to: {output_right}")
    print()
    print("Next steps:")
    print("1. Annotate images with bounding boxes:")
    print("   pip install labelimg")
    print("   labelimg")
    print()
    print("2. Then fine-tune the model:")
    print("   python train_real_model.py")
    print()

if __name__ == "__main__":
    num_images = 50
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except:
            pass
    
    collect_stereo_dart_images(num_images=num_images)
