#!/usr/bin/env python3
"""
Quick manual dart labeling tool. 
Draw bounding boxes around darts in images to create YOLO annotations.
Much faster than LabelImg for this specific use case.
"""

import cv2
import os
from pathlib import Path
from typing import List, Tuple

class QuickDartLabeler:
    """Interactive tool to quickly label darts in training images."""
    
    def __init__(self):
        self.images_dir = Path("training_data/finetuning_data/images/train")
        self.labels_dir = Path("training_data/finetuning_data/labels/train")
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.drawing = False
        self.start_point = None
        self.rectangles: List[Tuple[int, int, int, int]] = []
        self.current_image = None
        self.current_img_copy = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = self.current_img_copy.copy()
            cv2.rectangle(img, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Quick Dart Labeler - Draw rectangles", img)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x_min, y_min = self.start_point
            x_max, y_max = x, y
            
            # Ensure correct order
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            
            self.rectangles.append((x_min, y_min, x_max, y_max))
            
            # Draw on persistent copy
            cv2.rectangle(self.current_img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Quick Dart Labeler - Draw rectangles", self.current_img_copy)
    
    def label_images(self):
        """Main labeling interface."""
        # Get all images
        image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║           Quick Dart Labeler                                   ║
║                                                                ║
║  Instructions:                                                 ║
║  - CLICK & DRAG to draw rectangles around darts              ║
║  - SPACE: Save and next image                                 ║
║  - Z: Undo last rectangle                                     ║
║  - C: Clear all rectangles                                    ║
║  - Q: Quit                                                    ║
║                                                                ║
║  Images: {len(image_files)}                                          
╚════════════════════════════════════════════════════════════════╝
        """)
        
        labeled_count = 0
        skipped_count = 0
        
        for i, img_path in enumerate(image_files, 1):
            # Check if already labeled
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists() and label_path.stat().st_size > 0:
                print(f"[{i:3d}/{len(image_files)}] ✓ {img_path.name} (already labeled, skip)")
                continue
            
            # Load and display image
            self.current_image = cv2.imread(str(img_path))
            if self.current_image is None:
                print(f"[{i:3d}/{len(image_files)}] ✗ {img_path.name} (read failed, skip)")
                continue
            
            self.current_img_copy = self.current_image.copy()
            self.rectangles = []
            
            cv2.namedWindow("Quick Dart Labeler - Draw rectangles", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Quick Dart Labeler - Draw rectangles", 800, 600)
            cv2.setMouseCallback("Quick Dart Labeler - Draw rectangles", self.mouse_callback)
            cv2.imshow("Quick Dart Labeler - Draw rectangles", self.current_img_copy)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE: Save and continue
                    # Save labels
                    height, width = self.current_image.shape[:2]
                    label_lines = []
                    
                    for x_min, y_min, x_max, y_max in self.rectangles:
                        center_x = (x_min + x_max) / 2.0 / width
                        center_y = (y_min + y_max) / 2.0 / height
                        w = (x_max - x_min) / width
                        h = (y_max - y_min) / height
                        
                        label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")
                    
                    label_path.write_text("\n".join(label_lines))
                    
                    status = f"✓ {img_path.name}: {len(self.rectangles)} darts"
                    print(f"[{i:3d}/{len(image_files)}] {status}")
                    labeled_count += 1
                    break
                
                elif key == ord('z'):  # Z: Undo
                    if self.rectangles:
                        self.rectangles.pop()
                        self.current_img_copy = self.current_image.copy()
                        for x_min, y_min, x_max, y_max in self.rectangles:
                            cv2.rectangle(self.current_img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.imshow("Quick Dart Labeler - Draw rectangles", self.current_img_copy)
                
                elif key == ord('c'):  # C: Clear
                    self.rectangles = []
                    self.current_img_copy = self.current_image.copy()
                    cv2.imshow("Quick Dart Labeler - Draw rectangles", self.current_img_copy)
                
                elif key == ord('q'):  # Q: Quit
                    cv2.destroyAllWindows()
                    print("\n⚠️  Labeling stopped by user")
                    break
            
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        print(f"\n✅ Labeling complete!")
        print(f"   Labeled: {labeled_count}")
        print(f"   Total: {len(image_files)}")
        print(f"\n📊 Dataset ready for training!")

if __name__ == "__main__":
    labeler = QuickDartLabeler()
    labeler.label_images()
