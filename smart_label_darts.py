#!/usr/bin/env python3
"""
Smart dart labeling tool.
Zeichnen Sie nur die Dart-SPITZE, und das Tool erstellt automatisch große Bounding Boxes.
"""

import cv2
from pathlib import Path
from typing import List, Tuple

class SmartDartLabeler:
    """Intelligenter Labeling-Tool: Zeichne die Spitze, Box wird auto-vergrößert."""
    
    def __init__(self):
        self.images_dir = Path("training_data/raw_images")
        self.labels_dir = None  # Will be set per image (next to image)
        
        self.drawing = False
        self.points: List[Tuple[int, int]] = []
        self.current_image = None
        self.current_img_copy = None
        self.current_image_path = None
        
        # Dart size relative to image
        # Ein Dart ist üblicherweise ca. 10-15% der Dartboard-Breite
        self.DART_WIDTH_PERCENT = 0.12   # 12% of image width
        self.DART_HEIGHT_PERCENT = 0.18  # 18% of image height
        
    def mouse_callback(self, event, x, y, flags, param):
        """Click to mark dart center/tip."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            # Draw circle at click point
            cv2.circle(self.current_img_copy, (x, y), 5, (0, 0, 255), -1)
            
            # Draw predicted bounding box
            h, w = self.current_image.shape[:2]
            box_w_px = int(w * self.DART_WIDTH_PERCENT)
            box_h_px = int(h * self.DART_HEIGHT_PERCENT)
            
            x_min = x - box_w_px // 2
            y_min = y - box_h_px // 2
            x_max = x + box_w_px // 2
            y_max = y + box_h_px // 2
            
            cv2.rectangle(self.current_img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            cv2.imshow("Smart Dart Labeler", self.current_img_copy)
    
    def label_images(self):
        """Main labeling interface."""
        # Get images from main dir AND subdirs (left/, right/)
        image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        image_files += sorted(self.images_dir.glob("*/*.jpg")) + sorted(self.images_dir.glob("*/*.png"))
        image_files = sorted(set(image_files))  # Remove duplicates and sort
        
        if not image_files:
            print(f"❌ No images found in {self.images_dir}")
            return
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║        Smart Dart Labeler - Intelligente Annotation            ║
║                                                                ║
║  Wie es funktioniert:                                         ║
║  1. CLICK auf die Dart-SPITZE (roter Punkt)                   ║
║  2. Grünes RECHTECK wird automatisch gezeichnet               ║
║  3. SPACE: Speichern und nächstes Bild                        ║
║  4. Z: Letzten Dart löschen                                   ║
║  5. C: Alle Darts löschen                                     ║
║  6. Q: Abbrechen                                              ║
║                                                                ║
║  Images zu beschriften: {len(image_files)}                            
╚════════════════════════════════════════════════════════════════╝
        """)
        
        for i, img_path in enumerate(image_files, 1):
            # Label file is saved next to image
            label_path = img_path.with_suffix(".txt")
            label_content = label_path.read_text().strip() if label_path.exists() else ""
            
            # Check if labels look reasonable (not old tiny annotations)
            looks_good = False
            if label_content:
                for line in label_content.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) == 5:
                            w = float(parts[3])
                            if w > 0.10:  # Box is at least 10% wide (good!)
                                looks_good = True
                                break
            
            if looks_good:
                print(f"[{i:3d}/{len(image_files)}] ✓ {img_path.name} (already good, skip)")
                continue
            
            # Load image
            self.current_image = cv2.imread(str(img_path))
            if self.current_image is None:
                print(f"[{i:3d}/{len(image_files)}] ✗ {img_path.name} (read failed)")
                continue
            
            self.current_image_path = img_path
            self.current_img_copy = self.current_image.copy()
            self.points = []
            
            cv2.namedWindow("Smart Dart Labeler", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Smart Dart Labeler", 1000, 750)
            cv2.setMouseCallback("Smart Dart Labeler", self.mouse_callback)
            cv2.imshow("Smart Dart Labeler", self.current_img_copy)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE: Save
                    if self.points:  # Only save if points exist
                        h, w = self.current_image.shape[:2]
                        label_lines = []
                        
                        for x, y in self.points:
                            # Convert pixel coords to normalized YOLO format
                            cx = x / w
                            cy = y / h
                            box_w = self.DART_WIDTH_PERCENT
                            box_h = self.DART_HEIGHT_PERCENT
                            
                            label_lines.append(f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
                        
                        # Save label file next to image
                        label_path = self.current_image_path.with_suffix(".txt")
                        label_path.write_text("\n".join(label_lines))
                        print(f"[{i:3d}/{len(image_files)}] ✓ {img_path.name}: {len(self.points)} darts")
                    else:
                        print(f"[{i:3d}/{len(image_files)}] ⊘ {img_path.name}: skipped (no darts)")
                    break
                
                elif key == ord('z'):  # Z: Undo
                    if self.points:
                        self.points.pop()
                        self.current_img_copy = self.current_image.copy()
                        
                        h, w = self.current_image.shape[:2]
                        for x, y in self.points:
                            cv2.circle(self.current_img_copy, (x, y), 5, (0, 0, 255), -1)
                            box_w_px = int(w * self.DART_WIDTH_PERCENT)
                            box_h_px = int(h * self.DART_HEIGHT_PERCENT)
                            x_min = x - box_w_px // 2
                            y_min = y - box_h_px // 2
                            x_max = x + box_w_px // 2
                            y_max = y + box_h_px // 2
                            cv2.rectangle(self.current_img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        cv2.imshow("Smart Dart Labeler", self.current_img_copy)
                
                elif key == ord('c'):  # C: Clear
                    self.points = []
                    self.current_img_copy = self.current_image.copy()
                    cv2.imshow("Smart Dart Labeler", self.current_img_copy)
                
                elif key == ord('q'):  # Q: Quit
                    cv2.destroyAllWindows()
                    print("\n⚠️  Labeling stopped by user")
                    return
            
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"\n✅ Smart Labeling complete!")

if __name__ == "__main__":
    labeler = SmartDartLabeler()
    labeler.label_images()
