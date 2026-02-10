"""
Debug tool for dartboard pattern detection.

Tests detection with different parameter sets and visualizes results.

Author: Mario Neuhauser
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.dartboard_pattern_detector import DartboardPatternDetector


def test_detection_parameters(image: np.ndarray):
    """Test dartboard detection with various parameter sets."""
    
    print("\n" + "="*80)
    print("DARTBOARD DETECTION DEBUG")
    print("="*80)
    
    detector = DartboardPatternDetector()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"\nImage size: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Mean brightness: {gray.mean():.1f}")
    print(f"Std deviation: {gray.std():.1f}")
    
    # Test 1: CLAHE preprocessing
    print("\n--- Test 1: Contrast Enhancement ---")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    print(f"Enhanced mean: {enhanced.mean():.1f}, std: {enhanced.std():.1f}")
    cv2.imshow("1. Original", gray)
    cv2.imshow("2. CLAHE Enhanced", enhanced)
    
    # Test 2: Edge detection
    print("\n--- Test 2: Edge Detection ---")
    edges = cv2.Canny(enhanced, 30, 100)
    edge_ratio = (edges > 0).sum() / edges.size * 100
    print(f"Edge pixels: {edge_ratio:.2f}%")
    cv2.imshow("3. Edges", edges)
    
    # Test 3: Circle detection with multiple parameter sets
    print("\n--- Test 3: Hough Circle Detection ---")
    
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    param_sets = [
        ("Very Relaxed", 40, 15, 10),
        ("Relaxed", 50, 20, 15),
        ("Moderate", 60, 25, 15),
        ("Strict", 70, 30, 20),
    ]
    
    best_result = None
    best_count = 0
    
    for name, param1, param2, minRadius in param_sets:
        circles = cv2.HoughCircles(
            filtered,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=15,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=min(gray.shape) // 2
        )
        
        count = len(circles[0]) if circles is not None else 0
        print(f"  {name:15} (p1={param1}, p2={param2}, r={minRadius}): {count} circles")
        
        if count > best_count:
            best_count = count
            best_result = (circles, name)
    
    # Visualize best result
    if best_result is not None:
        circles, name = best_result
        print(f"\n✅ Best result: {name} with {best_count} circles")
        
        # Apply same filtering as detector (Board-First approach)
        h, w = gray.shape
        
        # Find board-sized circles
        min_board_radius = min(h, w) * 0.10
        max_board_radius = min(h, w) * 0.40
        
        board_candidates = []
        for circle in circles[0]:
            cx, cy, r = circle
            if min_board_radius < r < max_board_radius:
                dist_from_img_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                if dist_from_img_center < min(w, h) * 0.3:
                    board_candidates.append(circle)
        
        if not board_candidates:
            print("❌ No board-sized circles found")
            return
        
        # Use largest as board
        board_circle = max(board_candidates, key=lambda c: c[2])
        board_cx, board_cy, board_radius = board_circle
        
        print(f"Board detected: center=({board_cx:.1f}, {board_cy:.1f}), radius={board_radius:.1f}")
        
        # Filter to concentric circles only
        filtered_circles = []
        tolerance = board_radius * 0.1
        
        for circle in circles[0]:
            cx, cy, r = circle
            dist = np.sqrt((cx - board_cx)**2 + (cy - board_cy)**2)
            if dist < tolerance and 15 < r < board_radius * 0.95:
                filtered_circles.append(circle)
        
        print(f"After concentricity filtering: {len(filtered_circles)} circles")
        
        vis = image.copy()
        for circle in filtered_circles[:15]:  # Show max 15 best circles
            cx, cy, r = circle
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), 3)
        
        cv2.putText(vis, f"Filtered: {len(filtered_circles)} circles", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("4. Detected Circles (Filtered)", vis)
        
        # Analyze circle distribution
        print("\nFiltered Circle Analysis (showing first 10):")
        for i, (cx, cy, r) in enumerate(filtered_circles[:10]):
            print(f"  Circle {i+1}: center=({cx:.1f}, {cy:.1f}), radius={r:.1f}")
    else:
        print("\n❌ No circles detected with any parameter set")
    
    # Test 4: Full detection pipeline
    print("\n--- Test 4: Full Detection Pipeline ---")
    result = detector.detect_dartboard_features(image, visualize=True)
    
    if result is not None:
        points, vis = result
        print(f"✅ Detection successful: {len(points)} calibration points")
        cv2.imshow("5. Full Detection Result", vis)
    else:
        print("❌ Full detection failed")
    
    print("\n" + "="*80)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("🎯 Dartboard Detection Debug Tool")
    print("\nOptions:")
    print("  1. Test with live camera (press SPACE to analyze)")
    print("  2. Test with image file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Live camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n📷 Camera opened. Press SPACE to analyze frame, ESC to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                cv2.destroyWindow("Camera Feed")
                test_detection_parameters(frame)
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif choice == "2":
        # Image file
        filepath = input("Enter image path: ").strip()
        image = cv2.imread(filepath)
        
        if image is None:
            print(f"❌ Could not load image: {filepath}")
            return
        
        test_detection_parameters(image)
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
