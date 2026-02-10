# 🎯 Dartboard Pattern Stereo Calibration

## Overview

BullSight now supports **automatic stereo calibration using the dartboard itself** as a calibration pattern, eliminating the need for separate checkerboard patterns.

## Advantages

### 1. **Always Available**
- Dartboard is permanently mounted
- No need to print/prepare checkerboard
- Can re-calibrate anytime

### 2. **Standardized Geometry**
Uses official PDC/WINMAU dartboard specifications:
- **Bull's Eye:** 12.7mm diameter
- **Bull Outer:** 31.8mm diameter
- **Triple Ring:** 99-107mm radius
- **Double Ring:** 162-170mm radius
- **20 Segments:** 18° intervals

### 3. **Multiple Reference Points**
- **81 calibration points total:**
  - 1 center point (Bull's Eye)
  - 40 triple ring intersections (20 segments × 2 edges)
  - 40 double ring intersections (20 segments × 2 edges)

## Detection Algorithm

### Step 1: Circle Detection
```python
# Detect concentric rings using Hough Circle Transform
circles = cv2.HoughCircles(image, method=cv2.HOUGH_GRADIENT, ...)
# Identifies: Bull, Triple (inner/outer), Double (inner/outer)
```

### Step 2: Segment Boundaries
```python
# Detect radial lines (wires between segments)
lines = cv2.HoughLines(edges, ...)
# Identifies 20 segment boundaries at ~18° intervals
```

### Step 3: Intersection Points
```python
# Compute ring-segment intersections
for angle in segment_angles:
    for radius in ring_radii:
        point = (center_x + r*cos(angle), center_y + r*sin(angle))
```

## Usage

### UI Selection
1. Open **Stereo Calibration** from main menu
2. Select **"🎯 Dartboard (Recommended - Always Available)"**
3. Ensure dartboard is visible in both camera views
4. Press **SPACE** when pattern is detected
5. Optionally capture 5-10 images from different angles
6. Click **"🎯 Calibrate Cameras"**

### Expected Accuracy
- **RMS Error:** 0.5-1.5 pixels (vs. 0.3-0.8 for checkerboard)
- **Baseline Accuracy:** ±1-2mm (excellent for dart detection)
- **3D Position Error:** ±3-5mm at board distance

## Best Practices

### Lighting
- ✅ Bright, uniform lighting
- ❌ Avoid shadows, reflections
- ✅ Use overhead lighting or ring lights

### Board Condition
- ✅ Remove all darts before calibration
- ✅ Clean board surface
- ❌ Don't calibrate with worn/damaged board

### Multiple Captures
- **Minimum:** 1 capture (if board is perfectly aligned)
- **Recommended:** 5-10 captures from slight angle variations
- **Maximum:** 20 captures (diminishing returns)

### Verification
After calibration, check:
- **RMS Error:** Should be < 2.0 pixels
- **Baseline Distance:** Should match physical camera separation
- **Focal Length:** Should be reasonable for your lenses

## Troubleshooting

### ❌ "No pattern detected"
**Causes:**
- Poor lighting (too dark or too bright)
- Board not fully visible
- Camera angle too oblique
- Darts blocking view

**Solutions:**
- Improve lighting conditions
- Adjust camera positions
- Remove darts from board
- Reduce camera angle (aim perpendicular)

### ❌ "Detection inconsistent"
**Causes:**
- Motion blur
- Focus issues
- Damaged board markings

**Solutions:**
- Use stable camera mount
- Ensure cameras are in focus
- Clean board surface

### ❌ "High RMS error (>2.0)"
**Causes:**
- Too few captures
- Poor feature detection
- Camera synchronization issues

**Solutions:**
- Capture more images (10-20)
- Improve lighting
- Check camera frame sync

## Technical Details

### Coordinate System
- **Origin:** Bull's Eye center
- **X-Axis:** Horizontal (positive = right)
- **Y-Axis:** Vertical (positive = down)
- **Z-Axis:** Depth (positive = away from camera)

### 3D Reference Points
Generated in millimeters from center:
```python
# Center
[0, 0, 0]

# Triple inner at segment 20 (top, -90°)
[0, -99.0, 0]

# Triple outer at segment 20
[0, -107.0, 0]

# And so on for all 20 segments...
```

### Detection Parameters
Tunable in `DartboardPatternDetector`:
- `hough_circle_param1`: Circle detection edge threshold (default: 100)
- `hough_circle_param2`: Circle detection accumulator (default: 30)
- `hough_line_threshold`: Line detection threshold (default: 80)
- `min_circle_distance`: Minimum distance between circles (default: 20px)

## Comparison: Dartboard vs. Checkerboard

| Feature | Dartboard | Checkerboard |
|---------|-----------|--------------|
| **Setup Time** | Instant | 5-10 min (print) |
| **Convenience** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Accuracy** | ⭐⭐⭐⭐ (1-2mm) | ⭐⭐⭐⭐⭐ (0.5-1mm) |
| **Robustness** | ⭐⭐⭐ (lighting critical) | ⭐⭐⭐⭐⭐ |
| **Re-calibration** | Easy (always available) | Requires pattern |
| **Points Detected** | 81 | 54 (9×6) |
| **Use Case** | Production | High-precision |

## Recommendation

**Use Dartboard Calibration when:**
- ✅ Quick setup needed
- ✅ Re-calibration during operation
- ✅ Good lighting available
- ✅ 3-5mm accuracy sufficient

**Use Checkerboard when:**
- ✅ Initial high-precision calibration
- ✅ Poor dartboard visibility
- ✅ Sub-millimeter accuracy required
- ✅ Research/validation purposes

## Future Enhancements

Potential improvements:
1. **Number Detection:** OCR on segment numbers for auto-rotation correction
2. **Adaptive Thresholds:** Auto-tune detection parameters
3. **Outlier Rejection:** Robust fitting with RANSAC
4. **Dynamic Re-calibration:** Auto-recalibrate during gameplay
5. **Hybrid Approach:** Combine both methods for optimal accuracy

## References
- **PDC Dartboard Specifications:** Official tournament standard
- **OpenCV Stereo Calibration:** `cv2.stereoCalibrate()`
- **Hough Transform:** Circle and line detection
- **Sub-pixel Refinement:** Corner detection optimization
