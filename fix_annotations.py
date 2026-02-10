#!/usr/bin/env python3
"""
Fix dart annotations: expand tiny point annotations to proper bounding boxes.
The user only marked dart tips, but YOLO needs full dart bounding boxes.
"""

from pathlib import Path

def expand_annotations():
    """Expand all point annotations to reasonable dart bounding box sizes."""
    
    labels_dir = Path("training_data/finetuning_data/labels/train")
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"🔧 Fixing {len(label_files)} annotation files...")
    print()
    
    # Expansion factor: make each point into a reasonable dart box
    # Average dart width: ~4% of image width
    # Average dart height: ~8% of image height
    width_expansion = 0.04  # 4% of image width on each side
    height_expansion = 0.08  # 8% of image height on each side
    
    fixed_count = 0
    
    for label_path in sorted(label_files):
        try:
            content = label_path.read_text().strip()
            if not content:
                continue
            
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                cls_id = parts[0]
                center_x = float(parts[1])
                center_y = float(parts[2])
                
                # Expand to reasonable dart size
                new_width = width_expansion
                new_height = height_expansion
                
                fixed_lines.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {new_width:.6f} {new_height:.6f}")
            
            if fixed_lines:
                label_path.write_text("\n".join(fixed_lines))
                fixed_count += 1
                
                # Show before/after for first few files
                if fixed_count <= 3:
                    print(f"✓ {label_path.name}")
                    print(f"  └─ Old: {lines[0]}")
                    print(f"  └─ New: {fixed_lines[0]}")
                    print()
        
        except Exception as e:
            print(f"✗ Error processing {label_path.name}: {e}")
    
    print(f"✅ Fixed {fixed_count} annotation files")
    print(f"   Each dart now has:")
    print(f"   - Width: {width_expansion*100:.1f}% of image")
    print(f"   - Height: {height_expansion*100:.1f}% of image")
    print()
    print("🚀 Ready for retraining!")

if __name__ == "__main__":
    expand_annotations()
