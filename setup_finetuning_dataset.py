#!/usr/bin/env python3
"""
Generate data.yaml for finetuning dataset with correct absolute paths.
This ensures YOLO finds the training images regardless of installation location.
"""

import yaml
from pathlib import Path

def create_finetuning_dataset_yaml():
    """Create data.yaml with correct absolute paths for finetuning."""
    
    # Get absolute base path
    base_dir = Path(__file__).parent / "training_data" / "finetuning_data"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure subdirectories exist
    (base_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml with absolute paths
    data_config = {
        'path': str(base_dir.absolute()),
        'train': 'images/train',
        'val': 'images/train',
        'test': 'images/train',
        'nc': 1,
        'names': {0: 'dart'}
    }
    
    yaml_file = base_dir / 'data.yaml'
    
    with open(yaml_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Dataset config created: {yaml_file}")
    print(f"  Base: {data_config['path']}")
    
    return yaml_file

if __name__ == '__main__':
    create_finetuning_dataset_yaml()
