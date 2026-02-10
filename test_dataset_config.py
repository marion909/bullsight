#!/usr/bin/env python3
"""Test the data.yaml configuration"""

import yaml
from pathlib import Path

yaml_file = Path('training_data/finetuning_data/data.yaml')

with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

print('✓ data.yaml configuration:')
print(f"  Base path: {data['path']}")
print(f"  Train images: {data['train']}")
print(f"  Classes: {data['names']}")
print()

# Check if directories exist
train_path = Path(data['path']) / data['train']
print(f"✓ Train directory exists: {train_path.exists()}")
print(f"  Full path: {train_path}")
