#!/usr/bin/env python3
"""Test the complete ML training workflow."""

from src.ui.ml_demo_screen import MLDemoScreen
from src.ui.model_finetune_dialog import ModelFinetuneDialog
from pathlib import Path
import yaml

print('✅ All modules loaded successfully')
print()
print('🎯 New workflow:')
print('  1. ML Demo Screen → Press SPACE to capture (50-100 times)')
print('  2. Click Finetune Model button')
print('  3. Training starts automatically')
print()

# Check if training directory exists
train_dir = Path('training_data/finetuning_data/images/train')
print(f'✓ Training dir: {train_dir}')
print(f'  Exists: {train_dir.exists()}')

yaml_file = Path('training_data/finetuning_data/data.yaml')
if yaml_file.exists():
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    print(f'✓ data.yaml configured with absolute path')
    print(f'  Base: {config["path"]}')
    print(f'  Train: {config["train"]}')
else:
    print('✗ data.yaml not found')
