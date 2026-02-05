#!/bin/bash
# BullSight Startup Script for Raspberry Pi
# Author: Mario Neuhauser

set -e

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install.sh first."
    exit 1
fi

source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Start the application
echo "🎯 Starting BullSight..."
python src/main.py
