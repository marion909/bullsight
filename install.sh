#!/bin/bash
# BullSight Installation Script for Raspberry Pi
# Author: Mario Neuhauser

set -e  # Exit on error

echo "🎯 BullSight Installation Script"
echo "================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "⚠️  Warning: Not running on Raspberry Pi. Some features may not work."
    echo "   Continuing anyway for development setup..."
    echo ""
fi

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p config/calibration
mkdir -p assets/sounds
mkdir -p assets/images
echo "✅ Directories created"
echo ""

# Set PYTHONPATH
echo "🔧 Setting up environment..."
export PYTHONPATH="$PWD"
echo "✅ PYTHONPATH set to $PWD"
echo ""

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
pytest tests/unit/test_ui_logic.py -v
if [ $? -eq 0 ]; then
    echo "✅ Basic tests passed!"
else
    echo "⚠️  Some tests failed. Please check the output above."
fi
echo ""

# Display next steps
echo "================================================"
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Set PYTHONPATH:"
echo "     export PYTHONPATH=\"$PWD\""
echo ""
echo "  3. Run the application:"
echo "     python src/main.py"
echo ""
echo "  4. Run all tests:"
echo "     pytest tests/ -v"
echo ""
echo "For Raspberry Pi with camera:"
echo "  - Ensure camera is connected and enabled"
echo "  - Run 'sudo raspi-config' to enable camera if needed"
echo "================================================"
