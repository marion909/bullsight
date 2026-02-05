# BullSight Quick Setup Guide

## 🚀 Quick Installation

### Raspberry Pi Setup (Production)

```bash
# 1. Clone repository
git clone https://github.com/marion909/bullsight.git
cd BullSight

# 2. Run automated installer
chmod +x install.sh
./install.sh

# 3. Start application
./run.sh
```

### Windows Setup (Development)

```bash
# 1. Clone repository
git clone https://github.com/marion909/bullsight.git
cd BullSight

# 2. Run automated installer
install.bat

# 3. Start application
run.bat
```

## 📋 System Requirements

### Minimum Requirements
- **CPU**: Raspberry Pi 4 (2GB RAM) or higher
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.11 or higher
- **Camera**: Raspberry Pi Camera Module v2 or v3
- **Display**: 7-inch touchscreen (800x480) or HDMI display

### Recommended Setup
- **CPU**: Raspberry Pi 5 (4GB RAM)
- **Camera**: Camera Module v3 (autofocus)
- **Display**: Official 7-inch touchscreen
- **Storage**: 16GB+ microSD card (Class 10)

### Windows Development
- **OS**: Windows 10/11
- **Python**: 3.11-3.13
- **Note**: Camera features require Raspberry Pi hardware

## 🎯 First-Time Setup

### 1. Hardware Assembly
1. Connect Camera Module to Raspberry Pi
2. Connect touchscreen display
3. Power on Raspberry Pi

### 2. Enable Camera
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot when prompted
```

### 3. Test Camera
```bash
libcamera-hello --timeout 5000
```

### 4. Run BullSight
```bash
cd BullSight
./run.sh
```

### 5. Initial Calibration
1. Navigate to Settings → Calibration
2. Click "Set Center" and tap bull's eye
3. Adjust ring radii with sliders
4. Click "Save Calibration"

## 🎮 Quick Usage Guide

### Starting a Game
1. **Main Menu** → "New Game"
2. **Add Players** → Enter 1-8 player names
3. **Select Mode** → Choose 301, 501, or Cricket
4. **Play!** → Throw darts and watch auto-scoring

### Manual Scoring (No Camera)
- Use on-screen buttons to input dart throws
- Useful for testing or when camera is unavailable

### Navigation
- **Back Button**: Return to previous screen
- **Settings**: Configure sound, view calibration
- **Menu**: Access game settings and options

## 🔧 Troubleshooting

### Installation Issues

**Problem**: `pip install` fails for picamera2 on Windows
- **Solution**: This is expected. Picamera2 only works on Raspberry Pi. All other features work fine.

**Problem**: Permission denied for install.sh
- **Solution**: Run `chmod +x install.sh run.sh`

**Problem**: Python version too old
- **Solution**: Update Python: `sudo apt update && sudo apt install python3.11`

### Runtime Issues

**Problem**: "No module named 'src'"
- **Solution**: Set PYTHONPATH: `export PYTHONPATH="$(pwd)"`

**Problem**: Camera not detected
- **Solution**: 
  1. Enable camera: `sudo raspi-config`
  2. Check connection: `libcamera-hello`
  3. Verify cable is properly seated

**Problem**: Display not working
- **Solution**: Check HDMI/DSI connection, verify display power

**Problem**: Touch not responding
- **Solution**: Ensure touchscreen drivers installed: `sudo apt install raspberrypi-ui-mods`

### Performance Issues

**Problem**: Slow UI response
- **Solution**: 
  - Use Raspberry Pi 4/5 (not Pi 3)
  - Close other applications
  - Reduce camera resolution in settings

**Problem**: Dart detection too slow
- **Solution**:
  - Ensure good lighting on dartboard
  - Adjust detection sensitivity in calibration
  - Use Camera Module v3 for better performance

## 📝 Configuration Files

### Calibration Data
- Location: `config/calibration/calibration.json`
- Format: JSON with center point and ring radii
- Backup: Recommended after successful calibration

### Logs
- Location: `logs/bullsight.log`
- Level: INFO (configurable in code)
- Rotation: Automatic when file reaches 10MB

## 🔄 Updating

```bash
cd BullSight
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## 🆘 Getting Help

### Check Logs
```bash
tail -f logs/bullsight.log
```

### Run Diagnostics
```bash
# Test camera
libcamera-hello

# Test Python environment
python --version
pip list | grep -E "(opencv|PySide6|numpy)"

# Run basic tests
pytest tests/unit/test_ui_logic.py -v
```

### Common Commands
```bash
# Restart application
./run.sh

# Run with debug output
PYTHONPATH="$(pwd)" python src/main.py --debug

# Check system resources
htop
```

## 📚 Additional Resources

- **GitHub Repository**: https://github.com/marion909/bullsight
- **Phase Documentation**: See `phase*.md` files
- **Test Suite**: Run `pytest tests/ -v` for verification

---

**Need more help?** Check the main README.md or phase documentation files.
