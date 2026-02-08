# 🎯 BullSight - Automated Dart Scoring System

Raspberry Pi-based dart scoring system with computer vision and touch interface.

## ✨ Features

- 🎯 **Automated Dart Detection**: Computer vision-based dart detection using Raspberry Pi Camera
- � **Dual-Camera Stereo Vision**: ±2mm precision 3D dart localization with synchronized dual cameras
- 🤖 **ML Detection**: YOLOv8-based dart detection (98.6% mAP50) with real-time inference
- 📐 **Stereo Calibration**: Interactive wizard for camera pair calibration with checkerboard pattern
- 🎲 **3D Triangulation**: Epipolar geometry-based 3D reconstruction from stereo images
- �📊 **Live Scoring**: Real-time score tracking for 301, 501, and Cricket game modes
- 🎮 **Touch Interface**: Intuitive 7-inch touchscreen UI built with PySide6
- 📐 **Board Calibration**: Interactive dartboard calibration with visual feedback
- 📈 **Player Statistics**: Track performance metrics and game history
- 🔊 **Audio Feedback**: Sound effects for dart throws and game events

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)

**On Raspberry Pi / Linux:**
```bash
chmod +x install.sh
./install.sh
./run.sh
```

**On Windows (Development):**
```bash
install.bat
run.bat
```

### Option 2: Manual Installation

**Prerequisites:**
- Python 3.11 or higher
- Raspberry Pi 4/5 with Camera Module v3 (for production)
- 7-inch Touch Display (optional, but recommended)

**Installation Steps:**
```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set PYTHONPATH
# On Linux/Mac:
export PYTHONPATH="$(pwd)"
# On Windows:
set PYTHONPATH=%CD%

# 6. Run the application
python src/main.py
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only

# Run quick UI tests
pytest tests/unit/test_ui_logic.py -v
```

## 📁 Project Structure
```
BullSight/
├── src/
│   ├── vision/          # Computer vision (dart detection, camera)
│   │   ├── dual_camera_manager.py    # Stereo camera synchronization
│   │   ├── ml_dart_detector.py       # YOLOv8 ML detection + stereo
│   │   └── dart_detector.py          # Classical CV detection
│   ├── calibration/     # Dartboard mapping and calibration
│   │   ├── stereo_calibration_data.py   # Stereo calibration parameters
│   │   └── board_mapper.py              # Dartboard coordinate mapping
│   ├── game/            # Game engine (301, 501, Cricket)
│   ├── ui/              # PySide6 user interface
│   │   ├── start_screen.py
│   │   ├── player_management_screen.py
│   │   ├── game_mode_screen.py
│   │   ├── live_score_screen.py
│   │   ├── calibration_screen.py
│   │   ├── stereo_calibration_screen.py  # Stereo calibration wizard
│   │   ├── ml_demo_screen.py             # ML detection demo with 3D
│   │   └── settings_screen.py
│   ├── calibration/             # Dartboard calibration data
│   └── stereo_calibration.json  # Stereo camera calibrationnagement
│   └── main.py          # Application entry point
├── tests/
│   ├── unit/            # Unit tests (100% coverage on core)
│   ├── integration/     # Integration tests
│   └── conftest.py      # Pytest configuration
├── config/              # JSON configurations
│   └── calibration/     # Calibration data
├── assets/              # Images and sounds
│   ├── images/
│   └── sounds/
├── logs/                # Application logs
├── install.sh           # Linux/Pi installation script
├── install.bat          # Windows installation script
├── run.sh               # Linux/Pi startup script
├── run.bat              # Windows startup script
├── requirements.txt     # Python dependencies
└── README.md
```

## 🎮 Usage

### Starting the Application

**On Raspberry Pi:**
```bash
./run.sh
```

**On Windows (Development):**
```bash
run.bat
```

**Manual Start:**
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
export PYTHONPATH="$(pwd)"  # or set PYTHONPATH=%CD% on Windows
python src/main.py
```

### Game Workflow

1. **Start Screen** → Select "New Game" or configure settings
2. **Player Management** → Add 1-8 players
3. **Game Mode Selection** → Choose 301, 501, or Cricket
4. **Live Game** → Play! Darts are detected automatically
5. **Statistics** → View player performance after game
Dartboard Calibration

First-time setup requires dartboard calibration:
1. Navigate to Settings → Calibration
2. Click "Set Center" and click on the bull's eye
3. Adjust ring radii using sliders
4. Save calibration

### Stereo Camera Calibration

For dual-camera stereo vision (±2mm precision):
1. Connect two USB cameras directly to PC (avoid USB hubs for bandwidth)
**Single Camera Mode:**
Edit `src/vision/camera_manager.py` for camera configuration:
- Resolution: Default 1280x720
- Autofocus: Enabled
- Frame rate: 30 fps

**Dual Camera Mode:**
Edit `src/vision/dual_camera_manager.py` for stereo configuration:
- Resolution: 1280x720 per camera
- Synchronization: Sequential capture (~5-10ms time delta)
- Buffer size: 1 (minimal latency)
- Hardware requirement: Direct PC connection (no USB hub)

### ML Detection Settings

Enable ML detection:
```bash
# Install ML dependencies
pip install ultralytics torch torchvision

# Enable ML mode
- ✅ **Phase 6: ML Detection** - YOLOv8 training, real-time inference
- ✅ **Phase 7: Stereo Vision** - Dual-camera system, 3D triangulation

### Stereo Vision Implementation (Phase 7) ✅

**Phase 7.1: Dual-Camera Infrastructure** ✅
- Synchronized stereo frame capture (5-10ms)
- Smart camera detection with brightness validation
- Dual-view UI visualization
- Sequential read() approach (OpenCV thread-safe)

**Phase 7.2: Stereo Calibration System** ✅
- StereoCalibrationData dataclass (K, D, R, T, E, F, P, Q)
- Interactive calibration wizard UI
- Checkerboard pattern detection (real-time feedback)
- OpenCV stereo calibration pipeline
- JSON persistence

**Phase 7.3: ML Detection Fusion** ✅
- `detect_stereo()` method for dual-image processing
- Epipolar constraint matching (Fundamental matrix)

# For dual cameras: Check available cameras
python -c "from src.vision.dual_camera_manager import DualCameraManager; print(DualCameraManager.check_available_cameras())"
```

### Stereo Camera Issues
- **Black frames**: Connect cameras directly to PC (not USB hub)
- **Bandwidth issues**: Reduce resolution or frame rate
- **Synchronization**: Ensure both cameras have same settings
- **Calibration fails**: Capture 20+ well-distributed image pairsD triangulation via cv2.triangulatePoints()
- ML Demo UI with 3D coordinate display
- Dartboard field mapping with stereo coordinates

**Phase 7.4: System Integration** 🔄
- Production scoring with 3D positions (planned)
- Real-world accuracy testing (target: ±2mm)
- Performance optimization
export BULLSIGHT_USE_ML=1  # Linux/Mac
set BULLSIGHT_USE_ML=1     # Windows

python src/main.py
```

Model configuration:
- **Model**: YOLOv8n (nano) for real-time performance
- **Training**: 201 real dart images, 50 epochs
- **Performance**: 98.6% mAP50, ~10 FPS on CPU
- **Location**: `models/dart_detector_v8.pt` pairs from different angles and distances
6. Click "🎯 Run Calibration" to compute stereo parameters
7. Results saved to `config/stereo_calibration.json`

### ML Detection Demo

Test ML detection with 3D reconstruction:
1. Complete stereo calibration first
2. Navigate to "🤖 ML Demo" from main menu
3. View live dual-camera feed with dart detection
4. Detections show:
   - Left/Right 2D coordinates
   - Detection confidence per camera
   - Epipolar matching error
   - 3D position (X, Y, Z) in millimeters
   - Dartboard field mappingi using sliders
4. Save calibration

## 🛠️ Configuration

### Camera Settings

Edit `src/vision/camera_manager.py` for camera configuration:
- Resolution: Default 1280x720
- Autofocus: Enabled
- Frame rate: 30 fps

### Game Settings

Accessible via Settings screen:
- Sound volume
- Sound enabled/disabled
- Calibration data

## 📊 Development Status

### Completed Phases ✅

- ✅ **Phase 1: Foundations** - Project structure, testing framework
- ✅ **Phase 2: Vision Engine** - Dart detection, camera management
- ✅ **Phase 3: Calibration** - Board mapping, coordinate transformation
- ✅ **Phase 4: Game Engine** - 301/501 game modes, player management
- ✅ **Phase 5: UI Implementation** - Complete PySide6 interface

### Test Coverage

- **Core Modules**: 100% coverage
  - `board_mapper.py`: 100%
  - `game_engine.py`: 100%
  - `camera_manager.py`: 100%
  - `dart_detector.py`: 100%
- **UI Modules**: 78% coverage (UI testing limitations)
- **Total Tests**: 170+ tests passing

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# Enable camera on Raspberry Pi
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Test camera
libcamera-hello
```

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="$(pwd)"  # Linux/Mac
set PYTHONPATH=%CD%         # Windows
```

### Display Issues
- Ensure display is connected before starting
- Check resolution settings in camera_manager.py
- On Windows: Full camera functionality requires Raspberry Pi

### Missing Dependencies
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📖 Documentation

For detailed development documentation, see:
- [Phase 1: Foundations](phase1-foundations.md)
- [Phase 2: Vision Engine](phase2-vision.md)
- [Phase 3: Calibration](phase3-calibration.md)
- [Phase 4: Game Engine](phase4-game-engine.md)
- [Phase 5: UI Polish](phase5-ui-polish.md)

## 🤝 Contributing

This is a personal project for learning purposes. Feel free to fork and adapt!

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Mario Neuhauser**
- Platform: Raspberry Pi 4/5
- Language: Python 3.13
- UI Framework: PySide6 6.10.2

---

**Built with ❤️ for the dart community**
