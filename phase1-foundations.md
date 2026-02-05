# 🧱 Phase 1 – Foundations & Infrastructure

**Dependencies:** None  
**Next Phase:** [Phase 2 – Vision Engine](phase2-vision.md)

---

## 🎯 Phase Goals

- Setup complete Raspberry Pi development environment
- Establish project structure and version control
- Verify all hardware components functionality
- Create central dependency management
- Achieve 100% test coverage for hardware integration

---

## 📋 Prerequisites

### Hardware Checklist
- [ ] Raspberry Pi 4 or 5 (4 GB RAM minimum)
- [ ] Raspberry Pi Camera Module v3 (Autofocus)
- [ ] 7-inch Touch Display
- [ ] LED Light Ring
- [ ] Standard Steel Dartboard
- [ ] MicroSD Card (32 GB+)
- [ ] Power Supply (official recommended)

### Optional Hardware
- [ ] Speakers for sound effects
- [ ] External power button
- [ ] 3D-printed housing/mount

---

## 🔧 Implementation Tasks

### 1.1 Operating System Setup

**Task:** Install and configure Raspberry Pi OS

```bash
# Recommended: Raspberry Pi OS Bullseye or Bookworm (64-bit)
# Use Raspberry Pi Imager for installation

# After first boot, update system
sudo apt update && sudo apt upgrade -y

# Enable required interfaces
sudo raspi-config
# → Interface Options → Camera → Enable
# → Interface Options → I2C → Enable (for sensors if needed)
# → Display Options → Configure resolution for 7" display
```

**Expected Outcome:** Fully updated Raspberry Pi OS with camera and display interfaces enabled

**Test Requirements:**
- Verify OS version: `cat /etc/os-release`
- Confirm camera interface: `vcgencmd get_camera`
- Validate display resolution: `xrandr`

---

### 1.2 Python Environment Setup

**Task:** Install Python 3.11+ and essential tools

```bash
# Install Python 3.11+ (if not default)
sudo apt install python3 python3-pip python3-venv -y

# Verify version (must be 3.11+)
python3 --version

# Install development tools
sudo apt install python3-dev build-essential -y
sudo apt install git -y
```

**Expected Outcome:** Python 3.11+ installed with pip and venv

**Test Requirements:**
- Python version check: `python3 --version` >= 3.11
- Pip functionality: `pip3 --version`
- Venv creation test: `python3 -m venv test_env && rm -rf test_env`

---

### 1.3 Project Structure Setup

**Task:** Create standardized directory structure

```bash
# Navigate to project root
cd /home/pi/BullSight  # or your preferred location

# Create directory structure
mkdir -p src/{vision,calibration,game,ui,config}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs
mkdir -p config
mkdir -p assets/{images,sounds}
mkdir -p logs
```

**Expected Structure:**
```
BullSight/
├── src/
│   ├── vision/          # Computer vision components
│   ├── calibration/     # Dartboard mapping
│   ├── game/            # Game engine logic
│   ├── ui/              # PyQt6 interface
│   └── config/          # Configuration management
├── tests/
│   ├── unit/            # Unit tests (isolated)
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── docs/                # API documentation
├── config/              # JSON/YAML configurations
├── assets/
│   ├── images/          # Reference images, calibration data
│   └── sounds/          # Audio effects
├── logs/                # Application logs
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest configuration
├── .gitignore
└── README.md
```

**Test Requirements:**
- All directories exist: `test -d src/vision && echo "OK"`
- Write permissions: `touch logs/test.log && rm logs/test.log`

---

### 1.4 Central Dependency Management

**Task:** Create requirements.txt with all project dependencies

**File:** `requirements.txt`

```txt
# Core Dependencies
opencv-python==4.9.0.80
numpy==1.26.4
picamera2==0.3.17

# UI Framework
PyQt6==6.6.1
PyQt6-Qt6==6.6.1

# Testing
pytest==8.0.0
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-qt==4.3.1

# Audio (optional, Phase 5)
pygame==2.5.2

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0

# Development
black==24.1.1
flake8==7.0.0
mypy==1.8.0
```

**Installation:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "from PyQt6 import QtCore; print(f'PyQt6: {QtCore.PYQT_VERSION_STR}')"
```

**Test Requirements:**
- All imports succeed without errors
- Version verification for critical packages
- 100% import coverage test

---

### 1.5 Git Repository Initialization

**Task:** Setup version control

```bash
# Initialize Git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log

# Config (if contains secrets)
config/local.yaml

# Assets (large files)
assets/images/*.jpg
assets/images/*.png

# OS
.DS_Store
Thumbs.db
EOF

# Create README
cat > README.md << 'EOF'
# 🎯 BullSight - Automated Dart Scoring System

Raspberry Pi-based dart scoring system with computer vision and touch interface.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest`
3. See [Phase 1](phase1-foundations.md) for setup details

## Phases
- [Phase 1: Foundations](phase1-foundations.md)
- [Phase 2: Vision Engine](phase2-vision.md)
- [Phase 3: Calibration](phase3-calibration.md)
- [Phase 4: Game Engine](phase4-game-engine.md)
- [Phase 5: UI & Polish](phase5-ui-polish.md)

**Author:** Mario Neuhauser  
**Platform:** Raspberry Pi 4/5  
**Language:** Python 3.11+
EOF

# Initial commit
git add .
git commit -m "Initial project setup - Phase 1 foundations"
```

**Test Requirements:**
- Git status shows clean working tree
- .gitignore excludes venv and cache files
- README.md renders correctly

---

### 1.6 Camera Hardware Test

**Task:** Verify Raspberry Pi Camera Module v3 functionality

**File:** `tests/integration/test_camera_hardware.py`

```python
"""
Camera hardware integration tests.
Tests Raspberry Pi Camera Module v3 functionality.

Coverage Target: 100%
"""

import pytest
from picamera2 import Picamera2
import numpy as np
import time


class TestCameraHardware:
    """Test suite for camera hardware integration."""

    @pytest.fixture
    def camera(self):
        """
        Initialize camera for testing.
        
        Yields:
            Picamera2: Configured camera instance
        """
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (1920, 1080)},
            buffer_count=2
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)  # Allow camera to stabilize
        yield cam
        cam.stop()
        cam.close()

    def test_camera_initialization(self, camera):
        """
        Test camera can be initialized successfully.
        
        Verifies:
        - Camera object creation
        - Configuration acceptance
        - Start/stop lifecycle
        """
        assert camera is not None
        assert camera.started

    def test_camera_capture_single_frame(self, camera):
        """
        Test single frame capture.
        
        Verifies:
        - Image capture succeeds
        - Image has correct dimensions
        - Image contains valid pixel data
        """
        frame = camera.capture_array()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[0] == 1080  # Height
        assert frame.shape[1] == 1920  # Width
        assert frame.shape[2] == 3     # RGB channels
        
        # Verify not all black (camera working)
        assert frame.mean() > 0

    def test_camera_autofocus(self, camera):
        """
        Test autofocus functionality (Camera Module v3 feature).
        
        Verifies:
        - Autofocus can be triggered
        - Focus completes without error
        """
        # Trigger autofocus
        camera.set_controls({"AfMode": 2})  # AfModeAuto
        time.sleep(1)
        
        # Capture should succeed after autofocus
        frame = camera.capture_array()
        assert frame is not None

    def test_camera_multiple_captures(self, camera):
        """
        Test multiple sequential captures (dart detection simulation).
        
        Verifies:
        - Camera can capture multiple frames
        - No memory leaks
        - Consistent frame quality
        """
        frames = []
        for _ in range(5):
            frame = camera.capture_array()
            frames.append(frame)
            time.sleep(0.2)
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (1080, 1920, 3)

    def test_camera_resolution_configuration(self):
        """
        Test different resolution configurations.
        
        Verifies:
        - Camera accepts various resolutions
        - Lower resolutions work (performance optimization)
        """
        resolutions = [(1920, 1080), (1280, 720), (640, 480)]
        
        for width, height in resolutions:
            cam = Picamera2()
            config = cam.create_still_configuration(
                main={"size": (width, height)}
            )
            cam.configure(config)
            cam.start()
            time.sleep(1)
            
            frame = cam.capture_array()
            assert frame.shape[0] == height
            assert frame.shape[1] == width
            
            cam.stop()
            cam.close()

    def test_camera_error_handling(self):
        """
        Test camera error scenarios.
        
        Verifies:
        - Proper exception on double start
        - Clean shutdown on errors
        """
        cam = Picamera2()
        config = cam.create_still_configuration()
        cam.configure(config)
        cam.start()
        
        # Attempting to start again should be handled
        with pytest.raises(Exception):
            cam.start()
        
        cam.stop()
        cam.close()


def test_camera_availability():
    """
    Test camera device availability at system level.
    
    Verifies:
    - Camera is detected by system
    - Camera interface is enabled
    """
    try:
        cam = Picamera2()
        cam.close()
        assert True
    except Exception as e:
        pytest.fail(f"Camera not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
pytest tests/integration/test_camera_hardware.py -v --cov --cov-report=term-missing
```

**Expected Coverage:** 100%

---

### 1.7 Touch Display Test

**Task:** Verify touch display functionality

**File:** `tests/integration/test_display_hardware.py`

```python
"""
Touch display hardware integration tests.
Tests 7-inch touch display functionality.

Coverage Target: 100%
"""

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QMainWindow
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtTest import QTest
import sys


class TestDisplayHardware:
    """Test suite for touch display integration."""

    @pytest.fixture(scope="module")
    def qapp(self):
        """
        Create QApplication instance for testing.
        
        Yields:
            QApplication: Qt application instance
        """
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

    @pytest.fixture
    def test_window(self, qapp):
        """
        Create test window for display testing.
        
        Args:
            qapp: QApplication instance
            
        Yields:
            QMainWindow: Test window
        """
        window = QMainWindow()
        window.setWindowTitle("Display Test")
        window.resize(800, 480)  # 7" display typical resolution
        
        button = QPushButton("Test Button", window)
        button.setGeometry(100, 100, 200, 100)
        button.setObjectName("testButton")
        
        yield window
        window.close()

    def test_display_initialization(self, test_window):
        """
        Test display window can be created.
        
        Verifies:
        - Window creation
        - Correct dimensions
        - Title set properly
        """
        assert test_window is not None
        assert test_window.windowTitle() == "Display Test"
        assert test_window.width() == 800
        assert test_window.height() == 480

    def test_touch_button_click(self, qtbot, test_window):
        """
        Test touch button interaction.
        
        Verifies:
        - Button can be clicked
        - Click event is triggered
        - Touch-sized buttons work
        
        Args:
            qtbot: PyQt test bot fixture
            test_window: Test window fixture
        """
        button = test_window.findChild(QPushButton, "testButton")
        assert button is not None
        
        # Simulate touch click
        with qtbot.waitSignal(button.clicked, timeout=1000):
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    def test_touch_button_size(self, test_window):
        """
        Test button meets minimum touch size requirements.
        
        Verifies:
        - Button is at least 60x60 pixels (touch-friendly)
        """
        button = test_window.findChild(QPushButton, "testButton")
        assert button.width() >= 60
        assert button.height() >= 60

    def test_display_resolution_detection(self, qapp):
        """
        Test display resolution detection.
        
        Verifies:
        - Screen resolution can be queried
        - Screen dimensions are reasonable for 7" display
        """
        screen = qapp.primaryScreen()
        assert screen is not None
        
        geometry = screen.geometry()
        # 7" displays typically 800x480 or 1024x600
        assert geometry.width() >= 800
        assert geometry.height() >= 480

    def test_fullscreen_mode(self, test_window):
        """
        Test fullscreen display mode.
        
        Verifies:
        - Window can enter fullscreen
        - Window state changes correctly
        """
        test_window.showFullScreen()
        assert test_window.isFullScreen()
        
        test_window.showNormal()
        assert not test_window.isFullScreen()

    def test_multiple_widgets_display(self, qapp):
        """
        Test multiple UI elements can be displayed simultaneously.
        
        Verifies:
        - Multiple widgets render correctly
        - No display artifacts
        """
        window = QMainWindow()
        
        buttons = []
        for i in range(5):
            btn = QPushButton(f"Button {i}", window)
            btn.setGeometry(50, 50 + i * 70, 150, 60)
            buttons.append(btn)
        
        window.show()
        assert len(buttons) == 5
        for btn in buttons:
            assert btn.isVisible()
        
        window.close()


def test_qt_platform_availability():
    """
    Test Qt platform is available.
    
    Verifies:
    - Qt libraries are installed
    - Display server is accessible
    """
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        assert app is not None
    except Exception as e:
        pytest.fail(f"Qt platform not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term-missing"])
```

**Run Tests:**
```bash
# May require DISPLAY environment variable on headless systems
export DISPLAY=:0
pytest tests/integration/test_display_hardware.py -v --cov --cov-report=term-missing
```

**Expected Coverage:** 100%

---

### 1.8 Pytest Configuration

**Task:** Configure pytest for consistent test execution

**File:** `pytest.ini`

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*
testpaths = tests

# Output options
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=100

# Markers
markers =
    unit: Unit tests (isolated, fast)
    integration: Integration tests (hardware/system)
    e2e: End-to-end tests (full workflow)
    slow: Slow running tests
    hardware: Requires physical hardware

# Coverage settings
[coverage:run]
source = src
omit = 
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False

[coverage:html]
directory = htmlcov
```

**Usage:**
```bash
# Run all tests with coverage
pytest

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/integration/test_camera_hardware.py

# Generate HTML coverage report
pytest --cov --cov-report=html
```

---

## ⚠️ Phase 1 Risks & Mitigations

### Risk: Pi Performance Too Slow

**Impact:** Camera processing or UI responsiveness insufficient

**Mitigation Strategy:**
1. Start with reduced camera resolution (1280x720 vs 1920x1080)
2. Profile code to identify bottlenecks early
3. Consider Raspberry Pi 5 upgrade if Pi 4 insufficient
4. Optimize OpenCV operations in Phase 2

**Test:**
```python
# Performance baseline test
def test_camera_capture_performance(camera):
    """Measure capture time for performance planning."""
    import time
    
    times = []
    for _ in range(10):
        start = time.time()
        frame = camera.capture_array()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    assert avg_time < 0.5  # Must capture in under 500ms
    print(f"Average capture time: {avg_time:.3f}s")
```

---

## ✅ Phase 1 Completion Checklist

### Environment
- [ ] Raspberry Pi OS installed and updated
- [ ] Python 3.11+ verified
- [ ] All hardware components connected
- [ ] Display configured and tested
- [ ] Camera interface enabled

### Project Setup
- [ ] Directory structure created
- [ ] requirements.txt installed successfully
- [ ] Git repository initialized
- [ ] README.md created
- [ ] .gitignore configured

### Testing
- [ ] pytest.ini configured
- [ ] Camera hardware tests pass (100% coverage)
- [ ] Display hardware tests pass (100% coverage)
- [ ] All dependencies import without errors
- [ ] Coverage reports generated successfully

### Performance Baseline
- [ ] Camera capture time < 500ms
- [ ] Display renders without lag
- [ ] Touch response time acceptable

### Documentation
- [ ] All code has docstrings
- [ ] Test purposes documented
- [ ] Hardware setup documented

---

## 📊 Coverage Report Example

**Expected Output:**
```
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_initialization PASSED
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_capture_single_frame PASSED
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_autofocus PASSED
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_multiple_captures PASSED
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_resolution_configuration PASSED
tests/integration/test_camera_hardware.py::TestCameraHardware::test_camera_error_handling PASSED
tests/integration/test_camera_hardware.py::test_camera_availability PASSED

----------- coverage: 100% -----------
Name                                              Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------------
tests/integration/test_camera_hardware.py           127      0   100%
tests/integration/test_display_hardware.py           98      0   100%
-------------------------------------------------------------------------------
TOTAL                                               225      0   100%
```

---

## 🔗 Next Steps

Once Phase 1 is complete with 100% coverage:

**Proceed to:** [Phase 2 – Vision Engine](phase2-vision.md)

**Phase 2 Requirements:**
- Camera capture working reliably
- OpenCV installed and tested
- Project structure established
- Git repository active

---

**Phase Status:** 🔴 Not Started  
**Estimated Duration:** 2-4 days  
**Dependencies Installed:** 0/12  
**Test Coverage:** 0%
