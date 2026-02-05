@echo off
REM BullSight Installation Script for Windows (Development)
REM Author: Mario Neuhauser

echo.
echo 🎯 BullSight Installation Script (Windows Development)
echo =====================================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.11 or higher.
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist venv (
    echo    Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📥 Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt
echo ✅ Dependencies installed
echo.

REM Create necessary directories
echo 📁 Creating project directories...
if not exist logs mkdir logs
if not exist config\calibration mkdir config\calibration
if not exist assets\sounds mkdir assets\sounds
if not exist assets\images mkdir assets\images
echo ✅ Directories created
echo.

REM Run tests to verify installation
echo 🧪 Running tests to verify installation...
pytest tests\unit\test_ui_logic.py -v
if errorlevel 1 (
    echo ⚠️  Some tests failed. Please check the output above.
) else (
    echo ✅ Basic tests passed!
)
echo.

REM Display next steps
echo ================================================
echo ✅ Installation complete!
echo.
echo Next steps:
echo   1. Activate virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Set PYTHONPATH (in each session):
echo      set PYTHONPATH=%%CD%%
echo.
echo   3. Run the application:
echo      python src\main.py
echo.
echo   4. Run all tests:
echo      pytest tests\ -v
echo.
echo Note: This is a development setup for Windows.
echo       Some features (camera) require Raspberry Pi hardware.
echo ================================================
pause
