@echo off
REM BullSight Startup Script for Windows
REM Author: Mario Neuhauser

if not exist venv (
    echo ❌ Virtual environment not found. Please run install.bat first.
    exit /b 1
)

call venv\Scripts\activate.bat
set PYTHONPATH=%CD%

echo 🎯 Starting BullSight...
python src\main.py
