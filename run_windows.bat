@echo off
echo ===================================================
echo  SDXL to Diffusers Converter - Automated Setup
echo ===================================================
echo.

:: Check if python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and try again.
    pause
    exit /b 1
)

echo [1/3] Checking and upgrading required dependencies...
echo Updating to the latest stable versions to prevent conflicts and import issues...
pip install --upgrade torch torchvision
pip install --upgrade diffusers transformers accelerate PyQt6
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Upgrade encountered an issue, performing fallback installation...
    pip install torch torchvision diffusers transformers accelerate PyQt6
)
echo.

echo [2/3] Checking for sdxl_converter.py...
if not exist sdxl_converter.py (
    echo ERROR: sdxl_converter.py not found in the current directory!
    echo Please ensure both files are in the same folder.
    pause
    exit /b 1
)
echo.

echo [3/3] Launching Converter GUI...
python sdxl_converter.py

echo.
echo Finished!
pause
