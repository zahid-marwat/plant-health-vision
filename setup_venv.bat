@echo off
REM Virtual Environment Setup Script for Plant Disease Detection Project

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║     Plant Disease Detection - Virtual Environment Setup       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo ✓ Python found:
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo Installing requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✓ Requirements installed

REM Install dev dependencies
echo.
echo Installing development dependencies...
pip install -r requirements-dev.txt

if errorlevel 1 (
    echo ⚠ Failed to install dev requirements (non-critical)
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"

REM Create project structure
echo.
echo Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "logs" mkdir logs

echo ✓ Project directories created

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                    SETUP COMPLETED ✓                         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo 📝 NEXT STEPS:
echo    1. Download PlantVillage dataset from Kaggle
echo    2. Extract to data/raw/
echo    3. Run: python src/data/dataset_loader.py --organize
echo    4. Run notebooks in order:
echo       - 01_data_exploration.ipynb
echo       - 02_preprocessing.ipynb
echo       - 03_baseline_model.ipynb
echo       - 04_model_comparison.ipynb
echo    5. Train model: python src/training/train_main.py --model resnet50
echo.
echo 🚀 STARTING WEB API:
echo    python inference/app.py --port 8000
echo.
echo 📚 DOCUMENTATION:
echo    - See README.md for detailed instructions
echo    - Check config.py for hyperparameter tuning
echo.

pause
