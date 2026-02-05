#!/bin/bash
# Virtual Environment Setup Script for Plant Disease Detection Project

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Plant Disease Detection - Virtual Environment Setup       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

echo "Python found:"
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

echo "Requirements installed"

# Install dev dependencies
echo ""
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Create project structure
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed models results logs

echo "Project directories created"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETED                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo "   1. Download PlantVillage dataset from Kaggle"
echo "   2. Extract to data/raw/"
echo "   3. Run: python src/data/dataset_loader.py --organize"
echo "   4. Run notebooks in order"
echo "   5. Train model: python src/training/train_main.py --model resnet50"
echo ""
echo "To activate the environment in new terminals:"
echo "   source venv/bin/activate"
echo ""
