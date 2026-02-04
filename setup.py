"""
Setup script for package installation.

Configure package for pip install.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plant-health-vision",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Plant Disease Detection using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plant-health-vision",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.14.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "opencv-python>=4.8.0.74",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "albumentations>=1.3.0",
        "fastapi>=0.103.0",
        "uvicorn>=0.23.2",
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "plant-disease-train=src.training.train_main:main",
            "plant-disease-predict=inference.predict:main",
            "plant-disease-batch-predict=inference.batch_predict:main",
        ],
    },
)
