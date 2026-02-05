"""
Data generators for training and evaluation using TensorFlow and PyTorch.

Provides efficient data loading with batch processing and preprocessing.
"""

import os
from typing import Tuple, List
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TFPlantDiseaseDataset:
    """TensorFlow data generator for plant disease dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: Tuple[int, int] = (224, 224),
        shuffle: bool = True
    ):
        """
        Initialize TensorFlow dataset generator.
        
        Args:
            data_dir: Path to dataset directory
            batch_size: Batch size for training
            img_size: Target image size (H, W)
            shuffle: Whether to shuffle data
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

    def get_train_datagen(self) -> ImageDataGenerator:
        """
        Get training data generator with augmentation.
        
        Returns:
            ImageDataGenerator for training data
        """
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def get_val_datagen(self) -> ImageDataGenerator:
        """
        Get validation data generator (no augmentation).
        
        Returns:
            ImageDataGenerator for validation data
        """
        return ImageDataGenerator(rescale=1./255)

    def get_train_generator(self):
        """
        Get training data generator.
        
        Returns:
            tf.keras.utils.Sequence for training
        """
        datagen = self.get_train_datagen()
        
        return datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=self.shuffle
        )

    def get_val_generator(self):
        """Get validation data generator."""
        datagen = self.get_val_datagen()
        
        return datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def get_test_generator(self):
        """Get test data generator."""
        datagen = self.get_val_datagen()
        
        return datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )


class PyTorchPlantDiseaseDataset(Dataset):
    """PyTorch dataset for plant disease images."""

    def __init__(
        self,
        img_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        transform=None,
        class_to_idx: dict = None
    ):
        """
        Initialize PyTorch dataset.
        
        Args:
            img_dir: Directory containing image subdirectories (one per class)
            img_size: Target image size
            transform: Image transformations to apply
            class_to_idx: Mapping from class name to index
        """
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Build image paths and labels
        self.img_paths = []
        self.labels = []
        
        if class_to_idx is None:
            # Create class mapping
            classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
            class_to_idx = {c: i for i, c in enumerate(classes)}
        
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Load image paths
        for class_name, class_idx in class_to_idx.items():
            class_dir = self.img_dir / class_name
            
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.img_paths.append(str(img_path))
                        self.labels.append(class_idx)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size, Image.Resampling.LANCZOS)
        
        if self.transform:
            # Albumentations expects numpy arrays and named arguments
            image_np = np.array(image)
            transformed = self.transform(image=image_np) if callable(self.transform) else None

            if isinstance(transformed, dict) and 'image' in transformed:
                image = transformed['image']
            else:
                # Fallback for torchvision-style transforms
                image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, label


class PyTorchDataLoader:
    """PyTorch data loader wrapper."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: Tuple[int, int] = (224, 224),
        num_workers: int = 4,
        transform_train=None,
        transform_val=None
    ):
        """
        Initialize PyTorch data loaders.
        
        Args:
            data_dir: Root data directory
            batch_size: Batch size
            img_size: Image size
            num_workers: Number of workers for loading
            transform_train: Training transforms
            transform_val: Validation transforms
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = PyTorchPlantDiseaseDataset(
            os.path.join(data_dir, 'train'),
            img_size,
            transform_train
        )
        
        self.val_dataset = PyTorchPlantDiseaseDataset(
            os.path.join(data_dir, 'val'),
            img_size,
            transform_val,
            self.train_dataset.class_to_idx
        )
        
        self.test_dataset = PyTorchPlantDiseaseDataset(
            os.path.join(data_dir, 'test'),
            img_size,
            transform_val,
            self.train_dataset.class_to_idx
        )
        
        self.num_classes = len(self.train_dataset.class_to_idx)

    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
