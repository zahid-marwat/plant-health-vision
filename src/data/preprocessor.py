"""
Image preprocessing and normalization utilities.

Handles image loading, resizing, normalization, and class balancing.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing utilities."""

    STANDARD_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STANDARD_STD = [0.229, 0.224, 0.225]   # ImageNet std

    @staticmethod
    def load_image(image_path: str, mode: str = 'RGB') -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            mode: 'RGB' or 'BGR'
            
        Returns:
            Image as numpy array
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        if mode == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    @staticmethod
    def resize_image(
        image: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = 'bicubic'
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            size: Target size (H, W)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        method = interp_methods.get(interpolation, cv2.INTER_CUBIC)
        return cv2.resize(image, (size[1], size[0]), interpolation=method)

    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: List[float] = None,
        std: List[float] = None
    ) -> np.ndarray:
        """
        Normalize image using mean and std.
        
        Args:
            image: Input image (H, W, C) with values 0-255
            mean: Mean values for each channel
            std: Std values for each channel
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = ImagePreprocessor.STANDARD_MEAN
        if std is None:
            std = ImagePreprocessor.STANDARD_STD
        
        # Convert to float32 and normalize to 0-1
        image = image.astype(np.float32) / 255.0
        
        # Apply channel-wise normalization
        if len(image.shape) == 3 and image.shape[2] == 3:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image

    @staticmethod
    def denormalize_image(
        image: np.ndarray,
        mean: List[float] = None,
        std: List[float] = None
    ) -> np.ndarray:
        """
        Reverse normalization to recover original pixel values.
        
        Args:
            image: Normalized image
            mean: Mean values used in normalization
            std: Std values used in normalization
            
        Returns:
            Denormalized image (0-1 range)
        """
        if mean is None:
            mean = ImagePreprocessor.STANDARD_MEAN
        if std is None:
            std = ImagePreprocessor.STANDARD_STD
        
        image = image.copy()
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            for i in range(3):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        
        return np.clip(image, 0, 1)

    @staticmethod
    def preprocess(
        image_path: str,
        size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to image
            size: Target size
            normalize: Whether to normalize
            
        Returns:
            Preprocessed image
        """
        image = ImagePreprocessor.load_image(image_path, mode='RGB')
        image = ImagePreprocessor.resize_image(image, size)
        
        if normalize:
            image = ImagePreprocessor.normalize_image(image)
        else:
            image = image.astype(np.float32) / 255.0
        
        return image


class ClassBalancer:
    """Handle class imbalance in dataset."""

    @staticmethod
    def get_class_weights(data_dir: str) -> Dict[int, float]:
        """
        Calculate class weights based on sample frequency.
        
        Args:
            data_dir: Path to data directory with class subdirectories
            
        Returns:
            Dictionary mapping class index to weight
        """
        class_counts = {}
        total_samples = 0
        
        # Count samples per class
        for class_idx, class_dir in enumerate(sorted(Path(data_dir).iterdir())):
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*')))
                class_counts[class_idx] = count
                total_samples += count
        
        # Calculate weights (inverse frequency)
        class_weights = {}
        for class_idx, count in class_counts.items():
            if count > 0:
                class_weights[class_idx] = total_samples / (len(class_counts) * count)
        
        return class_weights

    @staticmethod
    def get_sample_weights(
        image_labels: List[int],
        class_weights: Dict[int, float]
    ) -> np.ndarray:
        """
        Get sample weights based on class weights.
        
        Args:
            image_labels: List of class labels
            class_weights: Class weight dictionary
            
        Returns:
            Array of sample weights
        """
        weights = np.array([class_weights.get(label, 1.0) for label in image_labels])
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        return weights

    @staticmethod
    def calculate_loss_weights(data_dir: str) -> Dict[int, float]:
        """
        Calculate loss weights for handling class imbalance.
        
        Args:
            data_dir: Path to training data directory
            
        Returns:
            Dictionary of class weights for loss function
        """
        class_weights = ClassBalancer.get_class_weights(data_dir)
        
        # Normalize to average of 1.0
        avg_weight = np.mean(list(class_weights.values()))
        normalized_weights = {k: v / avg_weight for k, v in class_weights.items()}
        
        return normalized_weights


class DataValidator:
    """Validate dataset integrity and quality."""

    @staticmethod
    def check_corrupted_images(data_dir: str) -> List[str]:
        """
        Check for corrupted images in dataset.
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            List of corrupted image paths
        """
        corrupted = []
        
        for img_path in Path(data_dir).rglob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    Image.open(img_path).verify()
                except Exception as e:
                    corrupted.append(str(img_path))
                    logger.warning(f"Corrupted image: {img_path} - {str(e)}")
        
        return corrupted

    @staticmethod
    def remove_small_images(data_dir: str, min_size: int = 100) -> List[str]:
        """
        Remove images smaller than minimum size.
        
        Args:
            data_dir: Path to data directory
            min_size: Minimum image size (pixels)
            
        Returns:
            List of removed image paths
        """
        removed = []
        
        for img_path in Path(data_dir).rglob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    image = Image.open(img_path)
                    width, height = image.size
                    
                    if width < min_size or height < min_size:
                        img_path.unlink()
                        removed.append(str(img_path))
                        logger.info(f"Removed small image: {img_path} ({width}x{height})")
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {str(e)}")
        
        return removed
