"""
Image augmentation utilities using albumentations library.

Provides advanced augmentation techniques for better model generalization.
"""

from typing import Tuple, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class PlantDiseaseAugmentor:
    """Image augmentation for plant disease detection."""

    @staticmethod
    def get_train_transforms(
        img_size: Tuple[int, int] = (224, 224),
        p: float = 0.5
    ) -> Callable:
        """
        Get training augmentation pipeline.
        
        Args:
            img_size: Target image size
            p: Probability of applying augmentation
            
        Returns:
            Albumentation compose object
        """
        return A.Compose([
            A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=1.0),
            A.Rotate(limit=20, p=p),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))

    @staticmethod
    def get_val_transforms(
        img_size: Tuple[int, int] = (224, 224)
    ) -> Callable:
        """
        Get validation augmentation pipeline (minimal).
        
        Args:
            img_size: Target image size
            
        Returns:
            Albumentation compose object
        """
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def get_test_transforms(
        img_size: Tuple[int, int] = (224, 224)
    ) -> Callable:
        """
        Get test augmentation pipeline (minimal).
        
        Args:
            img_size: Target image size
            
        Returns:
            Albumentation compose object
        """
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def augment_image(image: np.ndarray, augmentation: Callable) -> np.ndarray:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image (H, W, C) in BGR or RGB
            augmentation: Augmentation function
            
        Returns:
            Augmented image
        """
        augmented = augmentation(image=image)
        return augmented['image']

    @staticmethod
    def visualize_augmentations(
        image: np.ndarray,
        augmentation: Callable,
        n_samples: int = 4
    ) -> list:
        """
        Generate multiple augmented versions of image.
        
        Args:
            image: Input image
            augmentation: Augmentation function
            n_samples: Number of augmented samples to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for _ in range(n_samples):
            aug_image = PlantDiseaseAugmentor.augment_image(image, augmentation)
            augmented_images.append(aug_image)
        
        return augmented_images


class CustomAugmentation:
    """Custom augmentation strategies for plant disease images."""

    @staticmethod
    def get_aggressive_augmentation(
        img_size: Tuple[int, int] = (224, 224)
    ) -> Callable:
        """
        Aggressive augmentation for limited data.
        
        Returns:
            Augmentation pipeline
        """
        return A.Compose([
            A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.6, 1.0), p=1.0),
            A.Rotate(limit=30, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomShadow(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def get_light_augmentation(
        img_size: Tuple[int, int] = (224, 224)
    ) -> Callable:
        """
        Light augmentation for large dataset.
        
        Returns:
            Augmentation pipeline
        """
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def get_medical_imaging_augmentation(
        img_size: Tuple[int, int] = (224, 224)
    ) -> Callable:
        """
        Augmentation optimized for medical/pathology imaging.
        
        Returns:
            Augmentation pipeline
        """
        return A.Compose([
            A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.3, p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
