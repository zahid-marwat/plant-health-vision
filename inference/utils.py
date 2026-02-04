"""
Inference utilities for preprocessing and prediction.

Helper functions for inference pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch


class InferencePreprocessor:
    """Preprocessing for inference."""

    @staticmethod
    def load_and_preprocess(
        image_path: str,
        size: tuple = (224, 224),
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to image
            size: Target size
            device: Device to put tensor on
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (size[1], size[0]))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(device)
        
        return image_tensor


class PredictionFormatter:
    """Format prediction outputs."""

    @staticmethod
    def format_result(
        image_path: str,
        disease_name: str,
        confidence: float,
        class_id: int
    ) -> dict:
        """Format prediction result."""
        return {
            'image': str(image_path),
            'disease': disease_name,
            'class_id': class_id,
            'confidence': float(confidence),
            'confidence_percent': f"{float(confidence)*100:.2f}%"
        }

    @staticmethod
    def format_top5(
        predictions: list,
        class_map: dict
    ) -> list:
        """Format top 5 predictions."""
        formatted = []
        for prob, class_idx in predictions:
            formatted.append({
                'disease': class_map.get(str(class_idx.item()), f"Class {class_idx.item()}"),
                'confidence': float(prob.item()),
                'confidence_percent': f"{float(prob.item())*100:.2f}%"
            })
        return formatted
