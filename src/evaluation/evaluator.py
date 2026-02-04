"""
Model evaluation utilities.

Calculate metrics like accuracy, precision, recall, F1-score per class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        num_classes: int = 38
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to evaluate on
            num_classes: Number of classes
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.model.eval()

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions on test set.
        
        Returns:
            Tuple of (predictions, true_labels, confidences)
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probs)
        )

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate overall metrics.
        
        Returns:
            Dictionary with metrics
        """
        predictions, true_labels, probs = self.get_predictions()
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(true_labels, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0)
        }
        
        return metrics

    def calculate_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Returns:
            Dictionary with per-class metrics
        """
        predictions, true_labels, _ = self.get_predictions()
        
        per_class_metrics = {}
        
        for class_idx in range(self.num_classes):
            # Binary classification for this class
            binary_true = (true_labels == class_idx).astype(int)
            binary_pred = (predictions == class_idx).astype(int)
            
            if binary_true.sum() > 0:  # Only if class exists in test set
                per_class_metrics[class_idx] = {
                    'precision': precision_score(binary_true, binary_pred, zero_division=0),
                    'recall': recall_score(binary_true, binary_pred, zero_division=0),
                    'f1': f1_score(binary_true, binary_pred, zero_division=0),
                    'support': binary_true.sum()
                }
        
        return per_class_metrics

    def calculate_roc_auc(self) -> Dict[int, float]:
        """
        Calculate ROC-AUC score per class.
        
        Returns:
            Dictionary with per-class ROC-AUC scores
        """
        _, true_labels, probs = self.get_predictions()
        
        # Binarize labels
        binary_labels = label_binarize(true_labels, classes=range(self.num_classes))
        
        roc_aucs = {}
        
        for class_idx in range(self.num_classes):
            try:
                roc_auc = roc_auc_score(
                    binary_labels[:, class_idx],
                    probs[:, class_idx]
                )
                roc_aucs[class_idx] = roc_auc
            except:
                roc_aucs[class_idx] = None
        
        return roc_aucs

    def print_metrics(self) -> None:
        """Print evaluation metrics."""
        metrics = self.calculate_metrics()
        per_class_metrics = self.calculate_per_class_metrics()
        
        print("\n" + "="*70)
        print("MODEL EVALUATION METRICS")
        print("="*70)
        
        print("\nOVERALL METRICS:")
        print("-"*70)
        for metric, value in metrics.items():
            print(f"{metric:.<30} {value:.4f}")
        
        print("\nPER-CLASS METRICS (Top 10):")
        print("-"*70)
        print(f"{'Class ID':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*70)
        
        sorted_classes = sorted(per_class_metrics.items(),
                               key=lambda x: x[1]['support'], reverse=True)[:10]
        
        for class_idx, class_metrics in sorted_classes:
            print(f"{class_idx:<10} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f} "
                  f"{class_metrics['support']:<10}")
        
        print("\n" + "="*70 + "\n")
