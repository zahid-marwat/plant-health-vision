"""
Visualization utilities for evaluation results.

Generate plots for confusion matrix, ROC curves, training history, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Visualize model evaluation results."""

    def __init__(self, output_dir: str = 'results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        figsize: Tuple[int, int] = (16, 14)
    ) -> str:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = self.output_dir / 'confusion_matrix.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {filepath}")
        return str(filepath)

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        figsize: Tuple[int, int] = (15, 5)
    ) -> str:
        """
        Plot training history.
        
        Args:
            history: Dictionary with train/val losses and accuracies
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            ax1.plot(history['train_loss'], label='Train Loss')
            ax1.plot(history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training History - Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_acc' in history and 'val_acc' in history:
            ax2.plot(history['train_acc'], label='Train Acc')
            ax2.plot(history['val_acc'], label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training History - Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'training_history.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history saved to {filepath}")
        return str(filepath)

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: List[str] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Plot ROC curves for top N classes.
        
        Args:
            y_true: True labels
            y_probs: Prediction probabilities
            class_names: List of class names
            top_n: Number of classes to plot
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        num_classes = y_probs.shape[1]
        
        plt.figure(figsize=figsize)
        
        for i in range(min(top_n, num_classes)):
            # Binary classification for class i
            binary_true = (y_true == i).astype(int)
            
            if binary_true.sum() > 0:
                fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_label = class_names[i] if class_names else f"Class {i}"
                plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (Top {top_n} Classes)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'roc_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {filepath}")
        return str(filepath)

    def plot_class_distribution(
        self,
        y_labels: np.ndarray,
        class_names: List[str] = None,
        figsize: Tuple[int, int] = (15, 6)
    ) -> str:
        """
        Plot class distribution.
        
        Args:
            y_labels: Array of labels
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        unique, counts = np.unique(y_labels, return_counts=True)
        
        plt.figure(figsize=figsize)
        bars = plt.bar(unique, counts)
        
        # Color bars by count
        colors = plt.cm.viridis(counts / counts.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        if class_names:
            labels = [class_names[i] if i < len(class_names) else f"Class {i}" 
                     for i in unique]
            plt.xticks(unique, labels, rotation=45, ha='right')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / 'class_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution saved to {filepath}")
        return str(filepath)

    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict[int, Dict[str, float]],
        class_names: List[str] = None,
        top_n: int = 15,
        figsize: Tuple[int, int] = (14, 6)
    ) -> str:
        """
        Plot per-class metrics.
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            class_names: List of class names
            top_n: Number of top classes to plot
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        # Sort by support
        sorted_metrics = sorted(per_class_metrics.items(),
                               key=lambda x: x[1].get('support', 0), reverse=True)[:top_n]
        
        class_indices = [x[0] for x in sorted_metrics]
        precision = [x[1]['precision'] for x in sorted_metrics]
        recall = [x[1]['recall'] for x in sorted_metrics]
        f1 = [x[1]['f1'] for x in sorted_metrics]
        
        class_labels = [class_names[i] if class_names and i < len(class_names)
                       else f"Class {i}" for i in class_indices]
        
        x = np.arange(len(class_labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1-Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics (Top {top_n})')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.output_dir / 'per_class_metrics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class metrics plot saved to {filepath}")
        return str(filepath)
