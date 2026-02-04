"""
Training loop and utilities.

Handles model training with early stopping, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: str = 'models',
        patience: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'acc': total_correct / total_samples
            })
        
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
        
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def fit(self, epochs: int) -> Dict:
        """
        Train model for specified epochs.
        
        Args:
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                logger.info("✓ Validation loss improved!")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"\nTraining completed in {elapsed_time}")
        
        self.history['total_time'] = str(elapsed_time)
        return self.history
