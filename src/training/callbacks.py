"""
Custom callbacks for training monitoring.

Additional training callbacks for logging and monitoring.
"""

import logging
from typing import Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at epoch start."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Called at epoch end."""
        pass

    def on_train_start(self, epochs: int) -> None:
        """Called at training start."""
        pass

    def on_train_end(self, logs: dict) -> None:
        """Called at training end."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.001
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.wait_count = 0
        self.stopped = False

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Check early stopping condition."""
        if self.monitor not in logs:
            logger.warning(f"Metric {self.monitor} not found in logs")
            return
        
        current_value = logs[self.monitor]
        
        # Determine if improvement
        if self.best_value is None:
            self.best_value = current_value
        elif 'loss' in self.monitor:
            # For loss, lower is better
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.wait_count = 0
                logger.info(f"Improvement: {self.monitor} = {current_value:.4f}")
            else:
                self.wait_count += 1
        else:
            # For accuracy/metrics, higher is better
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.wait_count = 0
                logger.info(f"Improvement: {self.monitor} = {current_value:.4f}")
            else:
                self.wait_count += 1
        
        if self.wait_count >= self.patience:
            logger.info(f"Early stopping: no improvement in {self.monitor} for {self.patience} epochs")
            self.stopped = True


class SchedulerCallback(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler):
        """
        Initialize scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Update learning rate."""
        if hasattr(self.scheduler, 'step'):
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")


class LoggingCallback(Callback):
    """Logging callback for metrics tracking."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize logging callback.
        
        Args:
            log_file: Optional file to log to
        """
        self.log_file = log_file
        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Log metrics."""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}: {logs}\n")
        
        logger.info(f"Metrics: {logs}")


class CheckpointCallback(Callback):
    """Save model checkpoint callback."""

    def __init__(
        self,
        checkpoint_dir: str = 'models',
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_frequency: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            save_best_only: Only save when metric improves
            save_frequency: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.best_value = None

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Save checkpoint if needed."""
        # Check frequency
        if epoch % self.save_frequency != 0:
            return
        
        # Check if best
        if self.save_best_only:
            if self.monitor not in logs:
                logger.warning(f"Metric {self.monitor} not found")
                return
            
            current_value = logs[self.monitor]
            
            if self.best_value is None:
                self.best_value = current_value
            elif 'loss' in self.monitor and current_value < self.best_value:
                self.best_value = current_value
            elif 'acc' in self.monitor and current_value > self.best_value:
                self.best_value = current_value
            else:
                return
        
        # Save checkpoint (handled by trainer)
        logger.info(f"Saving checkpoint at epoch {epoch}")
