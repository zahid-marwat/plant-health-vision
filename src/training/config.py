"""
Training configuration and hyperparameters.

Centralized configuration for model training experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model configuration
    model_name: str = 'resnet50'
    num_classes: int = 38
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout_rate: float = 0.5
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'exponential'
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Data configuration
    data_dir: str = 'data/processed'
    img_size: tuple = (224, 224)
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_type: str = 'standard'  # 'standard', 'aggressive', 'light', 'medical'
    
    # Regularization
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # Device
    device: str = 'cuda'
    mixed_precision: bool = False
    
    # Logging
    log_dir: str = 'results/tensorboard_logs'
    checkpoint_dir: str = 'models'
    save_frequency: int = 5  # Save every N epochs
    
    # Class weights for imbalanced data
    use_class_weights: bool = True
    class_weight_type: str = 'inverse_freq'  # 'inverse_freq', 'effective_num'
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    model_path: str = 'models/best_model.pth'
    test_dir: str = 'data/processed/test'
    batch_size: int = 32
    num_workers: int = 4
    device: str = 'cuda'
    
    # Metrics
    calculate_roc: bool = True
    calculate_confusion_matrix: bool = True
    calculate_grad_cam: bool = True
    
    # Output
    output_dir: str = 'results'
    save_visualizations: bool = True
    save_csv: bool = True


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_resnet50_config() -> TrainingConfig:
    """Get ResNet50 specific configuration."""
    config = TrainingConfig()
    config.model_name = 'resnet50'
    config.batch_size = 32
    config.learning_rate = 0.001
    config.epochs = 50
    return config


def get_mobilenetv2_config() -> TrainingConfig:
    """Get MobileNetV2 specific configuration."""
    config = TrainingConfig()
    config.model_name = 'mobilenetv2'
    config.batch_size = 64
    config.learning_rate = 0.001
    config.epochs = 40
    return config


def get_efficient_config() -> TrainingConfig:
    """Get EfficientNet specific configuration."""
    config = TrainingConfig()
    config.model_name = 'efficientnetb0'
    config.batch_size = 48
    config.learning_rate = 0.001
    config.epochs = 50
    return config


def get_yolov8_config() -> TrainingConfig:
    """Get YOLOv8 specific configuration."""
    config = TrainingConfig()
    config.model_name = 'yolov8'
    config.batch_size = 16
    config.learning_rate = 0.01
    config.epochs = 100
    config.img_size = (640, 640)
    return config


def get_baseline_config() -> TrainingConfig:
    """Get baseline CNN configuration."""
    config = TrainingConfig()
    config.model_name = 'baseline_cnn'
    config.batch_size = 64
    config.learning_rate = 0.01
    config.epochs = 30
    config.pretrained = False
    return config
