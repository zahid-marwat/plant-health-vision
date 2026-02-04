"""
Main training script entry point.

Script to run full training pipeline.
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from src.data.data_generator import PyTorchDataLoader
from src.data.augmentation import PlantDiseaseAugmentor
from src.models.transfer_learning import create_model as create_transfer_model
from src.models.base_cnn import create_baseline_cnn
from src.models.custom_models import create_custom_resnet
from src.models.yolo_models import create_yolov8_classify
from src.training.trainer import ModelTrainer
from src.training.config import (
    get_resnet50_config, get_mobilenetv2_config, 
    get_efficient_config, get_yolov8_config, get_baseline_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(config):
    """Create model based on configuration."""
    if config.model_name == 'resnet50':
        return create_transfer_model('resnet50', config.num_classes, 
                                     config.pretrained, config.freeze_backbone)
    
    elif config.model_name == 'mobilenetv2':
        return create_transfer_model('mobilenetv2', config.num_classes,
                                     config.pretrained, config.freeze_backbone)
    
    elif config.model_name == 'efficientnetb0':
        return create_transfer_model('efficientnetb0', config.num_classes,
                                     config.pretrained, config.freeze_backbone)
    
    elif config.model_name == 'vgg16':
        return create_transfer_model('vgg16', config.num_classes,
                                     config.pretrained, config.freeze_backbone)
    
    elif config.model_name == 'baseline_cnn':
        return create_baseline_cnn(config.num_classes)
    
    elif config.model_name == 'yolov8':
        return create_yolov8_classify(config.num_classes)
    
    elif config.model_name == 'custom_resnet':
        return create_custom_resnet(config.num_classes)
    
    else:
        raise ValueError(f"Unknown model: {config.model_name}")


def create_optimizer(model, config):
    """Create optimizer."""
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate,
                         weight_decay=config.weight_decay)
    
    elif config.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate,
                        momentum=config.momentum, weight_decay=config.weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_config_for_model(model_name):
    """Get default config for model."""
    configs = {
        'resnet50': get_resnet50_config,
        'mobilenetv2': get_mobilenetv2_config,
        'efficientnetb0': get_efficient_config,
        'yolov8': get_yolov8_config,
        'baseline_cnn': get_baseline_config
    }
    
    if model_name in configs:
        return configs[model_name]()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train plant disease detection model')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Get configuration
    config = get_config_for_model(args.model)
    config.model_name = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.device = device
    
    # Save config
    config.to_yaml('config_training.yaml')
    logger.info("Config saved to config_training.yaml")
    
    # Create data loaders
    logger.info("Loading dataset...")
    transforms_train = PlantDiseaseAugmentor.get_train_transforms(config.img_size)
    transforms_val = PlantDiseaseAugmentor.get_val_transforms(config.img_size)
    
    data_loader = PyTorchDataLoader(
        args.data_dir,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers,
        transform_train=transforms_train,
        transform_val=transforms_val
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    # Create model
    logger.info(f"Creating model: {config.model_name}...")
    model = create_model(config)
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = ModelTrainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        patience=config.patience
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(config.epochs)
    
    # Save final model
    final_model_path = Path(config.checkpoint_dir) / f'{config.model_name}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save history
    history_path = Path(config.checkpoint_dir) / f'{config.model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
