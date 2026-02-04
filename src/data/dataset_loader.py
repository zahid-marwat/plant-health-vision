"""
Data loading and organization utilities for PlantVillage dataset.

This module handles downloading, extracting, and organizing the PlantVillage
dataset into train/validation/test splits.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import requests
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)


class PlantVillageDatasetLoader:
    """Load and organize PlantVillage dataset."""

    # PlantVillage dataset information
    DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/arjuntejaswi/plant-village"
    SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    CROPS = [
        'Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange',
        'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean',
        'Squash', 'Strawberry', 'Tomato'
    ]

    def __init__(self, output_dir: str, seed: int = 42):
        """
        Initialize dataset loader.
        
        Args:
            output_dir: Directory to save dataset
            seed: Random seed for reproducible splits
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        logging.basicConfig(level=logging.INFO)

    def create_class_mapping(self, data_dir: Path) -> Dict[int, str]:
        """
        Create mapping from class indices to disease names.
        
        Args:
            data_dir: Path to data directory with disease folders
            
        Returns:
            Dictionary mapping class indices to disease names
        """
        class_map = {}
        class_idx = 0
        
        for crop in sorted(os.listdir(data_dir)):
            crop_path = os.path.join(data_dir, crop)
            if os.path.isdir(crop_path):
                for disease in sorted(os.listdir(crop_path)):
                    disease_path = os.path.join(crop_path, disease)
                    if os.path.isdir(disease_path):
                        class_name = f"{crop}_{disease}"
                        class_map[class_idx] = class_name
                        class_idx += 1
        
        return class_map

    def organize_dataset(
        self,
        raw_dir: str,
        processed_dir: str
    ) -> None:
        """
        Organize dataset into train/val/test splits.
        
        Args:
            raw_dir: Directory containing raw dataset
            processed_dir: Directory to save organized splits
        """
        raw_path = Path(raw_dir)
        processed_path = Path(processed_dir)
        
        logger.info("Organizing dataset into train/val/test splits...")
        
        # Create split directories
        for split in self.SPLITS.keys():
            split_dir = processed_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each disease class
        import random
        random.seed(self.seed)
        
        class_count = 0
        total_files = 0
        
        for root, dirs, files in os.walk(raw_path):
            if len(files) > 0 and files[0].endswith(('.jpg', '.JPG', '.jpeg')):
                # This is a disease class directory
                class_name = Path(root).name
                
                # Shuffle and split files
                random.shuffle(files)
                n = len(files)
                
                train_end = int(n * self.SPLITS['train'])
                val_end = train_end + int(n * self.SPLITS['val'])
                
                splits_data = {
                    'train': files[:train_end],
                    'val': files[train_end:val_end],
                    'test': files[val_end:]
                }
                
                # Copy files to split directories
                for split, split_files in splits_data.items():
                    split_class_dir = processed_path / split / class_name
                    split_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file in split_files:
                        src = Path(root) / file
                        dst = split_class_dir / file
                        shutil.copy2(src, dst)
                
                class_count += 1
                total_files += n
                
                logger.info(
                    f"✓ {class_name}: {len(splits_data['train'])} train, "
                    f"{len(splits_data['val'])} val, {len(splits_data['test'])} test"
                )
        
        logger.info(f"\nDataset organization complete!")
        logger.info(f"Total classes: {class_count}")
        logger.info(f"Total images: {total_files}")
        
        # Create class mapping
        class_map = self.create_class_mapping(processed_path / 'train')
        mapping_file = self.output_dir / 'class_mapping.json'
        
        with open(mapping_file, 'w') as f:
            json.dump(class_map, f, indent=2)
        
        logger.info(f"Class mapping saved to {mapping_file}")

    def get_dataset_stats(self, processed_dir: str) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            processed_dir: Directory with processed dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train': {'total': 0, 'by_class': {}},
            'val': {'total': 0, 'by_class': {}},
            'test': {'total': 0, 'by_class': {}}
        }
        
        processed_path = Path(processed_dir)
        
        for split in self.SPLITS.keys():
            split_dir = processed_path / split
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    n_images = len(list(class_dir.glob('*')))
                    stats[split]['by_class'][class_dir.name] = n_images
                    stats[split]['total'] += n_images
        
        return stats

    def print_stats(self, stats: Dict) -> None:
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        for split, data in stats.items():
            print(f"\n{split.upper()} Set: {data['total']} images")
            print("-" * 40)
            
            if data['by_class']:
                for class_name, count in sorted(data['by_class'].items()):
                    print(f"  {class_name}: {count}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main function for dataset organization."""
    parser = argparse.ArgumentParser(description='Organize PlantVillage dataset')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Path to raw dataset')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Path to save processed dataset')
    parser.add_argument('--stats_only', action='store_true',
                        help='Only print statistics')
    
    args = parser.parse_args()
    
    loader = PlantVillageDatasetLoader('.')
    
    if not args.stats_only:
        loader.organize_dataset(args.raw_dir, args.processed_dir)
    
    stats = loader.get_dataset_stats(args.processed_dir)
    loader.print_stats(stats)


if __name__ == '__main__':
    main()
