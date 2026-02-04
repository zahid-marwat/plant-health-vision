"""
Batch prediction script.

Classify multiple leaf images from a directory.
"""

import torch
import argparse
from pathlib import Path
import pandas as pd
import json
import logging
from tqdm import tqdm
from typing import List, Dict

from src.data.preprocessor import ImagePreprocessor
from src.models.transfer_learning import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_predict(
    image_dir: str,
    model_path: str,
    model_name: str = 'resnet50',
    class_mapping_file: str = 'class_mapping.json',
    device: str = 'cuda',
    num_classes: int = 38,
    output_csv: str = None
) -> List[Dict]:
    """
    Make predictions for multiple images.
    
    Args:
        image_dir: Directory containing images
        model_path: Path to trained model
        model_name: Name of model architecture
        class_mapping_file: Path to class mapping JSON
        device: Device to use
        num_classes: Number of classes
        output_csv: Optional CSV file to save results
        
    Returns:
        List of prediction dictionaries
    """
    # Setup
    device = device if torch.cuda.is_available() else 'cpu'
    image_dir = Path(image_dir)
    
    # Load model
    model = create_model(model_name, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model_path}")
    
    # Load class mapping
    with open(class_mapping_file, 'r') as f:
        class_map = json.load(f)
    
    # Get image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Process images
    results = []
    
    for image_path in tqdm(image_paths, desc='Predicting'):
        try:
            # Preprocess
            image = ImagePreprocessor.preprocess(str(image_path), normalize=True)
            image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
            image_tensor = image_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            disease_name = class_map.get(str(predicted_class), f"Unknown ({predicted_class})")
            
            result = {
                'image': str(image_path),
                'disease': disease_name,
                'class_id': predicted_class,
                'confidence': confidence
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Save to CSV if specified
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
    
    return results


def main():
    """Main batch prediction function."""
    parser = argparse.ArgumentParser(description='Batch predict diseases')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='Model name')
    parser.add_argument('--class_mapping', type=str, default='class_mapping.json',
                        help='Path to class mapping JSON')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Make predictions
    results = batch_predict(
        args.image_dir,
        args.model_path,
        args.model_name,
        args.class_mapping,
        args.device,
        output_csv=args.output_csv
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(results)}")
    
    # Group by disease
    disease_counts = {}
    for result in results:
        disease = result['disease']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    print("\nDisease Distribution:")
    print("-"*70)
    for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease}: {count}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
