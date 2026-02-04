"""
Single image inference script.

Classify a single plant leaf image.
"""

import torch
import argparse
from pathlib import Path
import logging
import json

from src.data.preprocessor import ImagePreprocessor
from src.models.transfer_learning import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_class_mapping(mapping_file: str) -> dict:
    """Load class name mapping."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def predict_single_image(
    image_path: str,
    model_path: str,
    model_name: str = 'resnet50',
    class_mapping_file: str = 'class_mapping.json',
    device: str = 'cuda',
    num_classes: int = 38
) -> dict:
    """
    Predict disease for a single image.
    
    Args:
        image_path: Path to leaf image
        model_path: Path to trained model
        model_name: Name of model architecture
        class_mapping_file: Path to class mapping JSON
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        Dictionary with prediction results
    """
    # Setup
    device = device if torch.cuda.is_available() else 'cpu'
    
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
    
    # Preprocess image
    image = ImagePreprocessor.preprocess(image_path, normalize=True)
    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
    image_tensor = image_tensor.to(device)
    
    logger.info(f"Image preprocessed: {image_path}")
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Load class mapping
    class_map = load_class_mapping(class_mapping_file)
    disease_name = class_map.get(str(predicted_class), f"Unknown ({predicted_class})")
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities[0], k=5)
    top5_predictions = []
    for prob, idx in zip(top5_probs, top5_indices):
        class_name = class_map.get(str(idx.item()), f"Unknown ({idx.item()})")
        top5_predictions.append({
            'disease': class_name,
            'confidence': prob.item()
        })
    
    result = {
        'image_path': str(image_path),
        'predicted_disease': disease_name,
        'predicted_class_id': predicted_class,
        'confidence': float(confidence),
        'top5_predictions': top5_predictions
    }
    
    return result


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Predict disease for single image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to leaf image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='Model name')
    parser.add_argument('--class_mapping', type=str, default='class_mapping.json',
                        help='Path to class mapping JSON')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_single_image(
        args.image_path,
        args.model_path,
        args.model_name,
        args.class_mapping,
        args.device
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Predicted Disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 5 Predictions:")
    print("-"*60)
    
    for i, pred in enumerate(result['top5_predictions'], 1):
        print(f"{i}. {pred['disease']:.<45} {pred['confidence']:.2%}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
