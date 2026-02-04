"""
FastAPI web application for plant disease detection inference.

REST API for real-time disease prediction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from pathlib import Path
import json
import tempfile
import logging
from typing import Optional

from src.data.preprocessor import ImagePreprocessor
from src.models.transfer_learning import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using deep learning",
    version="1.0.0"
)

# Global variables
model = None
device = None
class_map = None
model_name = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, device, class_map, model_name
    
    try:
        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load model
        model_path = 'models/resnet50_best.pth'
        model_name = 'resnet50'
        num_classes = 38
        
        if Path(model_path).exists():
            model = create_model(model_name, num_classes)
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()
            logger.info(f"Model loaded: {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}")
        
        # Load class mapping
        if Path('class_mapping.json').exists():
            with open('class_mapping.json', 'r') as f:
                class_map = json.load(f)
            logger.info(f"Loaded {len(class_map)} classes")
        else:
            logger.warning("Class mapping not found")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from uploaded image.
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        JSON with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if class_map is None:
        raise HTTPException(status_code=503, detail="Class mapping not loaded")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Preprocess image
        image = ImagePreprocessor.preprocess(tmp_path, normalize=True)
        image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get disease name
        disease_name = class_map.get(str(predicted_class), f"Unknown ({predicted_class})")
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities[0], k=5)
        top5_predictions = []
        for prob, idx in zip(top5_probs, top5_indices):
            class_name = class_map.get(str(idx.item()), f"Unknown ({idx.item()})")
            top5_predictions.append({
                "disease": class_name,
                "confidence": float(prob.item())
            })
        
        # Clean up
        Path(tmp_path).unlink()
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "disease": disease_name,
                "class_id": predicted_class,
                "confidence": float(confidence),
                "top5_predictions": top5_predictions
            }
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get list of all disease classes."""
    if class_map is None:
        raise HTTPException(status_code=503, detail="Class mapping not loaded")
    
    return JSONResponse({
        "total_classes": len(class_map),
        "classes": class_map
    })


if __name__ == '__main__':
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
