"""
Run evaluation for a trained model and generate reports/visualizations.
"""

import argparse
import json
from pathlib import Path
import logging
import torch

from src.data.data_generator import PyTorchDataLoader
from src.data.augmentation import PlantDiseaseAugmentor
from src.models.base_cnn import create_baseline_cnn
from src.models.transfer_learning import create_model as create_transfer_model
from src.models.yolo_models import create_yolov8_classify
from src.models.custom_models import create_custom_resnet
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.visualizer import EvaluationVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(model_name: str, num_classes: int):
    if model_name == "baseline_cnn":
        return create_baseline_cnn(num_classes)
    if model_name in {"resnet50", "mobilenetv2", "efficientnetb0", "vgg16"}:
        return create_transfer_model(model_name, num_classes, pretrained=False, freeze_backbone=False)
    if model_name == "yolov8":
        return create_yolov8_classify(num_classes)
    if model_name == "custom_resnet":
        return create_custom_resnet(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model")
    parser.add_argument("--model_name", type=str, default="baseline_cnn", help="Model name")
    parser.add_argument("--model_path", type=str, default="models/baseline_cnn_final.pth", help="Path to model weights")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for reports")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--history_path", type=str, default="models/baseline_cnn_history.json", help="Training history JSON")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info("Using device: %s", device)

    # Data loaders
    transforms_val = PlantDiseaseAugmentor.get_val_transforms((args.img_size, args.img_size))
    data_loader = PyTorchDataLoader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        num_workers=4,
        transform_train=None,
        transform_val=transforms_val,
    )

    test_loader = data_loader.get_test_loader()
    num_classes = data_loader.num_classes
    idx_to_class = data_loader.train_dataset.idx_to_class
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Model
    model = create_model(args.model_name, num_classes)
    model_path = Path(args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluation
    evaluator = ModelEvaluator(model, test_loader, device=device, num_classes=num_classes)
    predictions, true_labels, probs = evaluator.get_predictions()
    metrics = evaluator.calculate_metrics()
    per_class_metrics = evaluator.calculate_per_class_metrics()

    # Reports and visualizations
    report_gen = ReportGenerator(args.output_dir)
    viz = EvaluationVisualizer(args.output_dir)

    report_gen.generate_text_report(metrics, per_class_metrics, class_names, args.model_name)
    report_gen.generate_csv_report(per_class_metrics, class_names, args.model_name)
    report_gen.generate_json_report(metrics, per_class_metrics, args.model_name)

    viz_paths = {
        "Confusion Matrix": viz.plot_confusion_matrix(true_labels, predictions, class_names),
        "ROC Curves": viz.plot_roc_curves(true_labels, probs, class_names, top_n=10),
        "Per-Class Metrics": viz.plot_per_class_metrics(per_class_metrics, class_names, top_n=15),
    }

    # Training history plot (if available)
    history_path = Path(args.history_path)
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        viz_paths["Training History"] = viz.plot_training_history(history)

    report_gen.generate_html_report(metrics, per_class_metrics, class_names, args.model_name, viz_paths)

    logger.info("Evaluation complete. Reports saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
