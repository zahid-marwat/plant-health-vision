"""
Generate comprehensive evaluation reports.

Create detailed evaluation reports with metrics and visualizations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports."""

    def __init__(self, output_dir: str = 'results'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(
        self,
        metrics: Dict[str, float],
        per_class_metrics: Dict[int, Dict[str, float]],
        class_names: List[str] = None,
        model_name: str = 'model'
    ) -> str:
        """
        Generate text-based evaluation report.
        
        Args:
            metrics: Overall metrics dictionary
            per_class_metrics: Per-class metrics dictionary
            class_names: Optional list of class names
            model_name: Name of the model
            
        Returns:
            Path to report file
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"PLANT DISEASE DETECTION - EVALUATION REPORT")
        report_lines.append(f"Model: {model_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("OVERALL METRICS")
        report_lines.append("-" * 80)
        for metric, value in metrics.items():
            report_lines.append(f"{metric:.<40} {value:.4f}")
        report_lines.append("")
        
        # Per-class metrics
        report_lines.append("PER-CLASS METRICS")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Class ID':<8} {'Class Name':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        report_lines.append("-" * 80)
        
        for class_id, class_metrics in sorted(per_class_metrics.items()):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
            report_lines.append(
                f"{class_id:<8} {class_name:<30} {class_metrics['precision']:<12.4f} "
                f"{class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f}"
            )
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f'{model_name}_evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Text report saved to {report_path}")
        return str(report_path)

    def generate_csv_report(
        self,
        per_class_metrics: Dict[int, Dict[str, float]],
        class_names: List[str] = None,
        model_name: str = 'model'
    ) -> str:
        """
        Generate CSV report with per-class metrics.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            class_names: Optional list of class names
            model_name: Name of the model
            
        Returns:
            Path to CSV file
        """
        data = []
        
        for class_id, metrics in per_class_metrics.items():
            row = {
                'class_id': class_id,
                'class_name': class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}",
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'f1_score': metrics.get('f1', None),
                'support': metrics.get('support', None)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save CSV
        csv_path = self.output_dir / f'{model_name}_metrics.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV report saved to {csv_path}")
        return str(csv_path)

    def generate_json_report(
        self,
        metrics: Dict[str, float],
        per_class_metrics: Dict[int, Dict[str, float]],
        model_name: str = 'model'
    ) -> str:
        """
        Generate JSON report.
        
        Args:
            metrics: Overall metrics dictionary
            per_class_metrics: Per-class metrics dictionary
            model_name: Name of the model
            
        Returns:
            Path to JSON file
        """
        def to_builtin(value):
            if isinstance(value, dict):
                return {k: to_builtin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_builtin(v) for v in value]
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    pass
            return value

        report = {
            'model_name': model_name,
            'overall_metrics': to_builtin(metrics),
            'per_class_metrics': {str(k): to_builtin(v) for k, v in per_class_metrics.items()}
        }
        
        json_path = self.output_dir / f'{model_name}_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"JSON report saved to {json_path}")
        return str(json_path)

    def generate_html_report(
        self,
        metrics: Dict[str, float],
        per_class_metrics: Dict[int, Dict[str, float]],
        class_names: List[str] = None,
        model_name: str = 'model',
        visualizations: Dict[str, str] = None
    ) -> str:
        """
        Generate HTML report with visualizations.
        
        Args:
            metrics: Overall metrics dictionary
            per_class_metrics: Per-class metrics dictionary
            class_names: Optional list of class names
            model_name: Name of the model
            visualizations: Dictionary with paths to visualization images
            
        Returns:
            Path to HTML file
        """
        html_content = []
        
        html_content.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Plant Disease Detection - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; font-weight: bold; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .metric-box {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #27ae60; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 3px; }}
                .footer {{ text-align: center; color: #7f8c8d; margin-top: 40px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Plant Disease Detection - Evaluation Report</h1>
                <p><strong>Model:</strong> {model_name}</p>
        """)
        
        # Overall metrics section
        html_content.append("<h2>Overall Metrics</h2>")
        for metric, value in metrics.items():
            html_content.append(f'<div class="metric-box"><strong>{metric}:</strong> <span class="metric-value">{value:.4f}</span></div>')
        
        # Per-class metrics section
        html_content.append("<h2>Per-Class Metrics</h2>")
        html_content.append('<table>')
        html_content.append('<tr><th>Class ID</th><th>Class Name</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>')
        
        for class_id, class_metrics in sorted(per_class_metrics.items()):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
            html_content.append(
                f'<tr><td>{class_id}</td><td>{class_name}</td>'
                f'<td>{class_metrics["precision"]:.4f}</td>'
                f'<td>{class_metrics["recall"]:.4f}</td>'
                f'<td>{class_metrics["f1"]:.4f}</td>'
                f'<td>{class_metrics.get("support", "-")}</td></tr>'
            )
        
        html_content.append('</table>')
        
        # Visualizations section
        if visualizations:
            html_content.append("<h2>Visualizations</h2>")
            for viz_name, viz_path in visualizations.items():
                if Path(viz_path).exists():
                    html_content.append(f"<h3>{viz_name}</h3>")
                    html_content.append(f'<img src="{viz_path}" alt="{viz_name}">')
        
        html_content.append("""
                <div class="footer">
                    <p>Generated by Plant Disease Detection System</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        html_text = "\n".join(html_content)
        
        # Save HTML
        html_path = self.output_dir / f'{model_name}_report.html'
        with open(html_path, 'w') as f:
            f.write(html_text)
        
        logger.info(f"HTML report saved to {html_path}")
        return str(html_path)
