"""
Evaluation Metrics and Visualization Module

This module provides comprehensive evaluation metrics and visualization tools
for emotion detection models including accuracy, precision, recall, F1-score,
and confusion matrix visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import json

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Features:
    - Calculate accuracy, precision, recall, F1-score
    - Generate confusion matrices
    - Create performance comparison plots
    - Export evaluation results
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            class_names: List of emotion class names
        """
        self.class_names = class_names or ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
        self.results = {}
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("Initialized ModelEvaluator")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'{class_name}_precision'] = precision_per_class[i]
                metrics[f'{class_name}_recall'] = recall_per_class[i]
                metrics[f'{class_name}_f1'] = f1_per_class[i]
        
        self.results[model_name] = metrics
        
        logger.info(f"Calculated metrics for {model_name}")
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model", save_path: str = None):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path: str = None):
        """
        Plot comparison of metrics across different models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.results[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Score', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, model_name: str, save_path: str = None):
        """
        Plot per-class metrics for a specific model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.results:
            logger.error(f"Model {model_name} not found in results")
            return
        
        # Prepare data
        data = []
        for class_name in self.class_names:
            precision_key = f'{class_name}_precision'
            recall_key = f'{class_name}_recall'
            f1_key = f'{class_name}_f1'
            
            if all(key in self.results[model_name] for key in [precision_key, recall_key, f1_key]):
                data.append({
                    'Emotion': class_name,
                    'Precision': self.results[model_name][precision_key],
                    'Recall': self.results[model_name][recall_key],
                    'F1-Score': self.results[model_name][f1_key]
                })
        
        if not data:
            logger.warning("No per-class metrics found")
            return
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Emotion'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       model_name: str = "Model") -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=False)
        
        logger.info(f"Generated classification report for {model_name}")
        return report
    
    def compare_with_without_augmentation(self, results_with_aug: Dict, 
                                        results_without_aug: Dict,
                                        save_path: str = None):
        """
        Compare model performance with and without data augmentation.
        
        Args:
            results_with_aug: Results with augmentation
            results_without_aug: Results without augmentation
            save_path: Path to save the comparison plot
        """
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Prepare data
        data = []
        for metric in metrics:
            data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Without Augmentation': results_without_aug.get(metric, 0),
                'With Augmentation': results_with_aug.get(metric, 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Without Augmentation'], width, 
               label='Without Augmentation', alpha=0.8)
        ax.bar(x + width/2, df['With Augmentation'], width, 
               label='With Augmentation', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: With vs Without Data Augmentation')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Metric'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (without, with_aug) in enumerate(zip(df['Without Augmentation'], 
                                                   df['With Augmentation'])):
            ax.text(i - width/2, without + 0.01, f'{without:.3f}', 
                   ha='center', va='bottom')
            ax.text(i + width/2, with_aug + 0.01, f'{with_aug:.3f}', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Augmentation comparison saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, output_path: str):
        """
        Export all evaluation results to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results exported to: {output_path}")
    
    def print_summary(self):
        """Print a summary of all evaluation results."""
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_macro']:.4f}")
            print(f"  Recall:    {metrics['recall_macro']:.4f}")
            print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
        
        print("="*60)

def main():
    """Example usage of the model evaluator."""
    evaluator = ModelEvaluator()
    
    # Example usage:
    # y_true = np.array([0, 1, 2, 0, 1, 2])
    # y_pred = np.array([0, 1, 1, 0, 1, 2])
    # 
    # # Calculate metrics
    # metrics = evaluator.calculate_metrics(y_true, y_pred, "SVM")
    # 
    # # Generate plots
    # evaluator.plot_confusion_matrix(y_true, y_pred, "SVM")
    # evaluator.plot_per_class_metrics("SVM")
    # 
    # # Print summary
    # evaluator.print_summary()
    
    print("Model Evaluator ready!")

if __name__ == "__main__":
    main()
