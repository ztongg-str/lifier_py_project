# evaluation.py
# Combined module for model evaluation and visualization

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from modules.config import IMG_PATH
import pickle
import os
def plot_confusion_matrix_percentages(cm, model_name):
    """Plot confusion matrix as percentages.

    Args:
        cm: Confusion matrix array.
        model_name: Name of the model for plot title.
    """
    cm_percentage = cm.astype('float') / cm.sum() * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'Confusion Matrix (%) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{IMG_PATH}/confusion_matrix_{model_name}.png")
    plt.show()

def plot_roc_curves(y_probas, model_names, y_test):
    """Plot ROC curves for multiple models.

    Args:
        y_probas: List of prediction probabilities.
        model_names: List of model names.
        y_test: True labels.
    """
    plt.figure(figsize=(10, 8))
    for y_proba, name in zip(y_probas, model_names):
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        else:
            # Multi-class, plot micro-average
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)  # Adjust for multi-class
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f"{IMG_PATH}/roc_curves.png")
    plt.show()

class ModelEvaluator:
    """Evaluates machine learning models with metrics and visualizations."""

    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results = []

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model.

        Args:
            model: Trained model instance.
            X_test: Test features.
            y_test: Test labels.
            model_name: Name of the model.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            auc = None
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        result = {
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        }
        self.results.append(result)
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix_percentages(cm, model_name)
        # Classification report
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        return result

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate multiple models.

        Args:
            models: Dict of trained models.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            DataFrame of evaluation results.
        """
        results = []
        y_probas = []
        model_names = []
        for name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            results.append(result)
            if hasattr(model, 'predict_proba'):
                y_probas.append(model.predict_proba(X_test))
                model_names.append(name)

        # Plot ROC if applicable
        if y_probas:
            plot_roc_curves(y_probas, model_names, y_test)

        return pd.DataFrame(results)

class ModelExporter:
    """Object-oriented model exporter to rank and save models."""

    def __init__(self, output_dir=None):
        """Initialize ModelExporter.

        Args:
            output_dir: Directory to save models. Defaults to 'output'.
        """
        import os
        self._output_dir = output_dir or os.path.join("output")
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure output directory exists."""
        import os
        try:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
        except OSError as e:
            print(f"Error creating output directory: {e}")

    def export_best_model(self, models_dict, results_df, metric='F1-Score'):
        """Export the best performing model.

        Args:
            models_dict: Dict of trained models.
            results_df: DataFrame with evaluation results.
            metric: Metric to rank models by.

        Returns:
            Path to saved model file.
        """
        try:
            if results_df.empty:
                raise ValueError("Results dataframe is empty.")
            # Rank models by metric in descending order
            ranked_df = results_df.sort_values(by=metric, ascending=False)
            best_model_name = ranked_df.iloc[0]['Model']
            best_score = ranked_df.iloc[0][metric]
            best_model = models_dict.get(best_model_name)
            if best_model is None:
                raise ValueError(f"Best model '{best_model_name}' not found in models dictionary.")
            filepath = os.path.join(self._output_dir, "best_model_ranked.pkl")  
            with open(filepath, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Exported best model '{best_model_name}' (Score: {best_score:.4f}) to {filepath}")
            return filepath
        except Exception as e:
            raise RuntimeError(f"Failed to export best model: {e}")