# evaluation.py
# Combined module for model evaluation and visualization

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from modules.config import IMG_PATH
class BaseVisualizer(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError()

class ModelVisualizer(BaseVisualizer):
    def plot(self, *args, **kwargs):
        # Placeholder
        pass

def plot_confusion_matrix_percentages(cm, model_name):
    cm_percentage = cm.astype('float') / cm.sum() * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'Confusion Matrix (%) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{IMG_PATH}/confusion_matrix_{model_name}.png")
    plt.show()


class ModelEvaluator:
    """Evaluates machine learning models with various metrics"""
    def __init__(self):
        self.results = []

    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
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
        results = []
        y_probas = []
        model_names = []
        for name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            results.append(result)
            if hasattr(model, 'predict_proba'):
                y_probas.append(model.predict_proba(X_test))
                model_names.append(name)


        return pd.DataFrame(results)

class ModelExporter:
    """Exports trained models to disk"""
    """Object-oriented model exporter to rank and save models."""
    def __init__(self, output_dir=None):
        import os
        self._output_dir = output_dir or os.path.join("output")
        self._ensure_dir()

    def _ensure_dir(self):
        import os
        try:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
        except OSError as e:
            print(f"Error creating output directory: {e}")

    def export_best_model(self, models_dict, results_df, metric='F1-Score'):
        import pickle
        import os
        try:
            if results_df.empty:
                raise ValueError("Results dataframe is empty.")
            # Rank models by metric from best to worst
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