# evaluation.py
# Combined module for model evaluation and visualization

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
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

def plot_roc_curves(y_probas, model_names, y_test):
    from sklearn.metrics import roc_curve, auc
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
    def __init__(self):
        self.results = []

    def evaluate_model(self, model, X_test, y_test, model_name):
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