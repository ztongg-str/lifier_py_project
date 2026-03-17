# evaluator.py
# Module for model evaluation

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from visualizer import plot_confusion_matrix_percentages, plot_roc_curves

class ModelEvaluator:
    def evaluate_models(self, models, X_test, y_test):
        results = []
        y_probas = []
        model_names = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                y_probas.append(y_proba)
                model_names.append(name)
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

            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': auc
            })

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix_percentages(cm, name)

            # Classification report
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, y_pred))

            # CV scores
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_weighted')
            print(f"5-fold CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # ROC curves
        plot_roc_curves(y_test, y_probas, model_names)

        return pd.DataFrame(results)