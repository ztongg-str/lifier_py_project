# visualizer.py
# Module for visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from config import IMG_PATH, SAVE_FIG

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    if SAVE_FIG:
        path = os.path.join(IMG_PATH, f"{fig_id}.{fig_extension}")
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_boxplots(data, features, target):
    for feature in features[:5]:  # Limit to 5 for brevity
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=target, y=feature, data=data)
        plt.title(f'Boxplot of {feature} by {target}')
        save_fig(f'boxplot_{feature}')
        plt.show()

def plot_missing_correlation(data):
    missing_corr = data.isnull().corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm')
    plt.title('Missing Value Correlation')
    save_fig('missing_correlation')
    plt.show()

def plot_roc_curves(y_test, y_probas, model_names):
    plt.figure(figsize=(8, 6))
    for y_proba, name in zip(y_probas, model_names):
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        else:
            # For multiclass, plot micro-average
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    save_fig('roc_curves')
    plt.show()

def plot_confusion_matrix_percentages(cm, model_name):
    cm_percentage = cm.astype('float') / cm.sum() * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'Confusion Matrix (%) - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig(f'cm_percentage_{model_name}')
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 'o-', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    save_fig('learning_curve')
    plt.show()