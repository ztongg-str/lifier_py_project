# visualization.py
# OOP module for data analysis and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from modules.config import IMG_PATH, SAVE_FIG
import os

class BaseVisualizer(ABC):
    """Abstract Base Class for visualizers"""
    @abstractmethod
    def plot(self, data, **kwargs):
        raise NotImplementedError()

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        """Save figure to configured image path"""
        if SAVE_FIG:
            path = os.path.join(IMG_PATH, f"{fig_id}.{fig_extension}")
            if tight_layout:
                plt.tight_layout()
            plt.savefig(path, format=fig_extension, dpi=resolution)
            print(f"Saved figure: {path}")

class DataAnalysis:
    """Class for analyzing data characteristics"""

    def __init__(self, data):
        self.data = data
        self.missing_analysis = None
        self.correlation_matrix = None

    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        missing_df = pd.DataFrame({
            'column': self.data.columns,
            'missing_count': self.data.isna().sum(),
            'missing_percentage': (self.data.isna().sum() * 100 / len(self.data)).round(2)
        })
        self.missing_analysis = missing_df[missing_df["missing_count"] > 0].sort_values('missing_count', ascending=False)
        return self.missing_analysis

    def analyze_correlations(self, target_col=None):
        """Analyze correlations between numerical features"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        self.correlation_matrix = numeric_data.corr()

        if target_col and target_col in self.correlation_matrix.columns:
            return self.correlation_matrix[[target_col]].sort_values(target_col)
        return self.correlation_matrix

class DataVisualizer(BaseVisualizer):
    """Class for creating various data visualizations"""

    def __init__(self):
        sns.set_theme(style="darkgrid")

    def plot_missing_values_heatmap(self, data, **kwargs):
        """Plot heatmap of missing values"""
        if 'missing_analysis' not in kwargs:
            analysis = DataAnalysis(data)
            missing_data = analysis.analyze_missing_values()
        else:
            missing_data = kwargs['missing_analysis']

        if not missing_data.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(data[missing_data['column']].isna(), cbar=False)
            plt.title("Missing Values Heatmap")
            plt.xlabel("Features")
            plt.ylabel("Samples")
            self.save_fig("missing_values_heatmap")
            plt.show()
        else:
            print("No missing values found in the dataset.")

    def plot_target_distribution(self, data, target_col, **kwargs):
        """Plot distribution of target variable"""
        plt.figure(figsize=(10, 6))
        target_counts = data[target_col].value_counts().sort_index()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        self.save_fig("target_distribution")
        plt.show()

    def plot_numerical_distributions(self, data, num_cols=None, **kwargs):
        """Plot distributions of numerical features"""
        if num_cols is None:
            num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = len(num_cols)
        n_rows = (n_cols + 2) // 3  # 3 plots per row

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(num_cols):
            row = i // 3
            col_pos = i % 3

            if row < axes.shape[0] and col_pos < axes.shape[1]:
                ax = axes[row, col_pos]
                sns.histplot(data[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

        # Hide empty subplots
        for i in range(n_cols, n_rows * 3):
            row = i // 3
            col_pos = i % 3
            if row < axes.shape[0] and col_pos < axes.shape[1]:
                axes[row, col_pos].set_visible(False)

        plt.tight_layout()
        self.save_fig("numerical_distributions")
        plt.show()

    def plot_correlation_heatmap(self, correlation_matrix, **kwargs):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix,
                    cmap="mako",
                    mask=mask,
                    annot=True,
                    annot_kws={"size": 8},
                    fmt=".2f",
                    linewidth=0.5)
        plt.title("Feature Correlation Heatmap")
        self.save_fig("correlation_heatmap")
        plt.show()

    def plot(self, data=None, plot_type="missing_heatmap", **kwargs):
        """Main plotting method"""
        if plot_type == "missing_heatmap":
            if data is None:
                raise ValueError("data is required for missing_heatmap plot")
            self.plot_missing_values_heatmap(data, **kwargs)
        elif plot_type == "target_distribution":
            if data is None:
                raise ValueError("data is required for target_distribution plot")
            target_col = kwargs.pop('target_col', 'triage_acuity')
            self.plot_target_distribution(data, target_col, **kwargs)
        elif plot_type == "numerical_distributions":
            if data is None:
                raise ValueError("data is required for numerical_distributions plot")
            num_cols = kwargs.pop('num_cols', None)
            self.plot_numerical_distributions(data, num_cols, **kwargs)
        elif plot_type == "correlation_heatmap":
            corr_matrix = kwargs.pop('correlation_matrix', None)
            if corr_matrix is not None:
                self.plot_correlation_heatmap(corr_matrix, **kwargs)
            else:
                raise ValueError("correlation_matrix is required for correlation_heatmap plot")
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
