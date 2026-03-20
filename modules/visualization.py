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
    """Abstract Base Class for visualizers with common functionality."""
    def __init__(self, data=None):
        self.df = data
        sns.set_theme(style="darkgrid")

    @abstractmethod
    def plot(self, data=None, plot_type=None, **kwargs):
        """Abstract method for plotting. Must be implemented by subclasses."""
        pass

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        """Save figure to configured image path.

        Args:
            fig_id (str): Identifier for the figure filename.
            tight_layout (bool): Whether to apply tight layout. Default True.
            fig_extension (str): File extension for the image. Default 'png'.
            resolution (int): DPI resolution for the saved image. Default 300.
        """
        if SAVE_FIG:
            path = os.path.join(IMG_PATH, f"{fig_id}.{fig_extension}")
            if tight_layout:
                plt.tight_layout()
            plt.savefig(path, format=fig_extension, dpi=resolution)

    def __repr__(self):
        """String representation of the visualizer."""
        return f"<{self.__class__.__name__} with {len(self.df) if self.df is not None else 0} samples>"

    def __len__(self):
        """Return the number of samples in the data."""
        return len(self.df) if self.df is not None else 0

class DataAnalysis:
    """Class for analyzing data characteristics including missing values and correlations."""

    def __init__(self, data):
        """Initialize with dataset.

        Args:
            data (pd.DataFrame): The dataset to analyze.
        """
        self.data = data
        self.missing_analysis = None
        self.correlation_matrix = None

    def analyze_missing_values(self):
        """Analyze missing values in the dataset.

        Returns:
            pd.DataFrame: DataFrame with missing value statistics.
        """
        missing_df = pd.DataFrame({
            'column': self.data.columns,
            'missing_count': self.data.isna().sum(),
            'missing_percentage': (self.data.isna().sum()*100/len(self.data)).round(2)
        })
        self.missing_analysis = missing_df[missing_df["missing_count"] > 0].sort_values('missing_count', ascending=False)
        return self.missing_analysis

    def analyze_correlations(self, target_col=None):
        """Analyze correlations between numerical features.

        Args:
            target_col (str, optional): Target column to focus correlations on.

        Returns:
            pd.DataFrame: Correlation matrix or target correlations.
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        self.correlation_matrix = numeric_data.corr()

        if target_col and target_col in self.correlation_matrix.columns:
            return self.correlation_matrix[[target_col]].sort_values(target_col)
        return self.correlation_matrix

    def __repr__(self):
        """String representation of the data analysis object."""
        return f"<DataAnalysis with {len(self.data)} samples, {len(self.data.columns)} features>"

class TrainDataVisualizer(BaseVisualizer):
    """Class for creating various data visualizations for training data."""

    def __init__(self):
        """Initialize the visualizer."""
        super().__init__()

    def plot_missing_values_heatmap(self, data, **kwargs):
        """Plot heatmap of missing values.

        Args:
            data (pd.DataFrame): Dataset to visualize.
            **kwargs: Additional arguments, including 'missing_analysis'.
        """
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
        """Plot distribution of target variable.

        Args:
            data (pd.DataFrame): Dataset containing target.
            target_col (str): Name of target column.
        """
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
        """Plot distributions of numerical features.

        Args:
            data (pd.DataFrame): Dataset to visualize.
            num_cols (list, optional): List of numerical columns to plot.
        """
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
                sns.histplot(data[col].dropna(), ax=ax)
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

    def plot_correlation_heatmap(self, correlation_matrix):
        """Plot correlation heatmap.

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix to visualize.
        """
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

    def plot(self, data=None, plot_type="missing_heatmap", target_col=None, num_cols=None, correlation_matrix=None):
        if plot_type == "missing_heatmap":
            if data is None:
                raise ValueError("data is required")
            self.plot_missing_values_heatmap(data)
        elif plot_type == "target_distribution":
            if data is None or target_col is None:
                raise ValueError("data and target_col are required")
            self.plot_target_distribution(data, target_col)
        elif plot_type == "numerical_distributions":
            if data is None:
                raise ValueError("data is required")
            self.plot_numerical_distributions(data, num_cols)
        elif plot_type == "correlation_heatmap":
            if correlation_matrix is None:
                raise ValueError("correlation_matrix is required")
            self.plot_correlation_heatmap(correlation_matrix)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
class PatientHistoryDataVisualizer(BaseVisualizer):
    """Class for visualizing patient history data (binary medical conditions)."""

    def __init__(self, data):
        """Initialize with patient history data.

        Args:
            data (pd.DataFrame): Patient history dataset with binary condition flags.
        """
        super().__init__(data)

    def analyze_conditions(self):
        """Analyze medical conditions prevalence.

        Returns:
            pd.Series: Count of patients with each condition.
        """
        condition_columns = [col for col in self.df.columns if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        print("CONDITION ANALYSIS")
        print(f"Most common condition: {condition_counts.idxmax()}")
        print(f"Least common condition: {condition_counts.idxmin()}")
        print("Top 5 conditions:")
        print(condition_counts.sort_values(ascending=False).head())
        return condition_counts

    def plot_condition_counts(self, **kwargs):
        """Plot bar chart of condition counts.

        **kwargs: Additional arguments for customization.
        """
        condition_columns = [col for col in self.df.columns if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum().sort_values(ascending=False)
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x=condition_counts.index, 
                         y=condition_counts.values, palette='Reds_r')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom', fontsize=8)
        plt.title('Number of Patients with Each Medical Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=90)
        self.save_fig("patient_history_condition_counts")
        plt.show()

    def plot_condition_percentages(self, **kwargs):
        """Plot bar chart of condition percentages.

        **kwargs: Additional arguments for customization.
        """
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_pct = (self.df[condition_columns].sum() / 
                        len(self.df)).sort_values(ascending=False)
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x=condition_pct.index, 
                         y=condition_pct.values, palette='Blues_r')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom', fontsize=8)
        plt.title('Percentage of Patients with Each Medical Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Proportion of Patients')
        plt.xticks(rotation=90)
        self.save_fig("patient_history_condition_percentages")
        plt.show()

    def plot_condition_correlations(self, **kwargs):
        """Plot correlation heatmap between conditions.

        **kwargs: Additional arguments for customization.
        """
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        plt.figure(figsize=(16, 12))
        sns.heatmap(self.df[condition_columns].corr(), 
                    annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Between Medical Conditions')
        self.save_fig("patient_history_condition_correlations")
        plt.show()

    def plot_condition_line(self, **kwargs):
        """Plot line chart of condition counts.

        **kwargs: Additional arguments for customization.
        """
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        plt.figure(figsize=(14, 6))
        plt.plot(condition_counts.index, 
                 condition_counts.values, 
                 marker='o', color='steelblue')
        plt.title('Patient Count Per Condition')
        plt.xlabel('Medical Condition')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=90)
        self.save_fig("patient_history_condition_line")
        plt.show()

    def plot_condition_histograms(self, **kwargs):
        """Plot histograms of all conditions.

        **kwargs: Additional arguments for customization.
        """
        condition_columns = [col for col in self.df.columns if col != 'patient_id']
        self.df[condition_columns].hist(bins=50, figsize=(12, 8))
        plt.suptitle('Distribution of All Medical Conditions')
        self.save_fig("patient_history_condition_histograms")
        plt.show()

    def plot(self, data=None, plot_type="condition_counts", **kwargs):
        """Main plotting method for patient history data.

        Args:
            data: Not used, uses self.df.
            plot_type (str): Type of plot ('condition_counts', 'condition_percentages',
                          'condition_correlations', 'condition_line', 'condition_histograms').
            **kwargs: Additional arguments.
        """
        if plot_type == "condition_counts":
            self.plot_condition_counts(**kwargs)
        elif plot_type == "condition_percentages":
            self.plot_condition_percentages(**kwargs)
        elif plot_type == "condition_correlations":
            self.plot_condition_correlations(**kwargs)
        elif plot_type == "condition_line":
            self.plot_condition_line(**kwargs)
        elif plot_type == "condition_histograms":
            self.plot_condition_histograms(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")


class TextDataVisualizer(BaseVisualizer):
    """Class for visualizing text data characteristics."""
    def __init__(self, data, text_col='chief_complaint_raw'):
        """Initialize with text data.
        Args:
            data (pd.DataFrame): Dataset containing text column.
            text_col (str): Name of text column.
        """
        super().__init__(data)
        self.text_col = text_col

    def analyze_text_lengths(self):
        """Analyze text length statistics.
        Returns:
            dict: Statistics about text lengths.
        """
        text_lengths = self.df[self.text_col].astype(str).str.len()
        stats = {
            'mean_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'std_length': text_lengths.std()
        }
        print("TEXT LENGTH ANALYSIS")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        return stats

    def plot_text_length_distribution(self, **kwargs):
        """Plot distribution of text lengths.

        **kwargs: Additional arguments for customization.
        """
        text_lengths = self.df[self.text_col].astype(str).str.len()
        plt.figure(figsize=(10, 6))
        sns.histplot(text_lengths, bins=50, kde=True)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        self.save_fig("text_length_distribution")
        plt.show()

    def plot_word_count_distribution(self, **kwargs):
        """Plot distribution of word counts.

        **kwargs: Additional arguments for customization.
        """
        word_counts = self.df[self.text_col].astype(str).str.split().str.len()
        plt.figure(figsize=(10, 6))
        sns.histplot(word_counts, bins=50, kde=True)
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        self.save_fig("word_count_distribution")
        plt.show()

    def plot_common_words(self, top_n=20, **kwargs):
        """Plot most common words.

        Args:
            top_n (int): Number of top words to show.
            **kwargs: Additional arguments.
        """
        from collections import Counter
        import re

        all_text = ' '.join(self.df[self.text_col].astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        word_counts = Counter(words)
        common_words = dict(word_counts.most_common(top_n))

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(common_words.keys()), y=list(common_words.values()))
        plt.title(f'Top {top_n} Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        self.save_fig("common_words")
        plt.show()

    def plot_text_length_by_category(self, category_col='chief_complaint_system', **kwargs):
        """Plot text length distribution by category.

        Args:
            category_col (str): Column to group by.
            **kwargs: Additional arguments.
        """
        if category_col in self.df.columns:
            text_lengths = self.df[self.text_col].astype(str).str.len()
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.df[category_col], y=text_lengths)
            plt.title('Text Length Distribution by Category')
            plt.xlabel('Category')
            plt.ylabel('Text Length')
            plt.xticks(rotation=45)
            self.save_fig("text_length_by_category")
            plt.show()
        else:
            print(f"Category column '{category_col}' not found.")

    def plot(self, data=None, plot_type="text_length_distribution", **kwargs):
        """Main plotting method for text data.

        Args:
            data: Not used, uses self.df.
            plot_type (str): Type of plot ('text_length_distribution', 'word_count_distribution',
                          'common_words', 'text_length_by_category').
            **kwargs: Additional arguments.
        """
        if plot_type == "text_length_distribution":
            self.plot_text_length_distribution(**kwargs)
        elif plot_type == "word_count_distribution":
            self.plot_word_count_distribution(**kwargs)
        elif plot_type == "common_words":
            self.plot_common_words(**kwargs)
        elif plot_type == "text_length_by_category":
            self.plot_text_length_by_category(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")