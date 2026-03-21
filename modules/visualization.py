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
    def __init__(self,dataframe):
        self.df = dataframe
        sns.set_theme(style="darkgrid")

    @abstractmethod
    def plot(self, data):
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
    """Main data visualization class for various plots"""
    """Class for creating various data visualizations"""
    def __init__(self, dataframe=None):
        super().__init__(dataframe)
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
class PatientHistoryDataVisualizer(BaseVisualizer):
    """Visualizer for patient history data analysis"""
    def analyze(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        condition_counts = self.df[condition_columns].sum()
        print("CONDITION ANALYSIS")
        print(f"\nMost common condition:  {condition_counts.idxmax()}")
        print(f"Least common condition: {condition_counts.idxmin()}")
        print(f"\nTop 5 conditions:")
        print(condition_counts.sort_values(ascending=False).head())
    
    def plot(self, data=None, plot_type="all"):
        """Implement the abstract plot method"""
        if data is not None:
            # If data is provided, use it (for consistency with parent)
            self.df = data
            
        if plot_type == "condition_counts":
            self.plot_bar()
        elif plot_type == "condition_percentages":
            self.plot_bar_percentage()
        elif plot_type == "condition_correlations":
            self.plot_heatmap()
        elif plot_type == "conditions_per_patients":
            self.plot_line()
        elif plot_type == "distribution_medical_condition":
            self.plot_histogram()
        elif plot_type == "all":
            self.visualize()  # Show all plots
        else:
            raise ValueError(f"Unknown plot type: {plot_type}. Available: bar, percentage, heatmap, line, histogram, all")
    def visualize(self):
        self.plot_bar()
        self.plot_bar_percentage()
        self.plot_heatmap()
        self.plot_line()
        self.plot_histogram()

    def plot_bar(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
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
        plt.show()

    def plot_bar_percentage(self):
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
        plt.show()

    def plot_heatmap(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        plt.figure(figsize=(16, 12))
        sns.heatmap(self.df[condition_columns].corr(), 
                    annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Between Medical Conditions')
        plt.show()

    def plot_line(self):
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
        plt.show()

    def plot_histogram(self):
        condition_columns = [col for col in self.df.columns 
                            if col != 'patient_id']
        self.df[condition_columns].hist(bins=50, figsize=(12, 8))
        plt.suptitle('Distribution of All Medical Conditions')
        plt.show()
class TextDataVisualizer(BaseVisualizer):
    """Visualizer for text data analysis"""
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