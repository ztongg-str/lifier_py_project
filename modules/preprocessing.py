# preprocessing.py
# Combined module for data preprocessing, feature building, and related utilities
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from modules.config import *

class BasePreprocessor(ABC):
    """Abstract Base Class for OOP preprocessors with common interface."""

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Fit the preprocessor to the data.

        Args:
            df (pd.DataFrame): Input dataframe to fit on.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        """Transform the data using fitted parameters.

        Args:
            df (pd.DataFrame): Input dataframe to transform.

        Returns:
            Transformed data (type depends on implementation).
        """
        raise NotImplementedError()

    def __str__(self):
        """String representation of the preprocessor."""
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        """Detailed string representation of the preprocessor."""
        return f"{self.__class__.__name__}()"


class NumericalPreprocessor(BasePreprocessor):
    """Preprocessor for numerical features with outlier clipping and scaling."""

    def __init__(self):
        """Initialize the numerical preprocessor."""
        self._num_cols = None
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, df: pd.DataFrame):
        """Fit the numerical preprocessor.

        Args:
            df (pd.DataFrame): Input dataframe with numerical columns.
        """
        try:
            # Exclude post-triage/leakage fields
            leak_cols = {'ed_los_hours', 'triage_acuity'}
            self._num_cols = [c for c in df.select_dtypes(include='number').columns if c not in leak_cols]
            num_data = df[self._num_cols]
            self.lower_bounds = num_data.quantile(0.05)
            self.upper_bounds = num_data.quantile(0.95)
            clipped = num_data.clip(self.lower_bounds, self.upper_bounds, axis=1)
            imputed = self._imputer.fit_transform(clipped)
            self._scaler.fit(imputed)
        except KeyError as e:
            raise KeyError(f"Missing numerical columns in data: {e}")

    def transform(self, df: pd.DataFrame):
        """Transform numerical data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: Scaled and imputed numerical features.
        """
        try:
            num_data = df.reindex(columns=self._num_cols).copy()
            missing_cols = [c for c in self._num_cols if c not in num_data.columns]
            for c in missing_cols:
                num_data[c] = 0
            clipped = num_data.clip(self.lower_bounds, self.upper_bounds, axis=1)
            imputed = self._imputer.transform(clipped)
            scaled = self._scaler.transform(imputed)
            return scaled
        except Exception as e:
            raise RuntimeError(f"Error during numerical transformation: {e}")
        
class CategoricalPreprocessor(BasePreprocessor):
    """Preprocessor for categorical features using one-hot encoding."""

    def __init__(self, max_categories=50, exclude_patterns=None):
        """Initialize the categorical preprocessor."""
        self.max_categories = max_categories
        self.exclude_patterns = exclude_patterns or ['*_id', '*_raw']
        self._cat_cols = None
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._imputer = SimpleImputer(strategy='constant', fill_value='unknown')

    def _should_encode_column(self, col_name, unique_count):
        """Determine if a column should be one-hot encoded.

        Args:
            col_name: Column name.
            unique_count: Number of unique values.

        Returns:
            bool: Whether to encode the column.
        """
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.endswith('*'):
                if col_name.startswith(pattern[:-1]):
                    return False
            elif pattern.startswith('*'):
                if col_name.endswith(pattern[1:]):
                    return False
            elif col_name == pattern:
                return False

        # Remove known post-triage targets/leakage columns
        if col_name in {'disposition'}:
            return False
        return unique_count <= self.max_categories

    def fit(self, df: pd.DataFrame):
        """Fit the categorical preprocessor.

        Args:
            df (pd.DataFrame): Input dataframe with categorical columns.
        """
        print("CategoricalPreprocessor.fit() START")
        try:
            all_cat_cols = df.select_dtypes(include='object').columns
            self._cat_cols = []

            print(f"CategoricalPreprocessor: Found {len(all_cat_cols)} categorical columns: {list(all_cat_cols)}")

            for col in all_cat_cols:
                unique_count = df[col].nunique()
                if self._should_encode_column(col, unique_count):
                    self._cat_cols.append(col)

            print(f"CategoricalPreprocessor: selected {len(self._cat_cols)} categorical columns")

            if not self._cat_cols:
                print("Warning: No categorical columns selected for encoding")
                return

            cat_data = df[self._cat_cols]
            self._imputer.fit(cat_data)
            imputed = self._imputer.transform(cat_data)
            self._encoder.fit(imputed)
            print(f"CategoricalPreprocessor: Successfully fitted on {len(self._cat_cols)} columns")
        except KeyError as e:
            raise KeyError(f"Missing categorical columns in data: {e}")
        print("CategoricalPreprocessor.fit() END")

    def transform(self, df: pd.DataFrame):
        """Transform categorical data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: One-hot encoded categorical features.
        """
        try:
            if not self._cat_cols:
                return np.empty((df.shape[0], 0))

            cat_data = df.reindex(columns=self._cat_cols).copy()
            cat_data.fillna('unknown', inplace=True)
            # In case columns are missing in new data, fill with unknown before encoding
            for c in self._cat_cols:
                if c not in cat_data.columns:
                    cat_data[c] = 'unknown'

            imputed = self._imputer.transform(cat_data)
            encoded = self._encoder.transform(imputed)
            return encoded
        except Exception as e:
            raise RuntimeError(f"Error during categorical transformation: {e}")

class TextPreprocessor(BasePreprocessor):
    """Preprocessor for text features using TF-IDF vectorization."""

    def __init__(self, text_col='chief_complaint_raw', max_features=500):
        """Initialize the text preprocessor.

        Args:
            text_col (str): Name of the text column.
            max_features (int): Maximum number of features for TF-IDF.
        """
        self._text_col = text_col
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features
        )

    def _clean_text(self, series: pd.Series):
        """Clean text data by lowercasing and removing non-alphanumeric characters.

        Args:
            series (pd.Series): Text series to clean.

        Returns:
            pd.Series: Cleaned text series.
        """
        return series.astype(str).str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.strip()

    def fit(self, df: pd.DataFrame):
        """Fit the text preprocessor.

        Args:
            df (pd.DataFrame): Input dataframe with text column.
        """
        try:
            texts = self._clean_text(df[self._text_col].fillna(''))
            self._tfidf.fit(texts.tolist())
        except KeyError:
            raise KeyError(f"Missing text column {self._text_col} in fit")

    def transform(self, df: pd.DataFrame):
        """Transform text data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: TF-IDF transformed text features.
        """
        try:
            texts = self._clean_text(df[self._text_col].fillna(''))
            return self._tfidf.transform(texts.tolist()).toarray()
        except Exception as e:
            raise RuntimeError(f"Error during text transformation: {e}")


class ClassImbalanceHandler:
    """Handles class imbalance in the target variable using SMOTE."""

    def __init__(self, k_neighbors=None, random_state=42):
        """Initialize the imbalance handler.

        Args:
            k_neighbors (int, optional): Number of neighbors for SMOTE.
            random_state (int): Random state for reproducibility.
        """
        self.k_neighbors = k_neighbors or SMOTE_K_NEIGHBORS
        self.random_state = random_state
        self._smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)

    def fit_resample(self, X, y):
        """Balance the dataset using SMOTE.

        Args:
            X: Feature array.
            y: Target array.

        Returns:
            tuple: Resampled X and y arrays.
        """
        try:
            X_resampled, y_resampled = self._smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            raise RuntimeError(f"Error during SMOTE resampling: {e}")


class FeatureBuilder:
    """Combines multiple preprocessors to build feature matrices."""

    def __init__(self, preprocessors):
        """Initialize with list of preprocessors.

        Args:
            preprocessors (list): List of BasePreprocessor instances.
        """
        self._preprocessors = preprocessors

    def fit(self, df):
        """Fit all preprocessors.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        print(f"FeatureBuilder.fit() called with {len(self._preprocessors)} preprocessors")
        for i, p in enumerate(self._preprocessors):
            print(f"Fitting preprocessor {i}: {type(p).__name__}")
            p.fit(df)
            print(f"Preprocessor {i} fit completed")
        print("FeatureBuilder.fit() completed")

    def transform(self, df):
        """Transform data using all preprocessors and concatenate features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: Concatenated feature matrix.
        """
        features = [p.transform(df) for p in self._preprocessors]
        return np.hstack(features)