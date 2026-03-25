# preprocessing.py
# Combined module for data preprocessing, feature building, and related utilities
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from modules.config import *
import json
from collections import Counter
import re
from scipy import sparse
class BasePreprocessor(ABC):
    """Abstract Base Class for OOP preprocessors"""
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError()
 
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

class BaseFeatureBuilder(ABC):
    """Abstract Base Class for OOP Feature Builders"""
    @abstractmethod
    def build(self, df: pd.DataFrame):
        raise NotImplementedError()
        
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

class NumericalPreprocessor(BasePreprocessor):
    """Handles numerical feature preprocessing: imputation, scaling, outlier clipping"""
    def __init__(self):
        self._num_cols = None
        self._scaler = StandardScaler()
        # Use pandas fillna instead of sklearn imputer to avoid feature name issues
        self._imputer = None  # Not used
        self.upper_bounds = None
        self.lower_bounds = None
        self._is_fitted = False
    def fit(self, df):
        try:
            all_num_cols = df.select_dtypes(include='number').columns
            self._num_cols = [col for col in all_num_cols if col not in LEAKAGE_COLS]
            num_data = df[self._num_cols]
            self.lower_bounds = num_data.quantile(0.05)
            self.upper_bounds = num_data.quantile(0.95)
            # Store mean values for imputation
            clipped = num_data.clip(self.lower_bounds, self.upper_bounds, axis=1)
            self._means = clipped.mean()
            self._scaler.fit(clipped.fillna(self._means))
        except KeyError as e:
            raise KeyError(f"Missing numerical columns in data: {e}")

    def transform(self, df):
        try:
            # Use the columns that were fitted, not re-select from current df
            num_data = df[self._num_cols]
            clipped = num_data.clip(self.lower_bounds, self.upper_bounds, axis=1)
            imputed = clipped.fillna(self._means)
            scaled = self._scaler.transform(imputed.values)
            return pd.DataFrame(scaled, columns=self._num_cols, index=df.index)
        except Exception as e:
            raise RuntimeError(f"Error during numerical transformation: {e}")

    def __str__(self):
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        num_cols = len(self._num_cols) if self._num_cols else 0
        return f"NumericalPreprocessor({fitted_status}, {num_cols} columns)"

    def __len__(self):
        return len(self._num_cols) if self._num_cols else 0

class CategoricalPreprocessor(BasePreprocessor):
    """Handles categorical feature preprocessing: imputation and one-hot encoding"""
    def __init__(self):
        self._cat_cols = None
        self._encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        # Use pandas fillna instead of sklearn imputer
        self._imputer = None  # Not used
        self._is_fitted = False
    def fit(self, df: pd.DataFrame):
        try:
            all_cat_cols = df.select_dtypes(include='object').columns
            # Remove data leakage cols and text cols
            exclude_cols = LEAKAGE_COLS + ['chief_complaint_raw']
            potential_cat_cols = [col for col in all_cat_cols if col not in exclude_cols]
            # Only include columns with reasonable number of unique values
            self._cat_cols = [col for col in potential_cat_cols if df[col].nunique() <= 5]
            cat_data = df[self._cat_cols]
            # Store mode values for imputation
            self._modes = cat_data.mode().iloc[0] if not cat_data.empty else pd.Series(dtype=object)
            filled_data = cat_data.fillna(self._modes)
            self._encoder.fit(filled_data.values)
        except KeyError as e:
            raise KeyError(f"Missing categorical columns in data: {e}")

    def transform(self, df: pd.DataFrame):
        try:
            # Use the columns that were fitted, not re-select from current df
            cat_data = df[self._cat_cols]
            filled_data = cat_data.fillna(self._modes)
            encoded = self._encoder.transform(filled_data.values)
            feature_names = self._encoder.get_feature_names_out(self._cat_cols)
            return pd.DataFrame(encoded, columns=feature_names, index=df.index)
        except Exception as e:
            raise RuntimeError(f"Error during categorical transformation: {e}")

    def __str__(self):
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        cat_cols = len(self._cat_cols) if self._cat_cols else 0
        return f"CategoricalPreprocessor({fitted_status}, {cat_cols} columns)"

    def __len__(self):
        return len(self._cat_cols) if self._cat_cols else 0

class TextPreprocessor(BasePreprocessor):
    """Handles text feature preprocessing: cleaning and TF-IDF vectorization"""
    def __init__(self, text_col='chief_complaint_raw', max_features=TFIDF_MAX_FEATURES):
        self._text_col = text_col
        self._is_fitted = False
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features
        )

    def _clean_text(self, series: pd.Series):
        #(lowercase + clean text case-insensitive)
        return series.astype(str).str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.strip()

    def fit(self, df: pd.DataFrame):
        try:
            texts = self._clean_text(df[self._text_col].fillna(''))
            self._tfidf.fit(texts)
        except KeyError:
            raise KeyError(f"Missing text column {self._text_col} in fit")

    def transform(self, df: pd.DataFrame):
        try:
            texts = self._clean_text(df[self._text_col].fillna(''))
            return self._tfidf.transform(texts)
        except Exception as e:
            raise RuntimeError(f"Error during text transformation: {e}")

    def __str__(self):
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        vocab_size = len(self._tfidf.vocabulary_) if hasattr(self._tfidf, 'vocabulary_') else 0
        return f"TextPreprocessor({fitted_status}, vocab_size={vocab_size})"

    def __len__(self):
        return len(self._tfidf.vocabulary_) if hasattr(self._tfidf, 'vocabulary_') else 0

class FeatureBuilder(BaseFeatureBuilder):
    """Combines multiple preprocessors to build final feature matrix"""
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors
        # Load ESI dictionary from JSON file
        with open(ESI_DICT, 'r') as f:
            esi_data = json.load(f)
        self._esi_dict = esi_data['ESI_WORD_DICT']
        self._esi_labels = esi_data['ESI_LABELS']

    def build(self, df: pd.DataFrame):
        try:
            features = []
            for p in self._preprocessors:
                transformed = p.transform(df)
                features.append(transformed)
            # Use sparse hstack if any feature is sparse
            if any(sparse.issparse(f) for f in features):
                X = sparse.hstack(features).toarray()
            else:
                X = np.hstack(features)

            # Add ESI word scores precisely like friend's code
            chief_complaints = df['chief_complaint_raw'].fillna('').astype(str).str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.strip()
            esi_scores = chief_complaints.apply(self._calculate_esi_score)
            
            X = np.hstack([X, esi_scores.values.reshape(-1, 1)])
            return X
        except Exception as e:
            raise RuntimeError(f"Error in feature building: {e}")

    def _get_esi_score_for_token(self, token):
        # Direct match first
        if token in self._esi_dict:
            return self._esi_dict[token]
        
        # Check all partial matches and get the minimum score
        scores = []
        for key, score in self._esi_dict.items():
            if key in token or token in key:
                scores.append(score)
        return min(scores) if scores else None

    def _calculate_esi_score(self, text):
        tokens = text.lower().split()
        min_esi = 5
        # Check unigrams
        for token in tokens:
            score = self._get_esi_score_for_token(token)
            if score is not None and score < min_esi:
                min_esi = score       
        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i+1]
            score = self._get_esi_score_for_token(bigram)
            if score is not None and score < min_esi:
                min_esi = score
                
        return min_esi

    def __add__(self, other):
        # Dunder method for easy concatenation of builders
        if isinstance(other, FeatureBuilder):
            combined_preprocessors = self._preprocessors + other._preprocessors
            return FeatureBuilder(combined_preprocessors)
        raise TypeError("Can only add FeatureBuilder instances")
class ClassImbalanceHandler:
    """Handles class imbalance using SMOTE oversampling"""
    def __init__(self, k_neighbors=None, random_state=42):
        self.k_neighbors = k_neighbors or SMOTE_K_NEIGHBORS
        self.random_state = random_state
        self._smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
    def fit_resample(self, X, y):
        try:
            min_class_count = min(Counter(y).values())
            k = self.k_neighbors
            if min_class_count < k + 1:
                print(f"Warning: Minority class size ({min_class_count}) < k_neighbors+1 ({k+1})")
                # Fallback to default SMOTE or adjust k_neighbors
                k = max(1, min_class_count - 1)
                self._smote = SMOTE(k_neighbors=k, random_state=self.random_state)
            
            X_resampled, y_resampled = self._smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            raise RuntimeError(f"Error during SMOTE resampling: {e}")
