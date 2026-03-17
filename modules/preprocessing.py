# preprocessing.py
# Combined module for data preprocessing, feature building, and related utilities

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from modules.config import *
import re
import json

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
    def __init__(self, num_cols=None):
        self._num_cols = num_cols or KEY_VITALS + BP_COLS + ['age', 'weight_kg', 'height_cm', 'bmi', 'shock_index', 'news2_score', 'pain_score', 'gcs_total']
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy='mean')

    def fit(self, df: pd.DataFrame):
        try:
            num_data = df[self._num_cols]
            self._imputer.fit(num_data)
            imputed = self._imputer.transform(num_data)
            self._scaler.fit(imputed)
        except KeyError as e:
            raise KeyError(f"Missing numerical columns in data: {e}")

    def transform(self, df: pd.DataFrame):
        try:
            num_data = df[self._num_cols]
            imputed = self._imputer.transform(num_data)
            scaled = self._scaler.transform(imputed)
            return pd.DataFrame(scaled, columns=self._num_cols, index=df.index)
        except Exception as e:
            raise RuntimeError(f"Error during numerical transformation: {e}")

class CategoricalPreprocessor(BasePreprocessor):
    def __init__(self, cat_cols=None):
        self._cat_cols = cat_cols or ['sex', 'language', 'insurance_type', 'transport_origin', 'pain_location', 'mental_status_triage', 'shift', 'arrival_season', 'arrival_mode']
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, df: pd.DataFrame):
        try:
            cat_data = df[self._cat_cols]
            self._imputer.fit(cat_data)
            imputed = self._imputer.transform(cat_data)
            self._encoder.fit(imputed)
        except KeyError as e:
            raise KeyError(f"Missing categorical columns in data: {e}")

    def transform(self, df: pd.DataFrame):
        try:
            cat_data = df[self._cat_cols]
            imputed = self._imputer.transform(cat_data)
            encoded = self._encoder.transform(imputed)
            feature_names = self._encoder.get_feature_names_out(self._cat_cols)
            return pd.DataFrame(encoded, columns=feature_names, index=df.index)
        except Exception as e:
            raise RuntimeError(f"Error during categorical transformation: {e}")

class TextPreprocessor(BasePreprocessor):
    def __init__(self, text_col='chief_complaint_raw', max_features=500):
        self._text_col = text_col
        # Encapsulation: Adapted exactly from friend's setup
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features
        )

    def _clean_text(self, series: pd.Series):
        # Friend's code logic verbatim (lowercase + clean text case-insensitive)
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

class FeatureBuilder(BaseFeatureBuilder):
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors
        # Load ESI dictionary from JSON file
        with open('data/esi_dictionary.json', 'r') as f:
            esi_data = json.load(f)
        self._esi_dict = esi_data['ESI_WORD_DICT']
        self._esi_labels = esi_data['ESI_LABELS']

    def build(self, df: pd.DataFrame):
        try:
            features = []
            for p in self._preprocessors:
                # Fit if not fitted and has fit method
                if hasattr(p, 'fit') and not hasattr(p, '_is_fitted'):
                    p.fit(df)
                    p._is_fitted = True
                
                if isinstance(p, TextPreprocessor):
                    text_features = p.transform(df)
                    features.append(text_features.toarray() if hasattr(text_features, 'toarray') else text_features)
                else:
                    features.append(p.transform(df))

            X = np.hstack(features)

            # Add ESI word scores precisely like friend's code
            chief_complaints = df['chief_complaint_raw'].fillna('').astype(str).str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).str.strip()
            esi_scores = chief_complaints.apply(self._calculate_esi_score)
            
            X = np.hstack([X, esi_scores.values.reshape(-1, 1)])
            return X
        except Exception as e:
            raise RuntimeError(f"Error in feature building: {e}")

    def _get_esi_score_for_token(self, token):
        if token in self._esi_dict:
            return self._esi_dict[token]
        for key, score in self._esi_dict.items():
            if key in token or token in key:
                return score
        return None

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

class DataProcessorOOP:
    """Orchestrates data processing pipelines"""
    def __init__(self):
        self._imputer = SimpleImputer(strategy='mean')
        self._smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
        self._selector = SelectKBest(score_func=f_classif, k=N_FEATURES_SELECT)

    def process(self, X_train, X_test, y_train):
        try:
            # Impute
            self._imputer.fit(X_train)
            X_train_imputed = self._imputer.transform(X_train)
            X_test_imputed = self._imputer.transform(X_test)

            # Ensure same columns
            if X_train_imputed.shape[1] != X_test_imputed.shape[1]:
                min_cols = min(X_train_imputed.shape[1], X_test_imputed.shape[1])
                X_train_imputed = X_train_imputed[:, :min_cols]
                X_test_imputed = X_test_imputed[:, :min_cols]

            # Feature selection
            self._selector.fit(X_train_imputed, y_train)
            X_train_selected = self._selector.transform(X_train_imputed)
            X_test_selected = self._selector.transform(X_test_imputed)

            # SMOTE
            X_train_smote, y_train_smote = self._smote.fit_resample(X_train_selected, y_train)

            return X_train_smote, X_test_selected, y_train_smote
        except Exception as e:
            raise RuntimeError(f"Error executing complete DataProcessor pipeline: {e}")

class FeatureSelector:
    def __init__(self):
        self.selector_kbest = SelectKBest(score_func=f_classif, k=SELECT_K_BEST)
        self.selector_sfm = None

    def select_features(self, X, y):
        try:
            X_kbest = self.selector_kbest.fit_transform(X, y)
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
            self.selector_sfm = SelectFromModel(rf, threshold='median')
            X_selected = self.selector_sfm.fit_transform(X_kbest, y)
            return X_selected
        except Exception as e:
            raise RuntimeError(f"Error in feature selection: {e}")

    def transform(self, X):
        X_kbest = self.selector_kbest.transform(X)
        return self.selector_sfm.transform(X_kbest)