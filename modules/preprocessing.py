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

# ESI word dictionary
ESI_WORD_DICT = {
    # ESI LEVEL 1
    "cardiac arrest": 1,
    "respiratory arrest": 1,
    "pulseless electrical": 1,
    "ventricular fibrillation": 1,
    "asthmaticus unresponsive": 1,
    "status epilepticus": 1,
    "epilepticus refractory": 1,
    "coma unknown": 1,
    "hypothyroid coma": 1,
    "collapse unresponsive": 1,
    "mva unresponsive": 1,
    "hypoglycaemia unresponsive": 1,
    "unresponsive": 1,
    "cardiac": 1,
    "arrest": 1,
    "drowning": 1,
    "near drowning": 1,
    "gunshot wound": 1,
    "traumatic amputation": 1,
    "traumatic brain": 1,
    "massive transfusion": 1,
    "tension pneumothorax": 1,
    "airway obstruction": 1,
    "airway compromise": 1,
    "locked syndrome": 1,
    "pulseless": 1,
    "fulminant hepatic": 1,
    "fulminant hepatitis": 1,
    # Add more as needed
}

class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError()

class BaseFeatureBuilder(ABC):
    @abstractmethod
    def build(self, df: pd.DataFrame):
        raise NotImplementedError()

class NumericalPreprocessor(BasePreprocessor):
    def __init__(self, num_cols=None):
        self._num_cols = num_cols or KEY_VITALS + BP_COLS + ['age', 'weight_kg', 'height_cm', 'bmi', 'shock_index', 'news2_score', 'pain_score', 'gcs_total']
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy='mean')

    def fit(self, df: pd.DataFrame):
        num_data = df[self._num_cols]
        self._imputer.fit(num_data)
        imputed = self._imputer.transform(num_data)
        self._scaler.fit(imputed)

    def transform(self, df: pd.DataFrame):
        num_data = df[self._num_cols]
        imputed = self._imputer.transform(num_data)
        scaled = self._scaler.transform(imputed)
        return pd.DataFrame(scaled, columns=self._num_cols, index=df.index)

class CategoricalPreprocessor(BasePreprocessor):
    def __init__(self, cat_cols=None):
        self._cat_cols = cat_cols or ['sex', 'language', 'insurance_type', 'transport_origin', 'pain_location', 'mental_status_triage', 'shift', 'arrival_season', 'arrival_mode']
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, df: pd.DataFrame):
        cat_data = df[self._cat_cols]
        self._imputer.fit(cat_data)
        imputed = self._imputer.transform(cat_data)
        self._encoder.fit(imputed)

    def transform(self, df: pd.DataFrame):
        cat_data = df[self._cat_cols]
        imputed = self._imputer.transform(cat_data)
        encoded = self._encoder.transform(imputed)
        feature_names = self._encoder.get_feature_names_out(self._cat_cols)
        return pd.DataFrame(encoded, columns=feature_names, index=df.index)

class TextPreprocessor(BasePreprocessor):
    def __init__(self, text_col='chief_complaint_raw', max_features=500):
        self._text_col = text_col
        self._tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit(self, df: pd.DataFrame):
        texts = df[self._text_col].fillna('').astype(str)
        self._tfidf.fit(texts)

    def transform(self, df: pd.DataFrame):
        texts = df[self._text_col].fillna('').astype(str)
        return self._tfidf.transform(texts)

class FeatureBuilder(BaseFeatureBuilder):
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors
        self._esi_dict = ESI_WORD_DICT

    def build(self, df: pd.DataFrame):
        features = []
        for p in self._preprocessors:
            if isinstance(p, TextPreprocessor):
                text_features = p.transform(df)
                features.append(text_features.toarray() if hasattr(text_features, 'toarray') else text_features)
            else:
                features.append(p.transform(df))

        X = np.hstack(features)

        # Add ESI word scores
        chief_complaints = df['chief_complaint_raw'].fillna('').astype(str).str.lower()
        esi_scores = chief_complaints.apply(self._calculate_esi_score)
        X = np.hstack([X, esi_scores.values.reshape(-1, 1)])

        return X

    def _calculate_esi_score(self, text):
        words = text.split()
        scores = [self._esi_dict.get(word, 5) for word in words if word in self._esi_dict]
        return min(scores) if scores else 5

    def __add__(self, other):
        if isinstance(other, FeatureBuilder):
            combined_preprocessors = self._preprocessors + other._preprocessors
            return FeatureBuilder(combined_preprocessors)
        raise TypeError("Can only add FeatureBuilder instances")

class DataProcessorOOP:
    def __init__(self):
        self._imputer = SimpleImputer(strategy='mean')
        self._smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
        self._selector = SelectKBest(score_func=f_classif, k=N_FEATURES_SELECT)

    def process(self, X_train, X_test, y_train):
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

class FeatureSelector:
    def __init__(self):
        self.selector_kbest = SelectKBest(score_func=f_classif, k=SELECT_K_BEST)
        self.selector_sfm = None

    def select_features(self, X, y):
        X_kbest = self.selector_kbest.fit_transform(X, y)
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
        self.selector_sfm = SelectFromModel(rf, threshold='median')
        X_selected = self.selector_sfm.fit_transform(X_kbest, y)
        return X_selected

    def transform(self, X):
        X_kbest = self.selector_kbest.transform(X)
        return self.selector_sfm.transform(X_kbest)