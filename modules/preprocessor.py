# preprocessor.py
# Handles data preprocessing with OOP principles

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod
from config import *

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessing (Abstraction)."""
    
    def __init__(self):
        self._scaler = StandardScaler()  # Encapsulated attribute
        self._imputer = SimpleImputer(strategy='mean')
        self._label_encoders = {}
        self._target_le = None
        self._selector = None
        self._smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
        self._is_fitted = False  # Encapsulated state
    
    @abstractmethod
    def preprocess(self, train_data, test_data, chief_complaint_data, patient_history_data):
        """Abstract method for preprocessing (Polymorphism)."""
        pass
    
    def __str__(self):
        """Dunder method for string representation."""
        return f"{self.__class__.__name__} - Fitted: {self._is_fitted}"
    
    def __repr__(self):
        """Dunder method for representation."""
        return self.__str__()

class DataPreprocessor(BasePreprocessor):
    """Concrete preprocessor inheriting from BasePreprocessor (Inheritance)."""
    
    def preprocess(self, train_data, test_data, chief_complaint_data, patient_history_data):
        try:
            # Merge data (File handling via pandas)
            train_data = train_data.merge(chief_complaint_data[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
            test_data = test_data.merge(chief_complaint_data[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
            train_data = train_data.merge(patient_history_data, on='patient_id', how='left')
            test_data = test_data.merge(patient_history_data, on='patient_id', how='left')

            # Add comorbidity_count
            hx_cols = [col for col in train_data.columns if col.startswith('hx_')]
            train_data['comorbidity_count'] = train_data[hx_cols].sum(axis=1)
            test_data['comorbidity_count'] = test_data[hx_cols].sum(axis=1)

            # Feature engineering
            self._add_clinical_flags(train_data)
            self._add_clinical_flags(test_data)

            # Handle missing values
            self._handle_missing(train_data, test_data)

            # Remove leakage
            leakage_cols = ['ed_los_hours', 'disposition']
            X = train_data.drop(columns=[TARGET, PATIENT_ID] + leakage_cols, errors='ignore')
            y = train_data[TARGET]
            X_test = test_data.drop(columns=[TARGET, PATIENT_ID] + leakage_cols, errors='ignore')

            # Encode categoricals
            self._encode_categoricals(X, X_test)

            # Encode target
            if isinstance(y, pd.Series) and y.dtype == 'object':
                self._target_le = LabelEncoder()
                y = self._target_le.fit_transform(y)

            # Feature selection
            self._selector = SelectKBest(score_func=f_classif, k=SELECT_K_BEST)
            X_selected = self._selector.fit_transform(X, y)
            X_test_selected = self._selector.transform(X_test)

            # SelectFromModel for TF-IDF
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
            sfm = SelectFromModel(rf, threshold='median')
            X_selected = sfm.fit_transform(X_selected, y)
            X_test_selected = sfm.transform(X_test_selected)

            # Scale
            X_scaled = self._scaler.fit_transform(X_selected)
            X_test_scaled = self._scaler.transform(X_test_selected)

            self._is_fitted = True
            return X_scaled, X_test_scaled, y
        except Exception as e:
            raise Exception(f"Preprocessing error: {e}")  # Error handling
    
    def _add_clinical_flags(self, data):
        """Private method for feature engineering (Encapsulation)."""
        if 'systolic_bp' in data.columns:
            data['hypotensive'] = (data['systolic_bp'] < 90).astype(int)
        if 'heart_rate' in data.columns:
            data['tachycardic'] = (data['heart_rate'] > 100).astype(int)
        if 'temperature_c' in data.columns:
            data['febrile'] = (data['temperature_c'] > 38).astype(int)
        if 'spo2' in data.columns:
            data['hypoxic'] = (data['spo2'] < 92).astype(int)
        if 'age' in data.columns and 'hypotensive' in data.columns:
            data['elderly_hypotensive'] = ((data['age'] > 65) & (data['hypotensive'] == 1)).astype(int)

    def _handle_missing(self, train_data, test_data):
        """Private method for missing value handling."""
        # BP imputation
        bp_cols = ['systolic_bp', 'diastolic_bp']
        for col in bp_cols:
            if col in train_data.columns:
                median_val = train_data[col].median()
                train_data[col] = train_data[col].fillna(median_val)
                test_data[col] = test_data[col].fillna(median_val)

        # Missing flags
        key_vitals = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2', 'respiratory_rate']
        for col in key_vitals:
            if col in train_data.columns:
                train_data[f'{col}_missing'] = train_data[col].isnull().astype(int)
                test_data[f'{col}_missing'] = test_data[col].isnull().astype(int)

        # Impute numerical
        numerical_cols = [col for col in train_data.select_dtypes(include=[np.number]).columns if col in test_data.columns and col != TARGET]
        train_data[numerical_cols] = self._imputer.fit_transform(train_data[numerical_cols])
        test_data[numerical_cols] = self._imputer.transform(test_data[numerical_cols])

    def _encode_categoricals(self, X, X_test):
        """Private method for encoding."""
        categorical_cols = [col for col in X.select_dtypes(include=['object']).columns if col in X_test.columns]
        for col in categorical_cols:
            # Combine train and test to fit encoder
            combined = pd.concat([X[col], X_test[col]], axis=0)
            le = LabelEncoder()
            le.fit(combined)
            X[col] = le.transform(X[col])
            X_test[col] = le.transform(X_test[col])
            self._label_encoders[col] = le

    def apply_smote(self, X_train, y_train):
        """Public method for SMOTE (Polymorphism if overridden)."""
        try:
            X_resampled, y_resampled = self._smote.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except Exception as e:
            raise Exception(f"SMOTE error: {e}")  # Error handling