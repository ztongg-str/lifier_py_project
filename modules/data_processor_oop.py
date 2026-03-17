# data_processor.py
"""Data loading and preprocessing with OOP design"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import *


class BaseProcessor(ABC):
    """Abstract base class for data processing"""
    
    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        self.metadata = {}
    
    @abstractmethod
    def fit(self, data):
        pass
    
    @abstractmethod
    def transform(self, data):
        pass
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def __str__(self):
        status = "Fitted" if self.is_fitted else "Not fitted"
        return f"{self.name} - {status}"
    
    def __repr__(self):
        return self.__str__()


class DataLoader(BaseProcessor):
    """Load data from CSV files"""
    
    def __init__(self):
        super().__init__("DataLoader")
        self.train = None
        self.test = None
        self.chief_complaints = None
        self.patient_history = None
        self.sample_submission = None
    
    def fit(self, data=None):
        """Load all required datasets from disk"""
        self.train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
        self.test = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
        self.chief_complaints = pd.read_csv(os.path.join(DATA_DIR, CHIEF_COMPLAINT_FILENAME))
        self.patient_history = pd.read_csv(os.path.join(DATA_DIR, PATIENT_HISTORY_FILENAME))
        self.sample_submission = pd.read_csv(os.path.join(DATA_DIR, SAMPLE_SUBMISSION_FILENAME))
        self.is_fitted = True
        return self
    
    def transform(self, data=None):
        if not self.is_fitted:
            raise ValueError("Loader not fitted. Call fit() first.")
        return {
            'train': self.train,
            'test': self.test,
            'chief_complaints': self.chief_complaints,
            'patient_history': self.patient_history,
            'sample_submission': self.sample_submission
        }
    
    def get_summary(self):
        if not self.is_fitted:
            return "Not loaded"
        return f"Train: {self.train.shape}, Test: {self.test.shape}, CC: {self.chief_complaints.shape}, PH: {self.patient_history.shape}"


class DataMerger(BaseProcessor):
    """Merge external data sources with main datasets"""
    
    def __init__(self):
        super().__init__("DataMerger")
        self.train_merged = None
        self.test_merged = None
    
    def fit(self, data_dict):
        """Merge all external data"""
        train = data_dict['train'].copy()
        test = data_dict['test'].copy()
        cc = data_dict['chief_complaints']
        ph = data_dict['patient_history']
        
        # Merge chief complaints
        train = train.merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
        test = test.merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
        
        # Merge patient history
        train = train.merge(ph, on='patient_id', how='left')
        test = test.merge(ph, on='patient_id', how='left')
        
        self.train_merged = train
        self.test_merged = test
        self.is_fitted = True
        return self
    
    def transform(self, data_dict=None):
        if not self.is_fitted:
            raise ValueError("Merger not fitted. Call fit() first.")
        return self.train_merged, self.test_merged


class FeatureEngineer(BaseProcessor):
    """Create clinical risk flags and domain features"""
    
    def __init__(self):
        super().__init__("FeatureEngineer")
    
    def fit(self, data):
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """Create clinical flags"""
        data = data.copy()
        
        # Clinical risk flags based on vital signs
        if 'systolic_bp' in data.columns:
            data['hypotensive'] = (data['systolic_bp'] < 90).astype(int)
        if 'heart_rate' in data.columns:
            data['tachycardic'] = (data['heart_rate'] > 100).astype(int)
        if 'temperature_c' in data.columns:
            data['febrile'] = (data['temperature_c'] > 38).astype(int)
        if 'spo2' in data.columns:
            data['hypoxic'] = (data['spo2'] < 92).astype(int)
        if 'respiratory_rate' in data.columns:
            data['tachypneic'] = (data['respiratory_rate'] > 20).astype(int)
        
        # Comorbidity count
        hx_cols = [col for col in data.columns if col.startswith('hx_')]
        if hx_cols:
            data['comorbidity_count'] = data[hx_cols].sum(axis=1)
        
        return data


class MissingValueHandler(BaseProcessor):
    """Handle missing values strategically"""
    
    def __init__(self):
        super().__init__("MissingValueHandler")
        self.median_values = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def fit(self, data):
        """Learn median values and imputation strategy"""
        # Store median for BP
        for col in BP_COLS:
            if col in data.columns:
                self.median_values[col] = data[col].median()
        
        # Fit general imputer on numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        self.imputer.fit(data[numerical_cols])
        
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """Apply learned imputation"""
        if not self.is_fitted:
            raise ValueError("Handler not fitted. Call fit() first.")
        
        data = data.copy()
        
        # Fill BP with median
        for col in BP_COLS:
            if col in data.columns and col in self.median_values:
                data[col] = data[col].fillna(self.median_values[col])
        
        # Create missing flags
        for col in KEY_VITALS:
            if col in data.columns:
                data[f'{col}_missing'] = data[col].isnull().astype(int)
        
        # Impute remaining numerical
        numerical_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != TARGET]
        data[numerical_cols] = self.imputer.transform(data[numerical_cols])
        
        # Fill text
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].fillna('unknown')
        
        return data


class CategoricalEncoder(BaseProcessor):
    """Encode categorical variables"""
    
    def __init__(self):
        super().__init__("CategoricalEncoder")
        self.encoders = {}
    
    def fit(self, X_train, X_test):
        """Fit encoders on combined train+test to handle all categories"""
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            le = LabelEncoder()
            le.fit(combined)
            self.encoders[col] = le
        
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """Encode categorical features"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        data = data.copy()
        for col, le in self.encoders.items():
            if col in data.columns:
                data[col] = le.transform(data[col].astype(str))
        
        return data


class FeatureSelector(BaseProcessor):
    """Select most important features"""
    
    def __init__(self, k=N_FEATURES_SELECT):
        super().__init__(f"FeatureSelector (k={k})")
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selected_features = None
    
    def fit(self, X, y):
        """Fit feature selector"""
        self.k = min(self.k, X.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Select features"""
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.selector.transform(X)
    
    def get_selected_features(self):
        return self.selected_features


class FeatureScaler(BaseProcessor):
    """Scale features to standard range"""
    
    def __init__(self):
        super().__init__("FeatureScaler")
        self.scaler = StandardScaler()
    
    def fit(self, X):
        """Fit scaler on training data"""
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Scale data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)


class DataProcessor:
    """Main data processing pipeline (uses all processors)"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.merger = DataMerger()
        self.engineer = FeatureEngineer()
        self.missing_handler = MissingValueHandler()
        self.categorical_encoder = CategoricalEncoder()
        self.feature_selector = FeatureSelector()
        self.scaler = FeatureScaler()
        
        self.X = None
        self.y = None
        self.X_test = None
        self.selected_features = None
    
    def load_and_process(self):
        """Execute full pipeline"""
        print("1. Loading data...")
        self.loader.fit()
        data_dict = self.loader.transform()
        
        print("2. Merging data...")
        self.merger.fit(data_dict)
        train_df, test_df = self.merger.transform()
        
        print("3. Engineering features...")
        self.engineer.fit(train_df)
        train_df = self.engineer.transform(train_df)
        test_df = self.engineer.transform(test_df)
        
        print("4. Handling missing values...")
        self.missing_handler.fit(train_df)
        train_df = self.missing_handler.transform(train_df)
        test_df = self.missing_handler.transform(test_df)
        
        print("5. Removing leakage...")
        self.y = train_df[TARGET].copy()
        self.X = train_df.drop(columns=[TARGET] + LEAKAGE_COLS, errors='ignore')
        self.X_test = test_df.drop(columns=[TARGET] + LEAKAGE_COLS, errors='ignore')
        
        print("6. Encoding categoricals...")
        self.categorical_encoder.fit(self.X, self.X_test)
        self.X = self.categorical_encoder.transform(self.X)
        self.X_test = self.categorical_encoder.transform(self.X_test)
        
        # Encode target
        if self.y.dtype == 'object':
            le = LabelEncoder()
            self.y = pd.Series(le.fit_transform(self.y), index=self.y.index)
        
        print("7. Selecting features...")
        self.feature_selector.fit(self.X, self.y)
        X_selected = self.feature_selector.transform(self.X)
        X_test_selected = self.feature_selector.transform(self.X_test)
        self.selected_features = self.feature_selector.get_selected_features()
        
        print("8. Scaling features...")
        self.scaler.fit(X_selected)
        X_scaled = self.scaler.transform(X_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        print("9. Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, self.y, test_size=TEST_SIZE, random_state=SEED, stratify=self.y
        )
        
        print("10. Applying SMOTE...")
        smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("✓ Processing complete!")
        
        return {
            'X_train': X_train_resampled,
            'X_val': X_val,
            'y_train': y_train_resampled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'selected_features': self.selected_features
        }
