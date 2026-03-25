# models.py
# Combined module for machine learning models and training

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from modules.config import *
from xgboost import XGBClassifier

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    @abstractmethod
    def create(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y):
        """Fit the model by creating and training it"""
        self.model = self.create()
        self.train(X, y)
        return self

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation"""
    def __init__(self, **kwargs):
        self.name = 'LogisticRegression'
        params = {**LR_PARAMS, **kwargs}
        self.model = LogisticRegression(**params)
        
    def create(self):
        return self.model

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }

    def __str__(self):
        return f"{self.name} (penalty={getattr(self.model, 'penalty', None)})"

class RandomForestModel(BaseModel):
    """Random Forest classifier implementation"""
    def __init__(self, **kwargs):
        self.name = 'RandomForest'
        params = {**RF_PARAMS, **kwargs}
        self.model = RandomForestClassifier(**params)

    def create(self):
        return self.model

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }


class XGBoostModel(BaseModel):
    """XGBoost classifier implementation"""
    def __init__(self, **kwargs):
        self.name = 'XGBoost'
        params = {**XG_PARAMS, **kwargs}
        self.model = XGBClassifier(**params)

    def create(self):
        return self.model

    def train(self, X, y):
        # Adjust labels for XGBoost (0-based)
        y_adj = y - 1
        self.model.fit(X, y_adj)
        return self

    def predict(self, X):
        # Adjust predictions back to 1-based
        return self.model.predict(X) + 1

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }

class ModelTrainer:
    """Manages training of multiple ML models"""
    def __init__(self):
        self.models = {}
        self.ensemble = None
        
        # Initialize using the BaseModel classes
        self.models['Logistic Regression'] = LogisticRegressionModel()
        self.models['Random Forest'] = RandomForestModel(n_estimators=100)
        self.models['XGBoost'] = XGBoostModel()
        

    def train_models(self, X_train, y_train):
        """Train all individual models"""
        trained = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
        return trained
    
    def create_ensemble(self, X_train, y_train, ensemble_type='voting'):
        """Create and train an ensemble of all available models"""
        estimators = []
        for name, model in self.models.items():
            if hasattr(model.model, 'predict_proba'):  # Check if model supports probability
                estimators.append((name.lower().replace(' ', '_'), model.model))
        
        if ensemble_type == 'voting':
            self.ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft'  # Use soft voting for probability-based
            )
        elif ensemble_type == 'stacking':
            from sklearn.linear_model import LogisticRegression
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=SEED)
            )
        # Train the ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Create a wrapper for the ensemble to match BaseModel interface
        ensemble_wrapper = type('EnsembleModel', (BaseModel,), {
            'name': f'Ensemble ({ensemble_type})',
            'model': self.ensemble,
            'create': lambda self: self.model,
            'train': lambda self, X, y: self.model.fit(X, y) or self,
            'predict': lambda self, X: self.model.predict(X),
            'evaluate': lambda self, X, y: {
                'accuracy': accuracy_score(y, self.predict(X)),
                'precision': precision_score(y, self.predict(X), average='weighted'),
                'recall': recall_score(y, self.predict(X), average='weighted'),
                'f1': f1_score(y, self.predict(X), average='weighted')
            }
        })()
        
        return ensemble_wrapper