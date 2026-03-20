# models.py
# Combined module for machine learning models and training

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from modules.config import SEED
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

class BaseModel(ABC):
    """Abstract base class for machine learning models."""

    @abstractmethod
    def create(self):
        """Create and return the model instance."""
        raise NotImplementedError()

    @abstractmethod
    def train(self, X, y):
        """Train the model on given data."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """Make predictions on given data."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate model performance on given data."""
        raise NotImplementedError()

    def fit(self, X, y):
        """Fit the model (create and train)."""
        self.model = self.create()
        self.train(X, y)
        return self

    def __str__(self):
        """String representation of the model."""
        return f"{self.name} ({self.__class__.__name__})"

    def __repr__(self):
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}')"

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""

    def __init__(self, **kwargs):
        """Initialize Logistic Regression model.

        Args:
            **kwargs: Additional arguments for LogisticRegression.
        """
        self.name = 'LogisticRegression'
        self.model = LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=1000, **kwargs)

    def create(self):
        """Create LogisticRegression model instance."""
        return self.model

    def train(self, X, y):
        """Train the LogisticRegression model.

        Args:
            X: Training features.
            y: Training labels.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions with LogisticRegression.

        Args:
            X: Input features.

        Returns:
            Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate LogisticRegression performance.

        Args:
            X: Test features.
            y: True labels.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }

    def __str__(self):
        """String representation of the model."""
        return f"{self.name} (penalty={getattr(self.model, 'penalty', None)})"

class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self, **kwargs):
        """Initialize Random Forest model.

        Args:
            **kwargs: Additional arguments for RandomForestClassifier.
        """
        self.name = 'RandomForest'
        self.model = RandomForestClassifier(random_state=SEED, class_weight='balanced', **kwargs)

    def create(self):
        """Create RandomForestClassifier model instance."""
        return self.model

    def train(self, X, y):
        """Train the RandomForest model.

        Args:
            X: Training features.
            y: Training labels.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions with RandomForest.

        Args:
            X: Input features.

        Returns:
            Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate RandomForest performance.

        Args:
            X: Test features.
            y: True labels.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, **kwargs):
        """Initialize XGBoost model.

        Args:
            **kwargs: Additional arguments for XGBClassifier.
        """
        self.name = 'XGBoost'
        self.model = XGBClassifier(random_state=SEED, **kwargs)

    def create(self):
        """Create XGBClassifier model instance."""
        return self.model

    def train(self, X, y):
        """Train the XGBoost model.

        Args:
            X: Training features.
            y: Training labels.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions with XGBoost.

        Args:
            X: Input features.

        Returns:
            Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate XGBoost performance.

        Args:
            X: Test features.
            y: True labels.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
class ModelTrainer:
    """Manages training of multiple ML models."""

    def __init__(self):
        """Initialize ModelTrainer with default models."""
        self.models = {}
        self.ensemble = None
        
        # Initialize using the BaseModel classes
        self.models['Logistic Regression'] = LogisticRegressionModel()
        self.models['Random Forest'] = RandomForestModel(n_estimators=100)
        self.models['XGBoost'] = XGBoostModel()
        

    def train_models(self, X_train, y_train):
        """Train all individual models.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dict of trained models.
        """
        trained = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
        return trained
    
    def create_ensemble(self, X_train, y_train, ensemble_type='voting'):
        
        # Prepare estimators for ensemble (only models that support predict_proba)
        estimators = []
        for name, model in self.models.items():
            if hasattr(model.model, 'predict_proba'):  # Check if model supports probability
                estimators.append((name.lower().replace(' ', '_'), model.model))
        
        if ensemble_type == 'voting':
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft'  # Use soft voting for probability-based
            )
        elif ensemble_type == 'stacking':
            from sklearn.linear_model import LogisticRegression
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=SEED)
            ) 
        # Train the ensemble
        ensemble.fit(X_train, y_train)
        
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
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare their performance"""
        trained_models = self.train_models(X_train, y_train)
        
        # Add ensemble
        ensemble_model = self.create_ensemble(X_train, y_train)
        trained_models[ensemble_model.name] = ensemble_model
        
        # Evaluate all models
        results = {}
        for name, model in trained_models.items():
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            results[name] = {
                'train': train_metrics,
                'test': test_metrics
            }
            
        return results, trained_models
    
    def get_best_model(self, X_train, y_train, X_test, y_test, metric='f1'):
        """Find the best performing model based on specified metric"""
        results, models = self.compare_models(X_train, y_train, X_test, y_test)
        
        best_model_name = max(
            results.keys(), 
            key=lambda name: results[name]['test'][metric]
        )
        
        return best_model_name, models[best_model_name], results