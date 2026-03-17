# models.py
# Combined module for machine learning models and training

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from modules.config import SEED
from xgboost import XGBClassifier

class BaseModel(ABC):
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
        self.model = self.create()
        self.train(X, y)
        return self

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.name = 'LogisticRegression'
        self.model = LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=1000, **kwargs)

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
    def __init__(self, **kwargs):
        self.name = 'RandomForest'
        self.model = RandomForestClassifier(random_state=SEED, class_weight='balanced', **kwargs)

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
    def __init__(self, **kwargs):
        self.name = 'XGBoost'
        self.model = XGBClassifier(random_state=SEED, **kwargs)

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

class TriagePipeline:
    def __init__(self, preprocessors, feature_builder, models=None):
        self._preprocessors = preprocessors
        self._feature_builder = feature_builder
        self._models = models or []

    def run(self, df: pd.DataFrame):
        for p in self._preprocessors:
            p.fit(df)
        X = self._feature_builder.build(df)
        return X

    def __len__(self):
        return len(self._preprocessors)

    def __iter__(self):
        return iter(self._preprocessors)

    def __call__(self, df: pd.DataFrame):
        return self.run(df)

    def __repr__(self):
        return f"TriagePipeline with {len(self._preprocessors)} preprocessors and {len(self._models)} models"

    def add_model(self, model: BaseModel):
        self._models.append(model)

class ModelTrainer:
    def __init__(self, include_xgboost=True):
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
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        
        # Prepare estimators for ensemble (only models that support predict_proba)
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




## Todo:
# Remove Kwargs, look how the dunder methods are used. also how the encapsulation still applied.