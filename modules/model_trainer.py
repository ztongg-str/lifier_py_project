# model_trainer.py
"""Model training with OOP design"""

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *


class BaseModel(ABC):
    """Abstract base class for models"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        self.predictions = None
        self.probabilities = None
    
    @abstractmethod
    def create_model(self):
        pass
    
    def fit(self, X, y):
        """Train the model"""
        if self.model is None:
            self.create_model()
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"{self.name} not trained yet")
        self.predictions = self.model.predict(X)
        return self.predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError(f"{self.name} not trained yet")
        if hasattr(self.model, 'predict_proba'):
            self.probabilities = self.model.predict_proba(X)
            return self.probabilities
        return None
    
    def evaluate(self, y_true, y_pred):
        """Calculate metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        return {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores
        }
    
    def __str__(self):
        status = "Trained" if self.is_trained else "Not trained"
        return f"{self.name} - {status}"
    
    def __repr__(self):
        return self.__str__()
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier"""
    
    def __init__(self):
        super().__init__("Logistic Regression")
    
    def create_model(self):
        self.model = LogisticRegression(**LR_PARAMS)


class RandomForestModel(BaseModel):
    """Random Forest classifier"""
    
    def __init__(self):
        super().__init__("Random Forest")
    
    def create_model(self):
        self.model = RandomForestClassifier(**RF_PARAMS)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier"""
    
    def __init__(self):
        super().__init__("Gradient Boosting")
    
    def create_model(self):
        self.model = GradientBoostingClassifier(**GB_PARAMS)


class ModelTrainer:
    """Trains and manages multiple models"""
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegressionModel(),
            'Random Forest': RandomForestModel(),
            'Gradient Boosting': GradientBoostingModel()
        }
        self.results = []
        self.best_model = None
        self.best_name = None
    
    def train_all(self, X_train, y_train):
        """Train all models"""
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
        print("✓ All models trained")
    
    def evaluate_all(self, X_val, y_val, X_train, y_train):
        """Evaluate all models"""
        print("\nEvaluating models...")
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            metrics = model.evaluate(y_val, y_pred)
            
            # Cross-validation
            cv_results = model.cross_validate(X_train, y_train)
            
            result = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'CV F1 Mean': cv_results['cv_mean'],
                'CV F1 Std': cv_results['cv_std']
            }
            results.append(result)
        
        self.results = pd.DataFrame(results)
        
        # Find best model
        best_idx = self.results['F1'].idxmax()
        self.best_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_name]
        
        print(f"\nBest Model: {self.best_name} (F1: {self.results.loc[best_idx, 'F1']:.4f})")
        
        return self.results
    
    def get_best_model(self):
        """Return the best performing model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_all() and evaluate_all()")
        return self.best_model, self.best_name
    
    def get_results(self):
        """Return evaluation results"""
        return self.results
    
    def get_all_models(self):
        """Return all trained models"""
        return self.models
