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

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

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

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.name = 'SVM'
        self.model = SVC(random_state=SEED, class_weight='balanced', **kwargs)

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
        if not _HAS_XGB:
            raise ImportError("XGBoost not available")
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
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=SEED, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=SEED)
        }
        self.ensemble = VotingClassifier(estimators=[
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting'])
        ], voting='soft')

    def train_models(self, X_train, y_train):
        trained = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained[name] = model
        self.ensemble.fit(X_train, y_train)
        trained['Ensemble'] = self.ensemble
        return trained