# models.py
# Module for machine learning models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from config import SEED

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