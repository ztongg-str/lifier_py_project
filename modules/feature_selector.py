# feature_selector.py
# Module for feature selection

from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from config import SELECT_K_BEST

class FeatureSelector:
    def __init__(self):
        self.selector_kbest = SelectKBest(score_func=f_classif, k=SELECT_K_BEST)
        self.selector_sfm = None

    def select_features(self, X, y):
        # SelectKBest
        X_kbest = self.selector_kbest.fit_transform(X, y)

        # SelectFromModel
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.selector_sfm = SelectFromModel(rf, threshold='median')
        X_selected = self.selector_sfm.fit_transform(X_kbest, y)

        return X_selected

    def transform(self, X):
        X_kbest = self.selector_kbest.transform(X)
        return self.selector_sfm.transform(X_kbest)