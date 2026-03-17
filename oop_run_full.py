"""Full OOP pipeline runner: trains models, evaluates, ranks, and saves top models.
Run from project root: `python oop_run_full.py`.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE

from modules.data_loader import load_data
from modules.preprocessors_oop import NumericalPreprocessor, CategoricalPreprocessor, TextPreprocessor
from modules.feature_builder import FeatureBuilder
from modules.oop_models import LogisticRegressionModel, RandomForestModel, XGBoostModel
from modules.oop_evaluator import ModelEvaluator
from modules.config import SEED


def build_feature_pipeline(train_df):
    num = NumericalPreprocessor()
    cat = CategoricalPreprocessor()
    text = TextPreprocessor(text_col='chief_complaint_system', max_features=2000)
    fb = FeatureBuilder([num, cat, text])

    # fit preprocessors on train
    for p in fb.preprocessors:
        p.fit(train_df)

    return fb


def evaluate_model(model, X_val, y_val, X_train=None, y_train=None):
    y_pred = model.predict(X_val)
    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision_weighted': float(precision_score(y_val, y_pred, average='weighted')),
        'f1_weighted': float(f1_score(y_val, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_val, y_pred, average='macro')),
        'cohen_kappa': float(cohen_kappa_score(y_val, y_pred))
    }
    return metrics


def main():
    out_dir = os.getcwd()
    cc, ph, train_df, test_df, sub = load_data()

    # target column detection
    if 'triage_acuity' in train_df.columns:
        target_col = 'triage_acuity'
    elif 'esi' in train_df.columns:
        target_col = 'esi'
    elif 'target' in train_df.columns:
        target_col = 'target'
    else:
        raise RuntimeError('No target column found in train data')

    fb = build_feature_pipeline(train_df)
    X_all = fb.build(train_df)
    y_all = train_df[target_col]

    # stratified split
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, stratify=y_all, test_size=0.2, random_state=SEED)

    # Apply SMOTE on training set only
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train.fillna(0), y_train)

    # Instantiate models
    models = {
        'LogisticRegression': LogisticRegressionModel(),
        'RandomForest': RandomForestModel(),
    }
    # XGBoost optional
    try:
        xgb = XGBoostModel()
        models['XGBoost'] = xgb
    except Exception:
        pass

    results = []

    # Grid search for RandomForest and XGBoost
    if 'RandomForest' in models:
        rf = models['RandomForest']
        # access underlying estimator
        estimator = rf.model
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
        }
        gs = GridSearchCV(estimator, param_grid, scoring='f1_weighted', cv=3, n_jobs=1)
        gs.fit(X_train_res, y_train_res)
        rf.model = gs.best_estimator_

    if 'XGBoost' in models:
        xgb = models['XGBoost']
        estimator = xgb.model
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.01]
        }
        gs = GridSearchCV(estimator, param_grid, scoring='f1_weighted', cv=3, n_jobs=1)
        gs.fit(X_train_res, y_train_res)
        xgb.model = gs.best_estimator_

    # Train all models
    for name, m in models.items():
        print(f'Training {name}...')
        m.train(X_train_res, y_train_res)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics_df = []
    for name, m in models.items():
        met = evaluate_model(m, X_val.fillna(0), y_val, X_train_res, y_train_res)
        met['Model'] = name
        metrics_df.append(met)

    metrics_df = pd.DataFrame(metrics_df).set_index('Model')
    print('\nEvaluation metrics:')
    print(metrics_df)

    # Rank by weighted F1
    ranked = metrics_df.sort_values('f1_weighted', ascending=False)
    print('\nRanked models:')
    print(ranked)

    # Save top 3 models and feature builder
    top_models = ranked.index.tolist()[:3]
    saved = []
    for i, name in enumerate(top_models):
        model_obj = models[name]
        fname = os.path.join(out_dir, f'model_{i+1}_{name}.pkl')
        joblib.dump(model_obj, fname)
        saved.append(fname)

    fb_file = os.path.join(out_dir, 'feature_builder.pkl')
    joblib.dump(fb, fb_file)

    # Save metrics
    metrics_path = os.path.join(out_dir, 'oop_metrics.csv')
    metrics_df.to_csv(metrics_path)

    print('\nSaved models:', saved)
    print('Saved feature builder:', fb_file)
    print('Saved metrics:', metrics_path)


if __name__ == '__main__':
    main()
