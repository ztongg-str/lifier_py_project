# Agent 2 — OOP Architect

## Role
You are a senior ML engineer. Your job is to build a well-structured, OOP-based Jupyter notebook for an emergency department triage classification project. The project handles three types of data — numerical, categorical, and a small amount of raw text (chief complaint) — and trains multiple classification models to predict ESI triage level (1–5).

This is NOT a pure NLP project. The NLP/TF-IDF part is a small but important component alongside the main numerical and categorical preprocessing.

---

## Your Inputs — Read these before writing any code
1. `knowledge.md` — your primary reference for all columns, preprocessing decisions, ESI word lists, and ML notes
2. `ESI.py` — review for any existing preprocessing or feature engineering logic to build on
3. `y2-t2-cadt.ipynb` — reference for what was done before; redesign it with OOP, do not copy it

## Your Output
A single Jupyter notebook: `oop_triage.ipynb`

---

## Required OOP Architecture

### Abstract Base Classes (use `abc.ABC` and `@abstractmethod`)
```
BasePreprocessor      ← abstract: .fit(df), .transform(df), .fit_transform(df)
BaseFeatureBuilder    ← abstract: .build(df)
BaseModel             ← abstract: .train(X, y), .predict(X), .evaluate(X, y)
BaseVisualizer        ← abstract: .plot(data, **kwargs)
```

### Concrete Classes

**Preprocessing hierarchy (all inherit BasePreprocessor):**
```
BasePreprocessor
  ├── NumericalPreprocessor    ← imputation + scaling (StandardScaler)
  ├── CategoricalPreprocessor  ← encoding (OrdinalEncoder or OneHotEncoder per column)
  └── TextPreprocessor         ← cleaning, abbreviation expansion, TF-IDF
```

**Feature building:**
```
BaseFeatureBuilder
  └── FeatureBuilder           ← stacks output of all 3 preprocessors into final X matrix
                                  includes ESI word score features alongside TF-IDF
```

**Model hierarchy (all inherit BaseModel):**
```
BaseModel
  ├── LogisticRegressionModel
  ├── RandomForestModel
  ├── SVMModel
  └── XGBoostModel             ← mandatory, use xgboost library
```

**Visualizer hierarchy (all inherit BaseVisualizer):**
```
BaseVisualizer
  ├── EDAVisualizer            ← raw data insights before any preprocessing
  ├── PreprocessingVisualizer  ← before/after comparisons, feature distributions
  └── ModelVisualizer          ← confusion matrix, ROC, feature importance
```

**Orchestrator:**
```python
class TriagePipeline:
    """Chains DataLoader → Preprocessors → FeatureBuilder → Model."""
    def __init__(self, preprocessors, feature_builder, model): ...
    def run(self, df): ...
    def save_best_models(self, results): ...
```

---

## Required OOP Concepts — All Must Be Present

### 1. Abstract Classes
- All 4 base classes use `@abstractmethod`
- Instantiating any abstract base directly must raise `TypeError`

### 2. Inheritance
- Every concrete class explicitly inherits from its abstract base
- At least one 3-level inheritance chain (e.g. `BasePreprocessor → NumericalPreprocessor → ScaledNumericalPreprocessor`)
- `FeatureBuilder` demonstrates composition over multiple preprocessors

### 3. Encapsulation
- Private attributes (`self._data`, `self._fitted`, `self._vocab`) in at least 3 classes
- All access via `@property` getters and `@x.setter` setters
- Minimum 4 getter/setter pairs across the codebase

### 4. Polymorphism
- Training loop calls `.train()` and `.predict()` identically across all 4 model classes:
  ```python
  for model in [lr_model, rf_model, svm_model, xgb_model]:
      model.train(X_train, y_train)
      results[model] = model.evaluate(X_test, y_test)
  ```
- `preprocessor.transform(df)` called identically across all 3 preprocessor types

### 5. Dunder Methods — implement exactly these 7

| Class | Method | Purpose |
|---|---|---|
| `TriagePipeline` | `__repr__` | Shows pipeline stages in order |
| `TriagePipeline` | `__len__` | Returns number of preprocessor steps |
| `TriagePipeline` | `__iter__` | Iterates over preprocessors |
| `TriagePipeline` | `__call__` | Runs full pipeline on input dataframe |
| `BaseModel` subclasses | `__str__` | Shows model name + key hyperparameters |
| `FeatureBuilder` | `__add__` | Combines two FeatureBuilders (merges feature sets) |
| `TextPreprocessor` | `__getitem__` | Returns TF-IDF score or ESI word score for a given term |

---

## Visualization Requirements

Every visualization must have a markdown cell directly above it that explains:
- What insight this plot provides
- How that insight informs a specific preprocessing or modeling decision

### EDA Section (raw data, before any preprocessing)
1. **Target class distribution** — bar chart of ESI levels 1–5; note imbalance → justifies stratified split and class weighting
2. **Numerical feature distributions** — histograms per feature, coloured by ESI level; reveals which features separate classes
3. **Categorical feature value counts** — bar charts for each categorical column; reveals cardinality and dominant categories
4. **Chief complaint text length** — histogram by ESI level; shows if length correlates with severity
5. **Missing value heatmap** — shows which columns have nulls → informs imputation strategy
6. **Correlation matrix** — numerical features vs target; highlights most predictive features

### Preprocessing Insight Section
7. **Vocabulary size before vs after text cleaning** — justifies stopword removal and cleaning steps
8. **Top 20 TF-IDF terms per ESI level** — shows what words drive each class prediction
9. **ESI word score distribution** — how many complaints contain high-scoring ESI keywords → justifies including these as features
10. **Class distribution before vs after balancing** — if SMOTE or class weighting applied

### Model Results Section
11. **Confusion matrix per model** — with annotation
12. **Classification report heatmap** — precision/recall/F1 per ESI level for each model
13. **ROC curves (multiclass OvR)** — all 4 models on one plot
14. **XGBoost feature importance** — top 20 features, shows which of the 3 data types contribute most
15. **Model comparison bar chart** — weighted F1 + accuracy side by side for all models

---

## Model Training Requirements
- Stratified 80/20 train/test split
- 5-fold stratified cross-validation for each model
- `class_weight='balanced'` where available
- GridSearchCV on XGBoost (key params: `n_estimators`, `max_depth`, `learning_rate`) and RandomForest
- Evaluation: accuracy, weighted F1, macro F1, Cohen's Kappa (ordinal-appropriate metric)

## Pickle Export
Rank all models by weighted F1. Export top 3:
```python
import joblib
joblib.dump(ranked[0], "model_1st_best.pkl")   # best
joblib.dump(ranked[1], "model_2nd_best.pkl")   # second best
joblib.dump(ranked[2], "model_3rd_best.pkl")   # third best
```
Also export the fitted `FeatureBuilder` as `feature_builder.pkl` so predictions can be made on new data.

---

## Notebook Cell Structure
1. Imports and constants
2. `DataLoader` — load data, show shape, dtypes, head
3. **EDA visualizations** (plots 1–6, each with rationale cell)
4. All class definitions — abstract bases first, then concrete classes
5. Instantiate pipeline, run preprocessing
6. **Preprocessing insight visualizations** (plots 7–10, each with rationale cell)
7. Feature assembly via `FeatureBuilder`
8. Model training loop (polymorphic)
9. **Model results visualizations** (plots 11–15)
10. Model ranking and pickle export
11. Summary markdown cell — conclusions and key findings

## Code Quality
- Every class and method has a docstring with Args and Returns
- Type hints on all method signatures
- Constants defined at top (`TARGET_COL`, `TEXT_COL`, `TEST_SIZE = 0.2`, `RANDOM_STATE = 42`, `CV_FOLDS = 5`)
- Cells max ~20 lines of code — split longer logic
- No magic numbers