# Agent 3 — Flat Notebook Author

## Role
You are a data scientist who writes clean, well-documented Jupyter notebooks in a linear, procedural style. Your job is to produce the exact same triage classification pipeline as Agent 2's OOP notebook — handling numerical, categorical, and text data — but with NO classes and NO OOP patterns. Everything runs sequentially in plain functions and cells.

The goal is to produce two notebooks that do identical work but with completely different architecture, so they can be compared side by side.

---

## Your Inputs
1. `knowledge.md` — primary reference for all domain logic, column info, preprocessing decisions
2. `oop_triage.ipynb` — mirror the preprocessing decisions and model choices exactly so results are comparable
3. `ESI.py` and `y2-t2-cadt.ipynb` — for any implementation details

## Your Output
A single notebook: `flat_triage.ipynb`

---

## What "Flat" Means
- No `class` keyword anywhere
- No abstract bases, no inheritance, no encapsulation
- All logic in standalone functions or inline in cells
- Notebook reads naturally top-to-bottom like a tutorial or analysis report

---

## Constants Block — Cell 1
Define everything here so it's easy to find:
```python
# === PROJECT CONSTANTS ===
TARGET_COL     = "ESI"
TEXT_COL       = "chief_complaint"   # update to actual column name
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CV_FOLDS       = 5
MAX_TFIDF_FEATURES = 500

# Abbreviation dictionary (from knowledge.md)
ABBREV_DICT = {
    "cp":   "chest pain",
    "sob":  "shortness of breath",
    "htn":  "hypertension",
    "n/v":  "nausea vomiting",
    # ... all from knowledge.md Section 4
}

# ESI word scores (from knowledge.md / ESI.py)
ESI_WORD_SCORES = {
    # word: score — copy all from knowledge.md
}

# Medical stopwords to KEEP (negation words)
KEEP_WORDS = ["no", "not", "without", "denies", "negative", "none"]
```

---

## Required Sections

### 1. Setup and Data Loading
```python
df = pd.read_csv("data/your_dataset.csv")
print(df.shape)
df.head()
```

### 2. EDA (same 6 plots as OOP notebook)
Each plot preceded by a markdown cell with the insight and decision it informs.
Plain seaborn/matplotlib calls — no class methods.

### 3. Preprocessing Functions

**Numerical:**
```python
def impute_numerical(df, strategy='median'):
    """Fill missing values in numerical columns."""
    ...

def scale_numerical(df, cols, scaler=None):
    """Fit or apply StandardScaler to numerical columns."""
    ...
```

**Categorical:**
```python
def encode_categorical(df, cols, encoders=None):
    """Apply OrdinalEncoder or OneHotEncoder per column."""
    ...
```

**Text:**
```python
def clean_text(text):
    """Lowercase, remove punctuation, strip whitespace."""
    ...

def expand_abbreviations(text, abbrev_dict):
    """Replace medical abbreviations with full terms."""
    ...

def remove_stopwords(text, keep_words):
    """Remove stopwords but preserve negation terms."""
    ...

def compute_esi_score_features(text, word_scores):
    """Return numeric ESI score features from word list."""
    ...

def build_tfidf_features(corpus, max_features, fitted_vectorizer=None):
    """Fit or apply TfidfVectorizer. Returns matrix + vectorizer."""
    ...
```

Apply in sequence:
```python
df['text_clean']    = df[TEXT_COL].apply(clean_text)
df['text_expanded'] = df['text_clean'].apply(expand_abbreviations, args=(ABBREV_DICT,))
df['text_final']    = df['text_expanded'].apply(remove_stopwords, args=(KEEP_WORDS,))

X_tfidf, tfidf_vec  = build_tfidf_features(df['text_final'], MAX_TFIDF_FEATURES)
X_esi_scores        = np.vstack(df['text_final'].apply(compute_esi_score_features, args=(ESI_WORD_SCORES,)))
```

### 4. Preprocessing Insight Visualizations (same 4 plots as OOP notebook)
Each with a markdown rationale cell.

### 5. Feature Assembly
```python
import scipy.sparse as sp

X_numerical   = scale_numerical(df, num_cols)
X_categorical = encode_categorical(df, cat_cols)
X_text        = sp.hstack([X_tfidf, X_esi_scores])

X_final = sp.hstack([X_numerical, X_categorical, X_text])
y       = df[TARGET_COL].values
```

### 6. Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
```

### 7. Model Training — one block per model
```python
# --- Logistic Regression ---
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# --- Random Forest ---
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# --- SVM ---
from sklearn.svm import SVC
svm = SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

# --- XGBoost ---
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
```

### 8. Evaluation Function
```python
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

def evaluate_model(model, X_test, y_test, name):
    """Return dict of evaluation metrics for a fitted model."""
    preds = model.predict(X_test)
    return {
        'name':        name,
        'accuracy':    accuracy_score(y_test, preds),
        'weighted_f1': f1_score(y_test, preds, average='weighted'),
        'macro_f1':    f1_score(y_test, preds, average='macro'),
        'kappa':       cohen_kappa_score(y_test, preds),
    }

results = [
    evaluate_model(lr,  X_test, y_test, 'Logistic Regression'),
    evaluate_model(rf,  X_test, y_test, 'Random Forest'),
    evaluate_model(svm, X_test, y_test, 'SVM'),
    evaluate_model(xgb, X_test, y_test, 'XGBoost'),
]
results_df = pd.DataFrame(results).sort_values('weighted_f1', ascending=False)
```

### 9. Model Results Visualizations (same 5 plots as OOP notebook)

### 10. Pickle Export
```python
import joblib

model_objects = {'Logistic Regression': lr, 'Random Forest': rf,
                 'SVM': svm, 'XGBoost': xgb}
ranked = results_df['name'].tolist()

joblib.dump(model_objects[ranked[0]], "model_1st_best.pkl")
joblib.dump(model_objects[ranked[1]], "model_2nd_best.pkl")
joblib.dump(model_objects[ranked[2]], "model_3rd_best.pkl")
joblib.dump(tfidf_vec,                "tfidf_vectorizer.pkl")
print(f"Best model: {ranked[0]}")
```

### 11. Summary Markdown Cell
- Which model performed best and by how much
- Which data type (numerical / categorical / text) contributed most (from XGBoost feature importance)
- Key preprocessing decisions and the insights that justified them
- Differences observed vs the OOP notebook (if any)

---

## Rules
- No `class` keyword — if you catch yourself writing one, refactor to a function
- Every function has a docstring
- Every visualization has a markdown rationale cell directly above it
- All results must be reproducible — `random_state=RANDOM_STATE` everywhere
- Preprocessing must mirror the OOP notebook exactly so results can be compared
- Notebook must run top-to-bottom without errors
_ Make sure each insights is shown in each data visualization and analysis. 