# config.py
# Configuration settings for the project

import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..","data")
IMG_PATH = os.path.join(os.path.dirname(__file__), "..","image")
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)

# Constants
SEED = 42
TARGET = "triage_acuity"
PATIENT_ID = "patient_id"

# File names
CHIEF_COMPLAINT_FILENAME = "chief_complaints.csv"
PATIENT_HISTORY_FILENAME = "patient_history.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

# Model parameters
TFIDF_MAX_FEATURES = 100
SELECT_K_BEST = 50
SMOTE_K_NEIGHBORS = 5

# Additional defaults used by OOP pipeline
BP_COLS = ['systolic_bp', 'diastolic_bp']
KEY_VITALS = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2', 'respiratory_rate']
N_FEATURES_SELECT = SELECT_K_BEST
TEST_SIZE = 0.2
LEAKAGE_COLS = ['ed_los_hours', 'disposition']

# Model hyperparameters for ModelTrainer
LR_PARAMS = {
    'random_state': SEED,
    'max_iter': 1000,
    'class_weight': 'balanced'
}

RF_PARAMS = {
    'n_estimators': 100,
    'random_state': SEED,
    'class_weight': 'balanced',
    'n_jobs': -1
}

GB_PARAMS = {
    'n_estimators': 100,
    'random_state': SEED,
    'learning_rate': 0.1,
    'max_depth': 5
}

# Visualization
SAVE_FIG = True