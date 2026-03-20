# This file is for configuring variables that will be static and used throughout the project.

import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..","data")
IMG_PATH = os.path.join(os.path.dirname(__file__), "..","image")
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
# Constants for data environment
SEED = 42
# Columns to remove initially
TARGET = "triage_acuity"
PATIENT_ID = "patient_id"
# File names to read
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
N_FEATURES_SELECT = SELECT_K_BEST
TEST_SIZE = 0.2
LEAKAGE_COLS = ['ed_los_hours', 'disposition',"news2_score"]

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