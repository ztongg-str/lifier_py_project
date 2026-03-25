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

# File names
CHIEF_COMPLAINT_FILENAME = "chief_complaints.csv"
PATIENT_HISTORY_FILENAME = "patient_history.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"
ESI_DICT = os.path.join(DATA_DIR,'esi_dictionary.json')

# Model parameters
TFIDF_MAX_FEATURES = 50
SMOTE_K_NEIGHBORS = 3

# Additional defaults used by OOP pipeline
TEST_SIZE = 0.2
LEAKAGE_COLS = ['ed_los_hours', 'disposition','news2_score',"patient_id"]

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

XG_PARAMS = {
    'n_estimators': 100,
    'random_state': SEED,
    "objective":'multi:softmax',
    "num_class":5
}

# Visualization
SAVE_FIG = True