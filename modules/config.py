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

# Visualization
SAVE_FIG = True