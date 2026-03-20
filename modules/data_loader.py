# data_loader.py
# Module for loading data
import pandas as pd
import os
from modules.config import DATA_DIR, CHIEF_COMPLAINT_FILENAME, PATIENT_HISTORY_FILENAME, SAMPLE_SUBMISSION_FILENAME, TEST_FILENAME, TRAIN_FILENAME
def get_file_path(filename):
    return os.path.join(DATA_DIR, filename)
def load_data():
    try:
        chief_complaint_data = pd.read_csv(get_file_path(CHIEF_COMPLAINT_FILENAME))
        patient_history_data = pd.read_csv(get_file_path(PATIENT_HISTORY_FILENAME))
        train_data = pd.read_csv(get_file_path(TRAIN_FILENAME))
        test_data = pd.read_csv(get_file_path(TEST_FILENAME))
        sample_submission_data = pd.read_csv(get_file_path(SAMPLE_SUBMISSION_FILENAME))
    
        train_data = train_data.merge(chief_complaint_data[['patient_id', 'chief_complaint_raw']],on='patient_id', how='left')
        test_data = test_data.merge(chief_complaint_data[['patient_id', 'chief_complaint_raw']],on='patient_id', how='left')
        train_data = train_data.merge(patient_history_data, on='patient_id',how='left')
        test_data = test_data.merge(patient_history_data, on='patient_id',how='left')

        return chief_complaint_data, patient_history_data, train_data, test_data, sample_submission_data
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Empty data file: {e}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")