# OOP Triage Prediction

## About the Project

This project focuses on predicting the triage acuity level of patients entering an emergency department. The goal is to prioritize patients effectively based on urgency, helping healthcare providers make faster and more accurate decisions.

The system uses machine learning models trained on patient data to classify acuity levels, improving efficiency and supporting better healthcare outcomes.

### Key Objectives

- Predict patient triage levels (acuity)
- Support decision-making in emergency departments
- Compare multiple machine learning models
- Provide a reusable and scalable OOP-based machine learning pipeline

### Data Sources

The model is trained using the following datasets:

- `patient_history.csv`
- `train.csv`
- `chief_complain.csv`

---

## Project Structure

```
.
├── modules/
│   ├── __pycache__/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── data/
│   ├── patient_history.csv
│   ├── train.csv
│   └── chief_complain.csv
│
├── demo_data/
│
├── image/
│
├── output/
│   ├── best_model_ranked.pkl
│   └── preprocessor.pkl
│
├── app.py
├── y2-t2.ipynb
├── NOTE.md
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Set Up the Environment

Create and activate a Python virtual environment (Python 3.8 or higher recommended):

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run the Training Pipeline

1. Open Jupyter Notebook:

```bash
jupyter notebook
```

2. Open and run `y2-t2.ipynb`.

What the notebook performs:

- Loads and preprocesses data via `data_loader.py` and `preprocessing.py`
- Engineers and selects features
- Trains models defined in `models.py` (Logistic Regression, Random Forest, XGBoost)
- Evaluates models using F1-score via `evaluation.py`
- Generates performance visualizations via `visualization.py`
- Saves the best model and preprocessor to the `output/` directory

> Note: Training time may vary depending on dataset size and available hardware.

### Run the Prediction Application

After training is complete, launch the prediction app:

```bash
python app.py
```

Prediction options available:

- Manual input of patient data
- Pre-configured sample data loaded from `demo_data/`

---

## Module Overview

| Module | Description |
|---|---|
| `config.py` | Central configuration for paths, hyperparameters, and settings |
| `data_loader.py` | Handles loading and merging of raw data sources |
| `preprocessing.py` | Manages missing values, scaling, encoding, and text processing |
| `models.py` | Defines Logistic Regression, Random Forest, and XGBoost classifiers |
| `evaluation.py` | Computes F1-score and generates comparison metrics |
| `visualization.py` | Produces plots and performance charts |

---

## Model Performance

| Model | Evaluation Metric |
|---|---|
| Logistic Regression | F1 Score |
| Random Forest | F1 Score |
| XGBoost | F1 Score (Best) |

The best-performing model (XGBoost) is saved as `output/best_model_ranked.pkl` and the fitted preprocessor is saved as `output/preprocessor.pkl`.

---

## Notes

- Ensure the project root directory is included in your Python path when running notebooks.
- All modules depend on settings defined in `modules/config.py`. Verify configuration paths before running.
- Training duration may vary based on dataset size and hardware specifications.

---

## Future Improvements

- Deploy the system as a web-based application
- Integrate real-time hospital data feeds
- Improve model interpretability with explainability tools (e.g., SHAP)
- Optimize the pipeline for production-grade performance
