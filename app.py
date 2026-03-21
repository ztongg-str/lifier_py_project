# demo/app.py
import gradio as gr
import pandas as pd
import numpy as np
import os
import sys
import pickle
import joblib
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.preprocessing import (
    NumericalPreprocessor,
    CategoricalPreprocessor,
    TextPreprocessor,
    FeatureBuilder
)
from modules.models import ModelTrainer
from modules.config import *

class TriagePredictor:
    """
    Triage prediction model loader and predictor.
    Loads saved model and preprocessor for inference.
    """
    
    def __init__(self, model_path="output/best_model_ranked.pkl", data_path="output/preprocessor.pkl"):
        """
        Initialize predictor by loading model and preprocessor.
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.data_path = data_path
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessor from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}")
            
            if os.path.exists(self.data_path):
                self.preprocessor = joblib.load(self.data_path)
                print(f"Preprocessor loaded from {self.data_path}")
            else:
                print(f"Warning: Preprocessor file not found at {self.data_path}")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
    
    def predict(self, input_data):
        """
        Predict triage acuity for input data.
        """
        if self.model is None or self.preprocessor is None:
            return "Model not loaded", 0.0
        
        try:
            input_df = pd.DataFrame([input_data])
            # Preprocess input data
            processed_df = self.preprocessor.transform(input_df)  
            # Predict
            prediction = self.model.predict(processed_df)[0]
            # Get prediction probability
            probabilities = self.model.predict_proba(processed_df)[0]
            confidence = np.max(probabilities)
            acuity_map = {1: "Immediate", 2: "Very Urgent", 3: "Urgent", 4: "Semi-Urgent", 5: "Non-Urgent"}
            return acuity_map.get(prediction, f"Level {prediction}"), confidence
        except Exception as e:
            return f"Prediction error: {str(e)}", 0.0


def load_sample_data(dataset_name):
    """
    Load sample data from demo_data/ directory.
    """
    data_dir = "demo_data"
    file_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(file_path):
        return pd.DataFrame()
    if dataset_name.endswith('.csv'):
        return pd.read_csv(file_path)
    elif dataset_name.endswith('.pkl'):
        return pd.read_pickle(file_path)
    else:
        return pd.DataFrame()

def create_sample_options():
    """
    Create sample data options for the dropdown.
    """
    data_dir = "demo_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.pkl'))]
    return sorted(files)

def predict_from_sample(sample_file, predictor):
    """
    Predict triage acuity for all records in a sample file.
    """
    if not sample_file:
        return pd.DataFrame(), "No sample file selected."
    df = load_sample_data(sample_file)
    if df.empty:
        return pd.DataFrame(), f"Could not load {sample_file}"
    results = []
    for idx, row in df.iterrows():
        input_dict = row.to_dict()
        acuity, confidence = predictor.predict(input_dict)
        results.append({
            'record_id': idx,
            'predicted_acuity': acuity,
            'confidence': f"{confidence:.2%}",
            'acuity_level': 1 if acuity == "Immediate" else 2 if acuity == "Very Urgent" else 3 if acuity == "Urgent" else 4 if acuity == "Semi-Urgent" else 5
        })
    results_df = pd.DataFrame(results)
    
    summary = f"Processed {len(results_df)} records from {sample_file}\n"
    if not results_df.empty:
        summary += f"\nAcuity Distribution:\n{results_df['predicted_acuity'].value_counts().to_string()}"
    
    return results_df, summary


def predict_single_input(age, heart_rate, systolic_bp, temperature, spo2, news2_score, 
                         respiratory_rate, shock_index, chief_complaint, predictor):
    """
    Predict triage acuity for single patient input.
    """
    input_data = {
        'age': age,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'temperature_c': temperature,
        'spo2': spo2,
        'news2_score': news2_score,
        'respiratory_rate': respiratory_rate,
        'shock_index': shock_index,
        'chief_complaint': chief_complaint
    }
    
    acuity, confidence = predictor.predict(input_data)
    return acuity, f"{confidence:.2%}"


def create_demo():
    """
    Create and launch Gradio demo interface.
    """
    predictor = TriagePredictor()
    
    with gr.Blocks(title="Triage Prediction System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Emergency Department Triage Prediction System")
        gr.Markdown("Predict patient acuity level based on clinical features and chief complaint.")
        
        with gr.Tab("Manual Entry"):
            gr.Markdown("### Enter Patient Data for Prediction")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Vital Signs")
                    manual_age = gr.Number(label="Age (years)", value=45, step=1)
                    manual_heart_rate = gr.Number(label="Heart Rate (bpm)", value=80, step=1)
                    manual_systolic_bp = gr.Number(label="Systolic BP (mmHg)", value=120, step=1)
                    manual_temperature = gr.Number(label="Temperature (°C)", value=37.0, step=0.1)
                    manual_spo2 = gr.Number(label="SpO2 (%)", value=98, step=1)
                    manual_respiratory_rate = gr.Number(label="Respiratory Rate", value=16, step=1)
                    manual_shock_index = gr.Number(label="Shock Index", value=0.67, step=0.01)
                    
                with gr.Column():
                    gr.Markdown("#### Clinical Scores & Complaint")
                    manual_news2 = gr.Number(label="NEWS2 Score", value=0, step=1)
                    manual_chief_complaint = gr.Textbox(label="Chief Complaint", lines=3, placeholder="e.g., Chest pain, Shortness of breath")
                    
            manual_predict_btn = gr.Button("Predict Acuity", variant="primary")
            manual_acuity = gr.Textbox(label="Predicted Acuity Level", interactive=False)
            manual_confidence = gr.Textbox(label="Confidence", interactive=False)
            
            manual_predict_btn.click(
                fn=predict_single_input,
                inputs=[manual_age, manual_heart_rate, manual_systolic_bp, manual_temperature, 
                        manual_spo2, manual_news2, manual_respiratory_rate, manual_shock_index,
                        manual_chief_complaint, gr.State(predictor)],
                outputs=[manual_acuity, manual_confidence]
            )
        
        with gr.Tab("Select Data"):
            with gr.Row():
                with gr.Column():
                    sample_files = create_sample_options()
                    sample_selector = gr.Dropdown(
                        choices=sample_files,
                        label="Select Demo Data File",
                        value=sample_files[0] if sample_files else None,
                        info="Available demo data files from demo_data/ folder"
                    )
                    load_btn = gr.Button("Load and Predict", variant="primary")
                    
                with gr.Column():
                    results_table = gr.Dataframe(label="Prediction Results", interactive=False)
                    summary_text = gr.Textbox(label="Summary", lines=5, interactive=False)
            
            load_btn.click(
                fn=predict_from_sample,
                inputs=[sample_selector, gr.State(predictor)],
                outputs=[results_table, summary_text]
            )
        
        gr.Markdown("---")
        gr.Markdown("**Acuity Levels:** 1=Immediate, 2=Very Urgent, 3=Urgent, 4=Semi-Urgent, 5=Non-Urgent") 
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)