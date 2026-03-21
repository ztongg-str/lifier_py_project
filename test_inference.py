import sys
import os
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    preprocessor = joblib.load('output/preprocessor.pkl')
    print("Preprocessor type:", type(preprocessor))
    
    model = joblib.load('output/best_model_ranked.pkl')
    print("Model type:", type(model))

    # Test dummy data
    input_data = {
        'age': 45,
        'heart_rate': 80,
        'systolic_bp': 120,
        'temperature_c': 37.0,
        'spo2': 98,
        'news2_score': 0,
        'respiratory_rate': 16,
        'shock_index': 0.67,
        'chief_complaint': "Chest pain"
    }
    input_df = pd.DataFrame([input_data])
    
    # Try preprocessing
    if hasattr(preprocessor, 'build'):
        processed = preprocessor.build(input_df)
    else:
        processed = preprocessor.transform(input_df)
    
    print("Processed shape:", processed.shape)
    
    # Try prediction
    print("Prediction:", model.predict(processed))
    if hasattr(model, 'predict_proba'):
        print("Proba:", model.predict_proba(processed)[0])
    elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
        print("Proba (via .model):", model.model.predict_proba(processed)[0])

except Exception as e:
    import traceback
    traceback.print_exc()
