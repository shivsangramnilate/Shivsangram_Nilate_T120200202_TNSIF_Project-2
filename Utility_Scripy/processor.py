import joblib
import pandas as pd
import numpy as np
import os
import sys

# *****************************************************************
# SCALABILITY/COMPATIBILITY FIX: Import all necessary components 
# from scikit-learn here. This is necessary to ensure joblib/pickle 
# can recognize and load objects like ColumnTransformer and Pipeline.
# *****************************************************************
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
# *****************************************************************

# This is the exact list of features (X data) the model expects.
# FIX APPLIED: Removed the target variable 'Parts_Per_Hour' from the input feature list.
EXPECTED_FEATURES = [
    'Timestamp', 'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time', 
    'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature', 'Machine_Age', 
    'Operator_Experience', 'Maintenance_Hours', 'Shift', 'Machine_Type', 
    'Material_Grade', 'Day_of_Week', 'Temperature_Pressure_Ratio', 
    'Total_Cycle_Time', 'Efficiency_Score', 'Machine_Utilization' # Parts_Per_Hour removed
]

# --- Model Loading (Executed Once on App Startup) ---
# Construct the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', "backend_model/latest_model.pkl")

MODEL_LOADED = False
try:
    # Check if file exists before attempting to load
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at expected path: {MODEL_PATH}")
        
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
    PREPROCESSOR = MODEL_BUNDLE['preprocessor']
    MODEL = MODEL_BUNDLE['model']
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    # Print a user-friendly error message to the console
    print(f"\n--- CRITICAL MODEL LOADING ERROR ---")
    print(f"ERROR: Model load failed. Please ensure your trained model ('iterationN_model.pkl') ")
    print(f"was copied to the 'backend_model' folder and renamed to 'latest_model.pkl'.")
    print(f"Specific Error: {e}")
    # Print the specific traceback for the user to see the full error in the terminal
    import traceback
    traceback.print_exc() 


def make_prediction(input_data: dict) -> float:
    """
    Takes raw input data (from Streamlit form) and returns the prediction.
    Explicitly converts None values to np.nan for safe imputation.
    """
    if not MODEL_LOADED:
        return np.nan 

    # 1. Convert None values to numpy.nan before creating the DataFrame.
    # This ensures consistency for the SimpleImputer.
    cleaned_input = {k: v if v is not None else np.nan for k, v in input_data.items()}
    
    # 2. Convert input dictionary to DataFrame
    df = pd.DataFrame([cleaned_input]).reindex(columns=EXPECTED_FEATURES)
    
    # 3. Handle Timestamp (Converts None/NaN/string to numeric format, coercing errors to NaN)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True).astype('int64') // 10**9
    
    # 4. Preprocess and predict
    try:
        # Preprocessor handles imputation, scaling, and encoding 
        X_scaled = PREPROCESSOR.transform(df)
        prediction = MODEL.predict(X_scaled)
        return float(prediction[0])
    except Exception as e:
        # This catches errors during transformation (e.g., if a feature column is entirely missing)
        print(f"\n--- Prediction Error ---")
        print(f"Error during data transformation or prediction: {e}")
        return np.nan
