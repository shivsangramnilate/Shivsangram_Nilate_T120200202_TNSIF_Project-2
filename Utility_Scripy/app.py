import streamlit as st
from utility_scripts.processor import make_prediction, MODEL_LOADED
import numpy as np

# --- 1. Configuration and Title ---
st.set_page_config(
    page_title="Manufacturing Output Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Capstone Project: Manufacturing Output Prediction")
st.markdown("---")

# Display a prominent error if the model failed to load
if not MODEL_LOADED:
    st.error("MODEL FAILED TO LOAD. Please check your terminal/console. Ensure **'latest_model.pkl'** is correctly placed in the `backend_model` folder.")
    st.stop()
    
st.markdown("### Predict Parts Per Hour")
st.markdown("Enter machine parameters below. **Minimum input is required for Injection Temp, Pressure, and Cycle Time** to start. Missing optional values will be imputed by the model.")

# --- 2. Form Structure and Input Fields (Collecting all 18 features) ---

# The Streamlit form will collect data and send it to the prediction function
with st.form("prediction_form"):
    
    # --- A. Key Operating Parameters (Highly Recommended) ---
    st.subheader("1. Key Operating Parameters (Highly Recommended)")
    col1, col2, col3 = st.columns(3)
    
    # Use st.text_input for required fields to ensure user enters SOMETHING
    with col1:
        inj_temp = st.text_input("Injection Temperature (°C)", value="220.5", help="Required. e.g., 220.5")
    with col2:
        inj_pressure = st.text_input("Injection Pressure (Bar)", value="1500.0", help="Required. e.g., 1500.0")
    with col3:
        cycle_time = st.text_input("Cycle Time (s)", value="45.2", help="Required. e.g., 45.2")
        
    st.markdown("---")

    # --- B. Secondary Numeric Data (OPTIONAL) ---
    st.subheader("2. Secondary & Contextual Data (Optional)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cooling_time = st.number_input("Cooling Time (s)", value=None, help="e.g., 12.0", step=0.1, format="%.1f")
        material_viscosity = st.number_input("Material Viscosity", value=None, help="e.g., 0.85", step=0.01, format="%.2f")
        machine_age = st.number_input("Machine Age (Years)", value=None, help="e.g., 5.0", step=0.1, format="%.1f")
        
    with col2:
        ambient_temp = st.number_input("Ambient Temp (°C)", value=None, help="e.g., 24.5", step=0.1, format="%.1f")
        operator_exp = st.number_input("Operator Exp. (Years)", value=None, help="e.g., 7.5", step=0.1, format="%.1f")
        maintenance_hours = st.number_input("Maintenance Hours (Recent)", value=None, help="e.g., 3.0", step=0.1, format="%.1f")

    with col3:
        temp_pressure_ratio = st.number_input("Temp/Pressure Ratio", value=None, help="e.g., 0.75", step=0.01, format="%.2f")
        total_cycle_time = st.number_input("Total Cycle Time (s)", value=None, help="e.g., 60.0", step=0.1, format="%.1f")
        machine_utilization = st.number_input("Machine Utilization (0-1)", value=None, help="e.g., 0.90", step=0.01, format="%.2f")
        parts_per_hour_placeholder = st.number_input("Parts Per Hour (Placeholder)", value=None, disabled=True)
        
    st.markdown("---")

    # --- C. Categorical Data (OPTIONAL) ---
    st.subheader("3. Categorical & Time Data (Optional)")
    col1, col2, col3 = st.columns(3)

    with col1:
        shift = st.selectbox("Shift", options=[None, "Day", "Night", "Evening"], index=0, format_func=lambda x: x if x else 'Select Option')
        machine_type = st.selectbox("Machine Type", options=[None, "Type_A", "Type_B", "Type_C"], index=0, format_func=lambda x: x if x else 'Select Option')
    with col2:
        material_grade = st.selectbox("Material Grade", options=[None, "Economy", "Standard", "Premium"], index=0, format_func=lambda x: x if x else 'Select Option')
        day_of_week = st.selectbox("Day of Week", options=[None, "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0, format_func=lambda x: x if x else 'Select Option')
    with col3:
        timestamp = st.text_input("Timestamp (DD-MM-YYYY HH:MM)", value="", help="e.g., 15-10-2025 10:00")
        
    
    # --- D. Submission Button ---
    submitted = st.form_submit_button("Generate Prediction", type="primary")

    if submitted:
        # Simple validation check for the three highly recommended fields
        if not inj_temp or not inj_pressure or not cycle_time:
            st.error("Please enter values for Injection Temperature, Injection Pressure, and Cycle Time.")
            st.stop()
            
        # 1. Collect all data into a dictionary (Parsing text inputs to float)
        try:
            input_data = {
                'Injection_Temperature': float(inj_temp),
                'Injection_Pressure': float(inj_pressure),
                'Cycle_Time': float(cycle_time),
                'Cooling_Time': cooling_time,
                'Material_Viscosity': material_viscosity,
                'Ambient_Temperature': ambient_temp,
                'Machine_Age': machine_age,
                'Operator_Experience': operator_exp,
                'Maintenance_Hours': maintenance_hours,
                'Temperature_Pressure_Ratio': temp_pressure_ratio,
                'Total_Cycle_Time': total_cycle_time,
                'Machine_Utilization': machine_utilization,
                'Parts_Per_Hour': parts_per_hour_placeholder,
                'Timestamp': timestamp if timestamp else None, # Convert empty string to None
                'Shift': shift,
                'Machine_Type': machine_type,
                'Material_Grade': material_grade,
                'Day_of_Week': day_of_week
            }
        except ValueError:
             st.error("Error: Key operating parameters must be valid numbers.")
             st.stop()


        # 2. Make Prediction
        with st.spinner('Calculating prediction...'):
            prediction_result = make_prediction(input_data)
        
        # 3. Display Result
        if not np.isnan(prediction_result):
            st.success(f"### Predicted Hourly Output: {prediction_result:.2f} Parts Per Hour")
            st.balloons()
        else:
            st.error("Prediction failed. Please check your **Command Prompt/Terminal** for the specific model loading or transformation error.")
