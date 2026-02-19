import streamlit as st
import pandas as pd
import joblib

# Set page config for a professional look
st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Load Saved Files ------------------ #
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("Logistic_Regg_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, expected_columns = load_assets()

if model is None:
    st.stop()

# ------------------ UI Header ------------------ #
st.title("❤️ Heart Disease Diagnostic AI")
st.markdown("---")

# ------------------ Input Sections ------------------ #
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Patient Bio")
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

    with col2:
        st.subheader("🩺 Clinical Metrics")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

st.markdown("---")

with st.container():
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Test Results")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

    with col4:
        st.subheader("📉 Advanced Metrics")
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write(" ") # Spacer

# ------------------ Prediction Logic ------------------ #
if st.button("RUN DIAGNOSTIC ANALYSIS"):
    
    # Feature Engineering
    input_data = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak
    }
    
    # Map categoricals (keeping your logic intact)
    input_data[f"Sex_{sex}"] = 1
    input_data[f"ChestPainType_{chest_pain}"] = 1
    input_data[f"RestingECG_{resting_ecg}"] = 1
    input_data[f"ExerciseAngina_{exercise_angina}"] = 1
    input_data[f"ST_Slope_{st_slope}"] = 1

    input_df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(scaled_input)[0]
    # Probability (if your model supports it)
    prob = model.predict_proba(scaled_input)[0][1] * 100

    st.markdown("---")
    
    if prediction == 1:
        st.error(f"### ⚠️ Result: High Risk ({prob:.1f}%)")
        st.warning("Based on the input data, there is a significant indication of heart disease. Please consult a medical professional.")
    else:
        st.success(f"### ✅ Result: Low Risk ({100-prob:.1f}%)")
        st.balloons()
        st.info("The analysis suggests a lower risk of heart disease. Maintain a healthy lifestyle!")