import streamlit as st
import joblib
import numpy as np
import sys

# Load the trained model and scaler
heart_model = joblib.load("heart_failure_model.pkl")
heart_scaler = joblib.load("scaler.pkl")



st.title("Heart Failure Prediction App by Aryan")
age = st.slider("Age", 20, 100, 50)
anaemia = st.selectbox("Anaemia", [0, 1])
cretinine = st.number_input("Creatinine Phosphokinase", 0, 10000, 500)
diabetes = st.radio("Diabetes", [0, 1], format_func= lambda x: "Yes" if x == 1 else "No")
ef = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=30)
hbp = st.radio("High Blood Pressure", [0, 1], format_func= lambda x: "Yes" if x == 1 else "No")
platelets = st.number_input("Platelets", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium", min_value=0, max_value=200, value=135)
sex = st.selectbox("sex (1=Male, 0=Female)", [0, 1])
smoking = st.selectbox("Smoking", [0, 1], format_func= lambda x: "Yes" if x == 1 else "No")
time = st.number_input("Time (Follow-up Period in days)", min_value=0, max_value=500, value=100)

if st.button("Predict"):
    heart_input = np.array([[age, anaemia, cretinine, diabetes, ef, hbp, platelets, 
                            serum_creatinine, serum_sodium, sex, smoking, time]])
    scaled = heart_scaler.transform(heart_input)
    result = heart_model.predict(scaled)
    if result[0] == 1:
        st.success("The patient is likely to experience a heart failure event.")
    else:       
        st.success("The patient is safe.")
    