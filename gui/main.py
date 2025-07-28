# main.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier


model = XGBClassifier()
model.load_model(r"D:\course\Courses\NTI final project\model.cmb")
scaler = joblib.load(r"D:\course\Courses\NTI final project\scaler.pkl")

st.set_page_config(page_title="Water Potability Predictor", layout="centered")
st.title("💧 Water Potability Prediction")
st.markdown("Enter the water test values to check if it's safe to drink:")

features = {
    "ph": st.number_input("pH (6.5 - 8.5)", min_value=0.0, max_value=14.0, value=7.0),
    "Hardness": st.number_input("Hardness", min_value=0.0, max_value=400.0, value=200.0),
    "Solids": st.number_input("Solids (ppm)", min_value=0.0, max_value=70000.0, value=20000.0),
    "Chloramines": st.number_input("Chloramines", min_value=0.0, max_value=15.0, value=6.0),
    "Sulfate": st.number_input("Sulfate", min_value=0.0, max_value=600.0, value=330.0),
    "Conductivity": st.number_input("Conductivity", min_value=0.0, max_value=1000.0, value=400.0),
    "Organic_carbon": st.number_input("Organic Carbon", min_value=0.0, max_value=40.0, value=14.0),
    "Trihalomethanes": st.number_input("Trihalomethanes", min_value=0.0, max_value=150.0, value=70.0),
    "Turbidity": st.number_input("Turbidity", min_value=0.0, max_value=10.0, value=4.0)
}

if st.button("🔍 Predict Potability"):
    df_input = pd.DataFrame([features])

    df_input['is_ph_neutral'] = df_input['ph'].between(6.5, 8.5).astype(int)
    df_input['is_hard'] = (df_input["Hardness"] > 300).astype(int)
    df_input['carbon_ratio'] = df_input['Organic_carbon'] / (df_input['Solids'] + 1)
    df_input['sulfate_to_solids'] = df_input['Sulfate'] / (df_input['Solids'] + 1)
    df_input['ph_x_chloramines'] = df_input['ph'] * df_input['Chloramines']
    df_input['hardness_minus_ph'] = df_input['Hardness'] - df_input['ph']

    cols_order = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                  'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity',
                  'is_ph_neutral', 'is_hard', 'carbon_ratio', 'sulfate_to_solids',
                  'ph_x_chloramines', 'hardness_minus_ph']
    
    df_input = df_input[cols_order]
    input_scaled = scaler.transform(df_input)
    y_pred=model.predict(input_scaled)


    print(y_pred)

    if y_pred == 1:
        st.success("✅ The water is **Potable** (Safe to Drink) 💧")
    else:
        st.error("❌ The water is **Not Potable** (Unsafe to Drink) ⚠️")
