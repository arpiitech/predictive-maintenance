import os
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download
import numpy as np

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🛠️")
st.title("Engine Predictive Maintenance – Failure Risk Prediction")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
if not HF_MODEL_REPO:
    st.error("HF_MODEL_REPO environment variable is not set. Please configure it in your Space settings.")
    st.stop()

MODEL_FILE = "model.joblib"

@st.cache_resource
def load_model():
    local_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(local_path)

model = load_model()

st.sidebar.header("Engine Sensor Readings")

col1, col2 = st.columns(2)
engine_rpm = col1.number_input("Engine RPM", value=800.0, step=10.0, help="Engine rotations per minute")
lub_oil_pressure = col1.number_input("Lube Oil Pressure (bar)", value=3.0, step=0.1, help="Lubrication oil pressure")
fuel_pressure = col1.number_input("Fuel Pressure (bar)", value=6.0, step=0.1, help="Fuel system pressure")
coolant_pressure = col1.number_input("Coolant Pressure (bar)", value=2.0, step=0.1, help="Cooling system pressure")

lub_oil_temperature = col2.number_input("Lube Oil Temp (°C)", value=80.0, step=0.5, help="Lubrication oil temperature")
coolant_temperature = col2.number_input("Coolant Temp (°C)", value=80.0, step=0.5, help="Coolant temperature")

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Current Sensor Readings")
    st.write(f"**Engine RPM:** {engine_rpm}")
    st.write(f"**Lube Oil Pressure:** {lub_oil_pressure} bar")
    st.write(f"**Fuel Pressure:** {fuel_pressure} bar")
    st.write(f"**Coolant Pressure:** {coolant_pressure} bar")
    st.write(f"**Lube Oil Temperature:** {lub_oil_temperature} °C")
    st.write(f"**Coolant Temperature:** {coolant_temperature} °C")

with col_right:
    if st.button("Predict Maintenance Need", type="primary"):
        X = np.array([[engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure,
                       lub_oil_temperature, coolant_temperature]])

        try:
            proba = model.predict_proba(X)[:, 1][0]
            pred = model.predict(X)[0]

            if pred == 1:
                st.error("⚠️ **MAINTENANCE REQUIRED**")
                st.metric("Failure Risk Score", f"{proba:.1%}")
            else:
                st.success("✅ **NORMAL OPERATION**")
                st.metric("Failure Risk Score", f"{proba:.1%}")

            # Risk level indicator
            if proba < 0.3:
                st.info("🟢 Low Risk")
            elif proba < 0.7:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("This app loads the latest model directly from the Hugging Face Model Hub.")
