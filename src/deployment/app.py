import os
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download
import numpy as np

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🛠️", layout="wide")
st.title("🚗 Engine Predictive Maintenance – Failure Risk Prediction")

# Get model repo from environment or use default
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "arnavarpit/engine-predictive-maintenance-sklearn")
MODEL_FILE = "model.joblib"

@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face Hub"""
    try:
        with st.spinner("🔄 Loading model from Hugging Face Hub..."):
            local_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILE)
            model = joblib.load(local_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info(f"Attempting to load from: {HF_MODEL_REPO}")
        st.stop()
        return None

# Load model
model = load_model()

# Show model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Model Repository:** {HF_MODEL_REPO}")
    st.write(f"**Model File:** {MODEL_FILE}")
    if model:
        st.write(f"**Model Type:** {type(model).__name__}")


# Input Section
st.markdown("### 📊 Enter Engine Sensor Readings")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pressure & RPM Readings**")
    engine_rpm = st.number_input("Engine RPM", value=800.0, min_value=0.0, max_value=5000.0, step=10.0, help="Engine rotations per minute")
    lub_oil_pressure = st.number_input("Lube Oil Pressure (bar)", value=3.0, min_value=0.0, max_value=20.0, step=0.1, help="Lubrication oil pressure")
    fuel_pressure = st.number_input("Fuel Pressure (bar)", value=6.0, min_value=0.0, max_value=20.0, step=0.1, help="Fuel system pressure")
    coolant_pressure = st.number_input("Coolant Pressure (bar)", value=2.0, min_value=0.0, max_value=20.0, step=0.1, help="Cooling system pressure")

with col2:
    st.markdown("**Temperature Readings**")
    lub_oil_temperature = st.number_input("Lube Oil Temp (°C)", value=80.0, min_value=-50.0, max_value=200.0, step=0.5, help="Lubrication oil temperature")
    coolant_temperature = st.number_input("Coolant Temp (°C)", value=80.0, min_value=-50.0, max_value=200.0, step=0.5, help="Coolant temperature")

st.markdown("---")

# Display current readings
st.markdown("### 📋 Current Sensor Summary")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Engine RPM", f"{engine_rpm:.0f}")
    st.metric("Lube Oil Pressure", f"{lub_oil_pressure:.1f} bar")
with col_b:
    st.metric("Fuel Pressure", f"{fuel_pressure:.1f} bar")
    st.metric("Coolant Pressure", f"{coolant_pressure:.1f} bar")
with col_c:
    st.metric("Lube Oil Temp", f"{lub_oil_temperature:.1f} °C")
    st.metric("Coolant Temp", f"{coolant_temperature:.1f} °C")

st.markdown("---")

# Prediction Section
st.markdown("### 🔮 Maintenance Prediction")
if st.button("🔍 Predict Maintenance Need", type="primary", use_container_width=True):
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        # Prepare input data
        X = np.array([[engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure,
                       lub_oil_temperature, coolant_temperature]])

        try:
            with st.spinner("Analyzing sensor data..."):
                proba = model.predict_proba(X)[:, 1][0]
                pred = model.predict(X)[0]

            # Display results
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                if pred == 1:
                    st.error("### ⚠️ MAINTENANCE REQUIRED")
                    st.warning("The engine shows signs of potential failure. Schedule maintenance immediately.")
                else:
                    st.success("### ✅ NORMAL OPERATION")
                    st.info("The engine is operating within normal parameters.")
            
            with result_col2:
                st.metric("Failure Risk Score", f"{proba:.1%}", delta=None)
                
                # Risk level indicator with color coding
                if proba < 0.3:
                    st.success("🟢 **Low Risk** - Continue normal operations")
                elif proba < 0.7:
                    st.warning("🟡 **Medium Risk** - Monitor closely")
                else:
                    st.error("🔴 **High Risk** - Immediate attention required")

            # Additional insights
            st.markdown("---")
            st.markdown("#### 📈 Risk Analysis")
            st.progress(proba)
            
            if proba > 0.5:
                st.markdown("""
                **Recommended Actions:**
                - Schedule comprehensive engine inspection
                - Check lubrication and cooling systems
                - Review sensor readings for anomalies
                - Prepare maintenance resources
                """)
            else:
                st.markdown("""
                **Current Status:**
                - All systems operating normally
                - Continue regular monitoring
                - Next scheduled maintenance as per routine
                """)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

st.markdown("---")
st.markdown("### 💡 About This Application")
st.info("""
This application uses machine learning to predict engine maintenance needs based on real-time sensor data. 
The model analyzes six key engine parameters to assess failure risk and provide maintenance recommendations.

**Features:**
- Real-time failure risk prediction
- Interactive sensor input controls
- Visual risk level indicators
- Maintenance recommendations

**Model Source:** The trained model is loaded directly from the Hugging Face Model Hub.
""")

st.caption(f"🤗 Model: `{HF_MODEL_REPO}` | Built with Streamlit")
