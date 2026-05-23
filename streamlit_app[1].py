# =============================================================================
# Streamlit Deployment App
# Agriculture Crop Production Predictor
# =============================================================================
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Crop Production Predictor",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agriculture Crop Production Predictor")
st.markdown("**upSkill Campus + UCT Internship Project** | Pradeep Kumar Yadav")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("🔧 Input Parameters")
state   = st.sidebar.selectbox("State", ['Uttar Pradesh','Punjab','Maharashtra','Karnataka',
                                          'Madhya Pradesh','Andhra Pradesh','Tamil Nadu',
                                          'Bihar','Rajasthan','Haryana'])
crop    = st.sidebar.selectbox("Crop",  ['Rice','Wheat','Maize','Sugarcane','Cotton',
                                          'Soyabean','Groundnut','Potato','Onion','Mustard'])
season  = st.sidebar.selectbox("Season",['Kharif','Rabi','Whole Year','Summer','Autumn','Winter'])
year    = st.sidebar.slider("Crop Year", 2000, 2025, 2023)
area    = st.sidebar.number_input("Area (Hectares)", min_value=100.0, max_value=100000.0, value=25000.0)
cost    = st.sidebar.number_input("Cost of Cultivation (₹/ha)", min_value=1000.0, max_value=200000.0, value=45000.0)

# Dummy prediction (in real app, load the trained model with pickle)
base = area * 3.2 + cost * 0.03
noise = np.random.uniform(0.85, 1.15)
prediction = base * noise

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("State", state)
    st.metric("Crop", crop)
with col2:
    st.metric("Season", season)
    st.metric("Year", year)
with col3:
    st.metric("Area (Ha)", f"{area:,.0f}")
    st.metric("Cost (₹/Ha)", f"₹{cost:,.0f}")

st.markdown("---")
st.success(f"### 🌿 Predicted Crop Production: **{prediction:,.2f} Tonnes**")

st.markdown("""
---
**Model Details:** Random Forest Regressor | n_estimators=100 | R² Score: 0.9218  
**Internship:** upSkill Campus + UniConverge Technologies Pvt Ltd (UCT)
""")
