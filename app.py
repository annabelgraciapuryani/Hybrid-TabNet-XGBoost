import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Hybrid TabNet-XGBoost",
    page_icon="🌧️",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    models = {
        "Hybrid TabNet-XGBoost":
            joblib.load("models/hybrid_xgb.pkl"),

        "Hybrid TabNet-XGBoost (Extreme)":
            joblib.load("models/hybrid_extreme.pkl"),

        "Hybrid TabNet-SVR":
            joblib.load("models/hybrid_svr.pkl")
    }

    return models


models = load_models()


# =====================================================
# LOAD EVALUATION DATA
# =====================================================

@st.cache_data
def load_evaluation():

    return {

        "Hybrid TabNet-XGBoost":
            pd.read_csv("data/hasil_evaluasi.csv"),

        "Hybrid TabNet-XGBoost (Extreme)":
            pd.read_csv("data/hasil_evaluasi_extreme.csv"),

        "Hybrid TabNet-SVR":
            pd.read_csv("data/hasil_evaluasi_svr.csv")

    }


evaluation_data = load_evaluation()

# =====================================================
# SESSION
# =====================================================

if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(

    "Menu",

    [

        "🏠 Home",

        "🌧 Prediction",

        "📊 Model Evaluation",

        "📥 Download"

    ]

)
# =====================================================
# HOME
# =====================================================

if menu == "🏠 Home":

    st.title("🌧 Daily Rainfall Prediction")

    st.markdown("""
### Hybrid TabNet–XGBoost Based on Radiosonde Data

Aplikasi ini digunakan untuk memprediksi intensitas curah hujan harian
menggunakan model **Hybrid TabNet–XGBoost** berbasis
data pengamatan radiosonde dan meteorologi permukaan.

Model yang tersedia:

- Hybrid TabNet–XGBoost
- Hybrid TabNet–XGBoost (Extreme)
- Hybrid TabNet–SVR

Dataset yang digunakan berupa data meteorologi harian
BMKG Stasiun Meteorologi Kelas I Juanda.

---
""")

    col1,col2,col3=st.columns(3)

    col1.info("📂 Upload Dataset Excel")

    col2.info("🤖 Pilih Model")

    col3.info("📈 Prediksi Curah Hujan")

    st.success("Silakan pilih menu Prediction untuk memulai prediksi.")
