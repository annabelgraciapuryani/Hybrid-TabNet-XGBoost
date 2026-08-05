import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Hybrid TabNet-XGBoost",
    page_icon="🌧️",
    layout="wide"
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

@st.cache_resource
def load_models():

    return {

        "Hybrid TabNet-XGBoost":{

            "model":joblib.load("models/hybrid_xgb.pkl"),
            "scaler":joblib.load("models/scaler_xgb.pkl")

        },

        "Hybrid TabNet-XGBoost (Extreme)":{

            "model":joblib.load("models/hybrid_extreme.pkl"),
            "scaler":joblib.load("models/scaler_extreme.pkl")

        },

        "Hybrid TabNet-SVR":{

            "model":joblib.load("models/hybrid_svr.pkl"),
            "scaler":joblib.load("models/scaler_svr.pkl")

        }

    }

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
            pd.read_csv("data/hasil_evaluasi_ektrem.csv"),

        "Hybrid TabNet-SVR":
            pd.read_csv("data/hasil_evaluasi_svr.csv")

    }

evaluation_data = load_evaluation()

# =====================================================
# FEATURE MODEL
# =====================================================

FEATURES = [

    "Humi0",
    "WS",
    "KI",
    "Press0",
    "LFC",
    "SI",
    "CCL",
    "500",
    "TT",
    "850",
    "LCL",
    "Height",
    "TPW",
    "700",
    "CAPE",

    "Humi0_lag1",
    "WS_lag1",
    "KI_lag1",
    "Press0_lag1",
    "LFC_lag1",
    "SI_lag1",
    "CCL_lag1",
    "500_lag1",
    "TT_lag1",
    "850_lag1",
    "LCL_lag1",
    "Height_lag1",
    "TPW_lag1",
    "700_lag1",
    "CAPE_lag1",

    "CH_lag1"

]

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

## Hybrid TabNet–XGBoost Based on Radiosonde Data

Aplikasi ini dikembangkan untuk memprediksi intensitas curah hujan harian menggunakan model **Hybrid TabNet–XGBoost** berbasis data pengamatan radiosonde dan meteorologi permukaan dari BMKG Stasiun Meteorologi Kelas I Juanda.

### Model yang tersedia

- Hybrid TabNet–XGBoost
- Hybrid TabNet–XGBoost (Extreme)
- Hybrid TabNet–SVR

### Alur Penggunaan

1. Upload dataset (.xlsx)
2. Pilih model prediksi
3. Jalankan proses prediksi
4. Lihat hasil evaluasi
5. Download hasil prediksi

""")

    col1, col2, col3 = st.columns(3)

    col1.info("📂 Upload Dataset")
    col2.info("🤖 Prediction")
    col3.info("📈 Evaluation")

    st.success("Silakan pilih menu Prediction untuk memulai proses prediksi.")

# =====================================================
# PREDICTION
# =====================================================

elif menu == "🌧 Prediction":

    st.title("🌧 Rainfall Prediction")

    model_name = st.selectbox(

        "Select Prediction Model",

        [

            "Hybrid TabNet-XGBoost",

            "Hybrid TabNet-XGBoost (Extreme)",

            "Hybrid TabNet-SVR"

        ]

    )

    uploaded = st.file_uploader(

        "Upload Dataset (.xlsx)",

        type=["xlsx"]

    )

    if uploaded:

        df = pd.read_excel(uploaded)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        if st.button("Predict"):

            df = df.copy()

            # ==========================
            # Membuat fitur lag
            # ==========================

            lag_features = [

                "Humi0",
                "WS",
                "KI",
                "Press0",
                "LFC",
                "SI",
                "CCL",
                "500",
                "TT",
                "850",
                "LCL",
                "Height",
                "TPW",
                "700",
                "CAPE"

            ]

            for col in lag_features:

                df[f"{col}_lag1"] = df[col].shift(1)

            df["CH_lag1"] = df["CH"].shift(1)

            df = df.dropna().reset_index(drop=True)

            X = df[FEATURES]

            scaler = models[model_name]["scaler"]

            X_scaled = scaler.transform(X)

            model = models[model_name]["model"]

            prediction = model.predict(X_scaled)

            result = pd.DataFrame({

                "Tanggal":df["Tanggal"],

                "Actual":df["CH"],

                "Prediction":prediction

            })

            st.session_state.prediction_result = result

            st.success("Prediction Completed Successfully!")

            st.dataframe(result.head())
