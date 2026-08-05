import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
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
# PATH
# =====================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    files = {

        "Hybrid TabNet-XGBoost":{

            "model":"hybrid_xgb.pkl",
            "scaler":"scaler_xgb.pkl"

        },

        "Hybrid TabNet-XGBoost (Extreme)":{

            "model":"hybrid_extreme.pkl",
            "scaler":"scaler_ekstrem.pkl"

        },

        "Hybrid TabNet-SVR":{

            "model":"hybrid_svr.pkl",
            "scaler":"scaler_svr.pkl"

        }

    }

    loaded = {}

    for name, f in files.items():

        model_path = MODEL_DIR / f["model"]
        scaler_path = MODEL_DIR / f["scaler"]

        if not model_path.exists():

            st.error(f"Model tidak ditemukan : {model_path}")
            st.stop()

        if not scaler_path.exists():

            st.error(f"Scaler tidak ditemukan : {scaler_path}")
            st.stop()

        loaded[name] = {

            "model":joblib.load(model_path),

            "scaler":joblib.load(scaler_path)

        }

    return loaded


models = load_models()

# =====================================================
# LOAD EVALUATION
# =====================================================

@st.cache_data
def load_evaluation():

    return {

        "Hybrid TabNet-XGBoost":
            pd.read_csv(DATA_DIR/"hasil_evaluasi.csv"),

        "Hybrid TabNet-XGBoost (Extreme)":
            pd.read_csv(DATA_DIR/"hasil_evaluasi_ektrem.csv"),

        "Hybrid TabNet-SVR":
            pd.read_csv(DATA_DIR/"hasil_evaluasi_svr.csv")

    }

evaluation_data = load_evaluation()

# =====================================================
# FEATURES
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

if menu=="🏠 Home":

    st.title("🌧 Daily Rainfall Prediction")

    st.markdown("""

## Hybrid TabNet–XGBoost Based on Radiosonde Data

Aplikasi ini digunakan untuk memprediksi intensitas curah hujan harian menggunakan model Hybrid TabNet–XGBoost berbasis data radiosonde dan meteorologi permukaan BMKG Stasiun Meteorologi Kelas I Juanda.

### Model

- Hybrid TabNet-XGBoost
- Hybrid TabNet-XGBoost (Extreme)
- Hybrid TabNet-SVR


""")

    c1,c2,c3=st.columns(3)

    c1.info("📂 Upload Dataset")

    c2.info("🤖 Prediction")

    c3.info("📈 Evaluation")

    st.success("Silakan pilih menu Prediction.")

# =====================================================
# PREDICTION
# =====================================================

elif menu == "🌧 Prediction":

    st.title("🌧 Daily Rainfall Prediction")

    model_name = st.selectbox(

        "Prediction Model",

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

    if uploaded is not None:

        df = pd.read_excel(uploaded)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.write("Jumlah Data :", len(df))

        if st.button("🚀 Predict"):

            df = df.copy()

            # ==========================================
            # Pastikan kolom numerik
            # ==========================================

            numeric_columns = [

                "CH",

                "Humi0","WS","KI","Press0","LFC",

                "SI","CCL","500","TT","850",

                "LCL","Height","TPW","700","CAPE"

            ]

            for col in numeric_columns:

                df[col] = pd.to_numeric(

                    df[col],

                    errors="coerce"

                )

            # ==========================================
            # Membuat Lag
            # ==========================================

            lag_features = [

                "Humi0","WS","KI","Press0","LFC",

                "SI","CCL","500","TT","850",

                "LCL","Height","TPW","700","CAPE"

            ]

            for col in lag_features:

                df[col + "_lag1"] = df[col].shift(1)

            df["CH_lag1"] = df["CH"].shift(1)

            df = df.dropna().reset_index(drop=True)

            # ==========================================
            # Cek Feature
            # ==========================================

            missing = [

                x for x in FEATURES

                if x not in df.columns

            ]

            if len(missing) > 0:

                st.error("Feature berikut tidak ditemukan")

                st.write(missing)

                st.stop()

            X = df[FEATURES]

            # ==========================================
            # Scaling
            # ==========================================

            scaler = models[model_name]["scaler"]

            X_scaled = scaler.transform(X)

            # ==========================================
            # Predict
            # ==========================================

            model = models[model_name]["model"]

            prediction = model.predict(X_scaled)

            result = pd.DataFrame({

                "Tanggal":df["Tanggal"],

                "Actual":df["CH"],

                "Prediction":prediction

            })

            st.session_state.prediction_result = result

            # ==========================================
            # Tampilkan Hasil
            # ==========================================

            st.success("Prediction Completed")

            st.subheader("Prediction Result")

            st.dataframe(result)

            # ==========================================
            # Grafik
            # ==========================================

            fig = px.line(

                result,

                x="Tanggal",

                y=["Actual","Prediction"],

                markers=True

            )

            fig.update_layout(

                xaxis_title="Date",

                yaxis_title="Rainfall (mm)"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )
# =====================================================
# MODEL EVALUATION
# =====================================================

elif menu == "📊 Model Evaluation":

    st.title("📊 Model Evaluation")

    model_name = st.selectbox(

        "Select Evaluation Model",

        [

            "Hybrid TabNet-XGBoost",

            "Hybrid TabNet-XGBoost (Extreme)",

            "Hybrid TabNet-SVR"

        ]

    )

    result = evaluation_data[model_name]

    actual = result["Actual"]
    pred = result["Prediction"]

    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)

    mask = actual != 0

    mape = np.mean(

        np.abs(

            (actual[mask]-pred[mask])/

            actual[mask]

        )

    )*100

    st.subheader("Performance Metrics")

    c1,c2,c3 = st.columns(3)

    c1.metric("RMSE",round(rmse,3))
    c2.metric("MAE",round(mae,3))
    c3.metric("MAPE (%)",round(mape,3))

    st.divider()

    st.subheader("Prediction Result (Testing Data 2025)")

    st.dataframe(result,use_container_width=True)

    st.divider()

    st.subheader("Actual vs Prediction")

    fig = px.line(

        result,

        x="Tanggal",

        y=["Actual","Prediction"],

        markers=False,

        template="plotly_white"

    )

    fig.update_layout(

        height=550,

        xaxis_title="Date",

        yaxis_title="Rainfall (mm)",

        legend_title="",

        hovermode="x unified"

    )

    fig.update_traces(

        line=dict(width=2)

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    st.subheader("Prediction Distribution")

    fig2 = px.scatter(

        result,

        x="Actual",

        y="Prediction",

        opacity=0.7,

        template="plotly_white"

    )

    fig2.update_layout(

        height=500,

        xaxis_title="Actual Rainfall (mm)",

        yaxis_title="Predicted Rainfall (mm)"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )
