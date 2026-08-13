import io
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Daily Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# MODEL CONFIG — SESUAI NOTEBOOK COLAB
# ============================================================

MODELS = {
    "Hybrid TabNet-XGBoost": {
        "file": "hybrid_xgb.pkl",
        "kind": "xgb",
        "features": [
            "Humi0_lag1",
            "TPW_lag1",
            "700_lag1",
            "LCL_lag1",
            "500_lag1",
            "KI_lag1",
            "850_lag4",
            "LI_lag1",
            "SI_lag1",
            "TT_lag3",
            "CAPE_lag3",
            "Height_lag1",
            "LFC_lag1",
            "Press0_lag7",
            "Temp0_lag7",
            "CH_lag1",
            "month",
            "month_sin",
            "month_cos",
        ],
    },
    "Hybrid TabNet-XGBoost (Extreme)": {
        "file": "hybrid_extreme.pkl",
        "kind": "xgb",
        "features": [
            "Humi0_lag1",
            "TPW_lag1",
            "700_lag1",
            "LCL_lag1",
            "500_lag1",
            "KI_lag1",
            "850_lag1",
            "LI_lag1",
            "CCL_lag1",
            "SI_lag1",
            "CIN_lag1",
            "CAPE_lag1",
            "Press0_lag7",
            "BOYDEN_lag1",
            "Temp0_lag7",
            "CH_lag1",
            "month",
            "month_sin",
            "month_cos",
        ],
    },
    "Hybrid TabNet-SVR": {
        "file": "hybrid_svr.pkl",
        "kind": "svr",
        "scaler_file": "scaler_svr.pkl",
        "features": [
            "Humi0_lag1",
            "TPW_lag1",
            "700_lag1",
            "LCL_lag1",
            "500_lag1",
            "KI_lag1",
            "850_lag4",
            "LI_lag1",
            "SI_lag1",
            "TT_lag3",
            "CIN_lag1",
            "Height_lag1",
            "Press0_lag7",
            "KO_lag1",
            "Temp0_lag7",
            "CH_lag1",
            "month",
            "month_sin",
            "month_cos",
        ],
    },
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f9fd 0%, #ffffff 48%, #f7fafc 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2942 0%, #125174 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 2.5rem 2.6rem;
        border-radius: 26px;
        background:
            radial-gradient(circle at 88% 18%, rgba(255,255,255,.20), transparent 22%),
            linear-gradient(135deg, #0b2942 0%, #146a98 58%, #2aa8c7 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 38px rgba(11,41,66,.18);
    }

    .hero h1 {
        font-size: 2.65rem;
        margin: 0 0 .45rem 0;
    }

    .hero p {
        font-size: 1.03rem;
        line-height: 1.75;
        max-width: 950px;
        margin: 0;
    }

    .card {
        padding: 1.35rem;
        border-radius: 19px;
        background: white;
        border: 1px solid #e2edf4;
        box-shadow: 0 8px 25px rgba(22,58,85,.07);
        min-height: 145px;
    }

    .card h3 {
        color: #0b2942;
        margin-top: 0;
    }

    .result-card {
        padding: 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #edf9ff, #ffffff);
        border: 1px solid #cbe8f4;
        text-align: center;
        box-shadow: 0 8px 24px rgba(24,117,151,.08);
    }

    .result-value {
        font-size: 3.1rem;
        font-weight: 750;
        color: #087ea4;
        line-height: 1.2;
        margin: .4rem 0;
    }

    .result-label {
        color: #607587;
        font-size: .92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    loaded = {}

    for name, cfg in MODELS.items():
        model_path = MODEL_DIR / cfg["file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}"
            )

        model = joblib.load(model_path)

        item = {
            "model": model,
            "kind": cfg["kind"],
            "features": cfg["features"],
        }

        if cfg["kind"] == "svr":
            scaler_path = MODEL_DIR / cfg["scaler_file"]

            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler SVR tidak ditemukan: {scaler_path}"
                )

            item["scaler"] = joblib.load(scaler_path)

        loaded[name] = item

    return loaded


@st.cache_data
def load_evaluation():
    files = {
        "Hybrid TabNet-XGBoost": "hasil_evaluasi.csv",
        "Hybrid TabNet-XGBoost (Extreme)": "hasil_evaluasi_ektrem.csv",
        "Hybrid TabNet-SVR": "hasil_evaluasi_svr.csv",
    }

    result = {}

    for name, filename in files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"File evaluasi tidak ditemukan: {path}"
            )

        result[name] = pd.read_csv(path)

    return result


try:
    MODEL_OBJECTS = load_models()
    EVALUATION_DATA = load_evaluation()
except Exception as exc:
    st.error("Model atau data evaluasi belum siap.")
    st.exception(exc)
    st.stop()


# ============================================================
# DATA / FEATURE ENGINEERING
# ============================================================

def parse_lag(feature):
    match = re.fullmatch(
        r"(.+)_lag([1-9][0-9]*)",
        feature
    )

    if not match:
        return None, None

    return match.group(1), int(match.group(2))


def required_base_columns(features):
    bases = []

    for feature in features:
        base, lag = parse_lag(feature)

        if base is not None:
            bases.append(base)
        elif feature not in {
            "month",
            "month_sin",
            "month_cos",
        }:
            bases.append(feature)

    return list(dict.fromkeys(bases))


def prepare_data(df, features):
    data = df.copy()

    if "Tanggal" not in data.columns:
        raise ValueError(
            "Kolom 'Tanggal' tidak ditemukan."
        )

    data["Tanggal"] = pd.to_datetime(
        data["Tanggal"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Tanggal"]
    ).copy()

    for col in data.columns:
        if col != "Tanggal":
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    if "CH" not in data.columns:
        data["CH"] = np.nan

    data = data.sort_values(
        "Tanggal"
    ).reset_index(drop=True)

    # Buat hanya lag yang diminta model.
    for feature in features:
        base, lag = parse_lag(feature)

        if base is not None:
            if base not in data.columns:
                data[feature] = np.nan
            else:
                data[feature] = data[base].shift(lag)

    # Fitur musiman persis seperti Colab.
    if any(x in features for x in ["month", "month_sin", "month_cos"]):
        data["month"] = data["Tanggal"].dt.month

        data["month_sin"] = np.sin(
            2 * np.pi * data["month"] / 12
        )

        data["month_cos"] = np.cos(
            2 * np.pi * data["month"] / 12
        )

    return data


def validate_model_object(model_name):
    cfg = MODELS[model_name]
    obj = MODEL_OBJECTS[model_name]

    model = obj["model"]
    features = cfg["features"]

    # XGBoost pada notebook dilatih TANPA StandardScaler.
    # SVR pada notebook menggunakan StandardScaler.
    if hasattr(model, "feature_names_in_"):
        names = [str(x) for x in model.feature_names_in_]

        if names != features:
            raise ValueError(
                "Model .pkl tidak cocok dengan notebook Colab.\n\n"
                f"Model meminta:\n{names}\n\n"
                f"Colab menggunakan:\n{features}\n\n"
                "Silakan ekspor ulang model dari Colab menggunakan "
                "kode penyimpanan yang saya berikan."
            )

    if hasattr(model, "n_features_in_"):
        if int(model.n_features_in_) != len(features):
            raise ValueError(
                f"Jumlah fitur model = {model.n_features_in_}, "
                f"sedangkan pipeline Colab = {len(features)}."
            )

    if cfg["kind"] == "svr":
        scaler = obj["scaler"]

        if hasattr(scaler, "n_features_in_"):
            if int(scaler.n_features_in_) != len(features):
                raise ValueError(
                    f"Jumlah fitur scaler SVR = {scaler.n_features_in_}, "
                    f"sedangkan pipeline Colab = {len(features)}."
                )


def predict_date(data, model_name, selected_date):
    cfg = MODELS[model_name]
    obj = MODEL_OBJECTS[model_name]

    features = cfg["features"]

    row_index = data.index[
        data["Tanggal"] == pd.Timestamp(selected_date)
    ].tolist()

    if not row_index:
        raise ValueError(
            "Tanggal yang dipilih tidak ditemukan."
        )

    idx = row_index[0]
    X = data.loc[[idx], features].copy()

    missing = X.columns[
        X.isna().any()
    ].tolist()

    if missing:
        raise ValueError(
            "Data untuk tanggal tersebut belum lengkap.\n\n"
            "Fitur yang kosong:\n"
            + ", ".join(missing)
            + "\n\n"
            "Pastikan data beberapa hari sebelumnya tersedia "
            "karena model menggunakan lag."
        )

    model = obj["model"]

    # XGBoost: langsung ke model, sesuai Colab.
    if cfg["kind"] == "xgb":
        prediction = model.predict(X)

    # SVR: StandardScaler dahulu, sesuai Colab.
    else:
        scaler = obj["scaler"]
        X_scaled = scaler.transform(X.values)
        prediction = model.predict(X_scaled)

    value = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return value, data.loc[idx]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(result):
    actual = pd.to_numeric(
        result["Actual"],
        errors="coerce"
    )

    prediction = pd.to_numeric(
        result["Prediction"],
        errors="coerce"
    )

    valid = actual.notna() & prediction.notna()

    actual = actual[valid]
    prediction = prediction[valid]

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    mae = mean_absolute_error(
        actual,
        prediction
    )

    nonzero = actual != 0

    if nonzero.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        actual[nonzero]
                        - prediction[nonzero]
                    )
                    / actual[nonzero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return rmse, mae, mape


def to_excel(df):
    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Evaluation"
        )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌧️ Rainfall AI")
    st.caption("Daily Rainfall Prediction")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🌧️ Prediction",
            "📊 Model Evaluation",
        ]
    )


# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Daily Rainfall Prediction</h1>
            <p>
                Sistem prediksi intensitas curah hujan harian.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>📁 Data Meteorologi</h3>
                <p>
                    Dataset Excel digunakan sebagai sumber data
                    meteorologi dan curah hujan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>🤖 Hybrid Model</h3>
                <p>
                    Tersedia Hybrid TabNet-XGBoost,
                    Extreme, dan Hybrid TabNet-SVR.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>📊 Evaluasi</h3>
                <p>
                    Evaluasi dilengkapi RMSE, MAE, MAPE,
                    residual dan visualisasi aktual-prediksi.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Cara Menggunakan")

    a, b, c = st.columns(3)

    with a:
        st.info(
            "**01 — Upload Dataset**\n\n"
            "Masukkan dataset bersih Excel."
        )

    with b:
        st.info(
            "**02 — Pilih Tanggal**\n\n"
            "Tentukan tanggal yang ingin diprediksi."
        )

    with c:
        st.info(
            "**03 — Lihat Hasil**\n\n"
            "Sistem menampilkan intensitas hujan dalam mm."
        )


# ============================================================
# PREDICTION
# ============================================================

elif menu == "🌧️ Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Rainfall Prediction</h1>
            <p>
                Pilih model dan tanggal.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_name = st.selectbox(
        "Prediction Model",
        list(MODELS.keys())
    )

    # Cek apakah .pkl sesuai pipeline Colab.
    try:
        validate_model_object(model_name)
    except Exception as exc:
        st.error("Model di folder models belum konsisten dengan Colab.")
        st.exception(exc)

        st.info(
            "Jangan mengganti daftar fitur di app.py. "
            "Ekspor ulang model dari notebook Colab menggunakan "
            "kode penyimpanan yang disediakan."
        )

        st.stop()

    uploaded = st.file_uploader(
        "Upload Dataset Bersih (.xlsx)",
        type=["xlsx"],
    )

    if uploaded is not None:

        features = MODELS[model_name]["features"]

        try:
            raw = pd.read_excel(uploaded)

            required = required_base_columns(
                features
            )

            missing = [
                col for col in required
                if col not in raw.columns
            ]

            if missing:
                st.error(
                    "Variabel dasar berikut tidak ditemukan:"
                )
                st.write(missing)
                st.stop()

            data = prepare_data(
                raw,
                features
            )

        except Exception as exc:
            st.error("Dataset tidak dapat diproses.")
            st.exception(exc)
            st.stop()

        st.success(
            f"Dataset berhasil dimuat — {len(data):,} baris."
        )

        with st.expander("Preview Dataset"):
            st.dataframe(
                raw.head(10),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Fitur Model"):
            st.write(
                f"**{len(features)} fitur**"
            )
            st.write(features)

        valid_mask = data[features].notna().all(
            axis=1
        )

        valid_dates = (
            data.loc[
                valid_mask,
                "Tanggal"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        if not valid_dates:
            st.error(
                "Tidak ada tanggal dengan seluruh fitur "
                "yang diperlukan."
            )
            st.stop()

        selected_date = st.selectbox(
            "Tanggal Prediksi",
            valid_dates,
            format_func=lambda x:
                pd.Timestamp(x).strftime(
                    "%d-%m-%Y"
                )
        )

        if st.button(
            "🔮 Prediksi Intensitas Hujan",
            type="primary",
            use_container_width=True
        ):

            try:
                prediction, selected_row = predict_date(
                    data,
                    model_name,
                    selected_date
                )

                st.session_state.prediction_result = {
                    "date": pd.Timestamp(
                        selected_date
                    ),
                    "model": model_name,
                    "prediction": prediction,
                    "actual": selected_row["CH"],
                }

            except Exception as exc:
                st.session_state.prediction_result = None
                st.error(
                    "Prediksi tidak berhasil."
                )
                st.exception(exc)

        result = st.session_state.get(
            "prediction_result"
        )

        if result is not None:
            if (
                result["model"] == model_name
                and result["date"]
                == pd.Timestamp(selected_date)
            ):

                st.markdown(
                    "### Hasil Prediksi"
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Intensitas Curah Hujan
                        </div>
                        <div class="result-value">
                            {result["prediction"]:.2f} mm
                        </div>
                        <div class="result-label">
                            {result["date"].strftime("%d-%m-%Y")}
                            &nbsp; • &nbsp;
                            {result["model"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Prediksi Curah Hujan",
                        f'{result["prediction"]:.2f} mm'
                    )

                with c2:
                    if pd.notna(result["actual"]):
                        st.metric(
                            "Curah Hujan Aktual",
                            f'{float(result["actual"]):.2f} mm'
                        )
                    else:
                        st.metric(
                            "Curah Hujan Aktual",
                            "Tidak tersedia"
                        )

                # Grafik sekitar tanggal yang dipilih.
                idx = data.index[
                    data["Tanggal"]
                    == pd.Timestamp(selected_date)
                ][0]

                context = data.iloc[
                    max(0, idx - 7):
                    min(len(data), idx + 8)
                ].copy()

                context["Prediction"] = np.nan

                context.loc[
                    context["Tanggal"]
                    == pd.Timestamp(selected_date),
                    "Prediction"
                ] = result["prediction"]

                fig = px.line(
                    context,
                    x="Tanggal",
                    y=["CH", "Prediction"],
                    markers=True,
                    template="plotly_white",
                    labels={
                        "value": "Rainfall (mm)",
                        "Tanggal": "Tanggal",
                        "variable": ""
                    }
                )

                fig.update_layout(
                    height=430,
                    hovermode="x unified",
                    legend_title=""
                )

                st.markdown(
                    "### Visualisasi Prediksi"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif menu == "📊 Model Evaluation":

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Evaluation</h1>
            <p>
                Evaluasi hasil pengujian model dengan metrik,
                tabel hasil dan visualisasi kesalahan prediksi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_name = st.selectbox(
        "Evaluation Model",
        list(MODELS.keys())
    )

    result = EVALUATION_DATA[
        model_name
    ].copy()

    if "Tanggal" in result.columns:
        result["Tanggal"] = pd.to_datetime(
            result["Tanggal"],
            errors="coerce"
        )

    rmse, mae, mape = calculate_metrics(
        result
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

    c2.metric(
        "MAE",
        f"{mae:.3f}"
    )

    c3.metric(
        "MAPE",
        "—"
        if np.isnan(mape)
        else f"{mape:.2f}%"
    )

    st.divider()

    # Download digabung di Model Evaluation.
    st.markdown(
        "### Download Hasil Evaluasi"
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=result.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="hasil_evaluasi.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=to_excel(result),
            file_name="hasil_evaluasi.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    st.divider()

    st.markdown(
        "### Prediction Result"
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Actual vs Prediction
    st.markdown(
        "### 1. Actual vs Prediction"
    )

    fig1 = px.line(
        result,
        x="Tanggal",
        y=["Actual", "Prediction"],
        template="plotly_white",
        labels={
            "value": "Rainfall (mm)",
            "Tanggal": "Date",
            "variable": ""
        }
    )

    fig1.update_layout(
        height=520,
        hovermode="x unified",
        legend_title=""
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # Scatter
    st.markdown(
        "### 2. Actual vs Predicted"
    )

    scatter = result.dropna(
        subset=[
            "Actual",
            "Prediction"
        ]
    ).copy()

    fig2 = px.scatter(
        scatter,
        x="Actual",
        y="Prediction",
        opacity=.7,
        template="plotly_white",
        labels={
            "Actual": "Actual Rainfall (mm)",
            "Prediction": "Predicted Rainfall (mm)"
        }
    )

    if not scatter.empty:
        low = min(
            scatter["Actual"].min(),
            scatter["Prediction"].min()
        )

        high = max(
            scatter["Actual"].max(),
            scatter["Prediction"].max()
        )

        fig2.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line=dict(dash="dash")
        )

    fig2.update_layout(
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Residual
    residual = result.copy()

    residual["Residual"] = (
        residual["Actual"]
        - residual["Prediction"]
    )

    st.markdown(
        "### 3. Residual Error"
    )

    fig3 = px.line(
        residual,
        x="Tanggal",
        y="Residual",
        template="plotly_white",
        labels={
            "Residual": "Residual (mm)",
            "Tanggal": "Date"
        }
    )

    fig3.add_hline(
        y=0,
        line_dash="dash"
    )

    fig3.update_layout(
        height=420
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # Error distribution
    st.markdown(
        "### 4. Distribution of Prediction Error"
    )

    fig4 = px.histogram(
        residual,
        x="Residual",
        nbins=30,
        template="plotly_white",
        labels={
            "Residual": "Prediction Error (mm)"
        }
    )

    fig4.update_layout(
        height=400,
        xaxis_title="Prediction Error (mm)",
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )
