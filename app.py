import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rainfall Prediction | Hybrid TabNet",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5faff 0%, #ffffff 48%, #f6f9fc 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2942 0%, #124e73 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 2.4rem 2.5rem;
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
        opacity: .96;
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

    .muted {
        color: #6d7d8d;
        font-size: .92rem;
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
        font-size: 3rem;
        font-weight: 750;
        color: #087ea4;
        line-height: 1.2;
        margin: .4rem 0;
    }

    .result-label {
        color: #607587;
        font-size: .92rem;
    }

    .section-title {
        color: #0b2942;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL / FEATURE CONFIGURATION
# ============================================================

MODEL_OPTIONS = [
    "Hybrid TabNet-XGBoost",
    "Hybrid TabNet-XGBoost (Extreme)",
    "Hybrid TabNet-SVR",
]

MODEL_FILES = {
    "Hybrid TabNet-XGBoost": {
        "model": "hybrid_xgb.pkl",
        "scaler": "scaler_xgb.pkl",
    },
    "Hybrid TabNet-XGBoost (Extreme)": {
        "model": "hybrid_extreme.pkl",
        "scaler": "scaler_ekstrem.pkl",
    },
    "Hybrid TabNet-SVR": {
        "model": "hybrid_svr.pkl",
        "scaler": "scaler_svr.pkl",
    },
}

BASE_FEATURES = [
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
]

FEATURES = BASE_FEATURES + [f"{col}_lag1" for col in BASE_FEATURES] + ["CH_lag1"]


# ============================================================
# RESOURCE LOADING
# ============================================================

@st.cache_resource
def load_models():
    loaded = {}

    for name, files in MODEL_FILES.items():
        model_path = MODEL_DIR / files["model"]
        scaler_path = MODEL_DIR / files["scaler"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_path}")

        loaded[name] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
        }

    return loaded


@st.cache_data
def load_evaluation():
    evaluation_files = {
        "Hybrid TabNet-XGBoost": "hasil_evaluasi.csv",
        "Hybrid TabNet-XGBoost (Extreme)": "hasil_evaluasi_ektrem.csv",
        "Hybrid TabNet-SVR": "hasil_evaluasi_svr.csv",
    }

    loaded = {}

    for name, filename in evaluation_files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"File evaluasi tidak ditemukan: {path}")

        loaded[name] = pd.read_csv(path)

    return loaded


# ============================================================
# DATA PREPARATION
# ============================================================

def validate_input_columns(df):
    required = ["Tanggal"] + BASE_FEATURES

    # CH tidak diwajibkan untuk tanggal yang ingin diprediksi.
    # CH tetap dipakai bila tersedia, karena diperlukan untuk CH_lag1
    # dan untuk membandingkan dengan aktual pada data historis.
    missing = [col for col in required if col not in df.columns]
    return missing


def prepare_prediction_data(df):
    data = df.copy()

    data["Tanggal"] = pd.to_datetime(data["Tanggal"], errors="coerce")

    for col in BASE_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "CH" in data.columns:
        data["CH"] = pd.to_numeric(data["CH"], errors="coerce")
    else:
        data["CH"] = np.nan

    data = data.dropna(subset=["Tanggal"]).copy()

    # Urutkan tanggal sebelum membuat lag.
    data = data.sort_values("Tanggal").reset_index(drop=True)

    # Jika ada tanggal ganda, gunakan baris pertama secara konsisten.
    # Aplikasi memberi peringatan agar pengguna mengetahui kondisi ini.
    duplicate_count = int(data["Tanggal"].duplicated(keep=False).sum())

    for col in BASE_FEATURES:
        data[f"{col}_lag1"] = data[col].shift(1)

    data["CH_lag1"] = data["CH"].shift(1)

    return data, duplicate_count


def get_prediction_dates(data):
    """
    Tanggal yang dapat diprediksi adalah tanggal dengan semua feature
    utama dan seluruh lag-1 tersedia.
    """
    valid = data[FEATURES].notna().all(axis=1)
    return data.loc[valid, "Tanggal"].drop_duplicates().sort_values()


def predict_for_date(data, model_name, selected_date):
    selected_date = pd.Timestamp(selected_date)

    matches = data.index[data["Tanggal"] == selected_date].tolist()

    if not matches:
        raise ValueError("Tanggal yang dipilih tidak ditemukan.")

    # Jika dataset memiliki tanggal duplikat, gunakan baris pertama.
    idx = matches[0]

    row = data.loc[[idx], FEATURES].copy()

    missing = row.columns[row.isna().any()].tolist()
    if missing:
        raise ValueError(
            "Fitur untuk tanggal tersebut belum lengkap: "
            + ", ".join(missing)
        )

    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]

    try:
        X_scaled = scaler.transform(row.values)
        prediction = np.asarray(model.predict(X_scaled)).reshape(-1)[0]
    except Exception as exc:
        raise ValueError(
            "Prediksi gagal. Pastikan dataset menggunakan struktur fitur "
            "yang sama dengan data training model."
        ) from exc

    return float(prediction), data.loc[idx]


# ============================================================
# METRICS / DOWNLOAD
# ============================================================

def calculate_metrics(result):
    actual = pd.to_numeric(result["Actual"], errors="coerce")
    prediction = pd.to_numeric(result["Prediction"], errors="coerce")

    valid = actual.notna() & prediction.notna()
    actual = actual[valid]
    prediction = prediction[valid]

    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)

    nonzero = actual != 0

    if nonzero.any():
        mape = (
            np.mean(
                np.abs(
                    (actual[nonzero] - prediction[nonzero])
                    / actual[nonzero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return rmse, mae, mape


def dataframe_to_excel(df):
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Evaluation")

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# INITIALIZE
# ============================================================

try:
    MODELS = load_models()
    EVALUATION_DATA = load_evaluation()
except Exception as exc:
    st.error("Aplikasi tidak dapat memuat model atau data evaluasi.")
    st.exception(exc)
    st.stop()


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌧️ Rainfall AI")
    st.caption("Hybrid TabNet Rainfall Prediction")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🌧️ Prediction",
            "📊 Model Evaluation",
        ],
    )

    st.divider()
    st.caption("Hybrid TabNet-XGBoost • Skripsi")


# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Daily Rainfall Prediction</h1>
            <p>
                Sistem prediksi intensitas curah hujan harian berbasis
                <b>Hybrid TabNet</b> dengan model XGBoost dan SVR.
                Sistem memanfaatkan variabel meteorologi permukaan dan
                parameter atmosfer untuk menghasilkan prediksi curah hujan
                dalam satuan milimeter (mm).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Tentang Sistem")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>📁 Data Meteorologi</h3>
                <p>
                    Dataset Excel berisi data tanggal, curah hujan,
                    dan parameter meteorologi yang digunakan sebagai
                    input sistem.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>🤖 Hybrid Model</h3>
                <p>
                    Sistem menyediakan Hybrid TabNet-XGBoost,
                    Hybrid TabNet-XGBoost Extreme, dan Hybrid TabNet-SVR.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>📈 Analisis</h3>
                <p>
                    Performa model dapat dilihat melalui RMSE, MAE,
                    MAPE, grafik aktual-prediksi, residual, dan distribusi error.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cara Menggunakan")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.info("**01 — Upload Dataset**\n\nMasukkan file Excel data bersih.")

    with s2:
        st.info("**02 — Pilih Tanggal**\n\nPilih tanggal yang ingin diprediksi.")

    with s3:
        st.info("**03 — Lihat Hasil**\n\nSistem menampilkan intensitas hujan dalam mm.")

    st.success(
        "💡 Silakan buka menu **Prediction** untuk mencoba prediksi."
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
                Pilih model dan tanggal yang ingin diprediksi.
                Sistem akan menghitung intensitas curah hujan berdasarkan
                fitur meteorologi dan data lag-1.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Prediction Model",
        MODEL_OPTIONS,
    )

    uploaded = st.file_uploader(
        "Upload Dataset Bersih (.xlsx)",
        type=["xlsx"],
        help="Gunakan dataset bersih dengan kolom Tanggal dan parameter meteorologi.",
    )

    if uploaded is not None:

        try:
            raw_df = pd.read_excel(uploaded)
        except Exception as exc:
            st.error("File Excel tidak dapat dibaca.")
            st.exception(exc)
            st.stop()

        missing = validate_input_columns(raw_df)

        if missing:
            st.error("Kolom berikut belum tersedia:")
            st.write(missing)
            st.stop()

        data, duplicate_count = prepare_prediction_data(raw_df)

        if duplicate_count:
            st.warning(
                f"Ditemukan {duplicate_count} baris yang memiliki tanggal duplikat. "
                "Sistem menggunakan baris pertama untuk tanggal yang sama."
            )

        if data.empty:
            st.error("Tidak ada data tanggal yang valid.")
            st.stop()

        st.success(
            f"Dataset berhasil dimuat: **{len(data):,} baris**."
        )

        with st.expander("Lihat Preview Dataset"):
            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        valid_dates = get_prediction_dates(data)

        if valid_dates.empty:
            st.error(
                "Tidak ada tanggal yang memenuhi seluruh kebutuhan fitur "
                "dan lag-1 untuk prediksi."
            )
            st.stop()

        st.markdown("### Pilih Tanggal Prediksi")

        selected_date = st.selectbox(
            "Tanggal",
            valid_dates.tolist(),
            format_func=lambda x: pd.Timestamp(x).strftime("%d-%m-%Y"),
        )

        if st.button(
            "🔮 Prediksi Intensitas Hujan",
            type="primary",
            use_container_width=True,
        ):
            try:
                prediction, selected_row = predict_for_date(
                    data,
                    model_name,
                    selected_date,
                )

                st.session_state.prediction_result = {
                    "date": pd.Timestamp(selected_date),
                    "model": model_name,
                    "prediction": prediction,
                    "actual": selected_row["CH"],
                }

            except Exception as exc:
                st.session_state.prediction_result = None
                st.error(str(exc))

        result = st.session_state.prediction_result

        if result is not None:

            if (
                result["model"] == model_name
                and result["date"] == pd.Timestamp(selected_date)
            ):

                st.markdown("### Hasil Prediksi")

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
                            {result["date"].strftime("%d %B %Y")}
                            &nbsp; • &nbsp;
                            {result["model"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Prediksi Curah Hujan",
                        f'{result["prediction"]:.2f} mm',
                    )

                with c2:
                    if pd.notna(result["actual"]):
                        st.metric(
                            "Curah Hujan Aktual",
                            f'{float(result["actual"]):.2f} mm',
                        )
                    else:
                        st.metric(
                            "Curah Hujan Aktual",
                            "Belum tersedia",
                        )

                # ------------------------------------------------
                # GRAFIK KONTEKS
                # ------------------------------------------------

                selected_index = data.index[
                    data["Tanggal"] == pd.Timestamp(selected_date)
                ][0]

                start = max(0, selected_index - 7)
                end = min(len(data), selected_index + 8)

                context = data.iloc[start:end].copy()

                context["Prediction"] = np.nan
                context.loc[
                    context["Tanggal"] == pd.Timestamp(selected_date),
                    "Prediction"
                ] = result["prediction"]

                chart = px.line(
                    context,
                    x="Tanggal",
                    y=["CH", "Prediction"],
                    markers=True,
                    template="plotly_white",
                    labels={
                        "value": "Rainfall (mm)",
                        "Tanggal": "Tanggal",
                        "variable": "",
                    },
                )

                chart.update_layout(
                    height=430,
                    hovermode="x unified",
                    legend_title="",
                )

                st.markdown("### Visualisasi Prediksi")

                st.plotly_chart(
                    chart,
                    use_container_width=True,
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
                Evaluasi model pada data pengujian, lengkap dengan metrik
                performa, hasil prediksi, visualisasi aktual versus prediksi,
                residual error, dan distribusi kesalahan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Evaluation Model",
        MODEL_OPTIONS,
    )

    result = EVALUATION_DATA[model_name].copy()

    if "Tanggal" in result.columns:
        result["Tanggal"] = pd.to_datetime(
            result["Tanggal"],
            errors="coerce",
        )

    rmse, mae, mape = calculate_metrics(result)

    st.markdown("### Performance Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", f"{rmse:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric(
        "MAPE",
        "—" if np.isnan(mape) else f"{mape:.2f}%",
    )

    st.divider()

    # --------------------------------------------------------
    # DOWNLOAD DIGABUNG DI MODEL EVALUATION
    # --------------------------------------------------------

    st.markdown("### Download Hasil Evaluasi")

    csv_data = result.to_csv(index=False).encode("utf-8")
    excel_data = dataframe_to_excel(result)

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="hasil_evaluasi.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=excel_data,
            file_name="hasil_evaluasi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()

    st.markdown("### Prediction Result — Testing Data")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # 1. ACTUAL VS PREDICTION
    # --------------------------------------------------------

    st.markdown("### 1. Actual vs Prediction")

    fig1 = px.line(
        result,
        x="Tanggal",
        y=["Actual", "Prediction"],
        template="plotly_white",
        labels={
            "value": "Rainfall (mm)",
            "Tanggal": "Date",
            "variable": "",
        },
    )

    fig1.update_layout(
        height=520,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        legend_title="",
        hovermode="x unified",
    )

    fig1.update_traces(line=dict(width=2))

    st.plotly_chart(
        fig1,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2. ACTUAL VS PREDICTED SCATTER
    # --------------------------------------------------------

    st.markdown("### 2. Actual vs Predicted")

    scatter = result.dropna(subset=["Actual", "Prediction"]).copy()

    fig2 = px.scatter(
        scatter,
        x="Actual",
        y="Prediction",
        opacity=0.70,
        template="plotly_white",
        labels={
            "Actual": "Actual Rainfall (mm)",
            "Prediction": "Predicted Rainfall (mm)",
        },
    )

    if not scatter.empty:
        min_value = min(
            scatter["Actual"].min(),
            scatter["Prediction"].min(),
        )
        max_value = max(
            scatter["Actual"].max(),
            scatter["Prediction"].max(),
        )

        fig2.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(dash="dash"),
        )

    fig2.update_layout(
        height=500,
        xaxis_title="Actual Rainfall (mm)",
        yaxis_title="Predicted Rainfall (mm)",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 3. RESIDUAL ERROR
    # --------------------------------------------------------

    st.markdown("### 3. Residual Error")

    residual = result.copy()
    residual["Residual"] = residual["Actual"] - residual["Prediction"]

    fig3 = px.line(
        residual,
        x="Tanggal",
        y="Residual",
        template="plotly_white",
        labels={
            "Residual": "Residual (mm)",
            "Tanggal": "Date",
        },
    )

    fig3.add_hline(
        y=0,
        line_dash="dash",
    )

    fig3.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Residual (mm)",
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 4. ERROR DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### 4. Distribution of Prediction Error")

    fig4 = px.histogram(
        residual,
        x="Residual",
        nbins=30,
        template="plotly_white",
        labels={
            "Residual": "Prediction Error (mm)",
        },
    )

    fig4.update_layout(
        height=400,
        xaxis_title="Prediction Error (mm)",
        yaxis_title="Frequency",
    )

    st.plotly_chart(
        fig4,
        use_container_width=True,
    )
