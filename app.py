import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
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

# Fitur asli yang memang ada di dataset_bersih.xlsx.
BASE_FEATURES = [
    "Humi0", "WS", "KI", "Press0", "LFC",
    "SI", "CCL", "500", "TT", "850",
    "LCL", "Height", "TPW", "700", "CAPE",
]

LAG_FEATURES = [f"{c}_lag1" for c in BASE_FEATURES]
FULL_FEATURES = BASE_FEATURES + LAG_FEATURES + ["CH_lag1"]


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
        background: linear-gradient(180deg, #0b2942 0%, #125174 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 2.35rem 2.5rem;
        border-radius: 26px;
        background:
            radial-gradient(circle at 88% 18%, rgba(255,255,255,.20), transparent 22%),
            linear-gradient(135deg, #0b2942 0%, #146a98 58%, #2aa8c7 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 38px rgba(11,41,66,.18);
    }

    .hero h1 {
        font-size: 2.6rem;
        margin: 0 0 .45rem 0;
    }

    .hero p {
        font-size: 1.03rem;
        line-height: 1.7;
        max-width: 950px;
        opacity: .96;
        margin: 0;
    }

    .card {
        padding: 1.3rem;
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
        padding: 1.55rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
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
    files = {
        "Hybrid TabNet-XGBoost": "hasil_evaluasi.csv",
        "Hybrid TabNet-XGBoost (Extreme)": "hasil_evaluasi_ektrem.csv",
        "Hybrid TabNet-SVR": "hasil_evaluasi_svr.csv",
    }

    result = {}

    for name, filename in files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"File evaluasi tidak ditemukan: {path}")

        result[name] = pd.read_csv(path)

    return result


try:
    MODELS = load_models()
    EVALUATION_DATA = load_evaluation()
except Exception as exc:
    st.error("Model atau file evaluasi tidak dapat dimuat.")
    st.exception(exc)
    st.stop()


# ============================================================
# MODEL INSPECTION
# ============================================================

def get_attr(obj, name):
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def unwrap_predictor(model):
    """
    Mendukung:
    - model sklearn/xgboost/svr biasa
    - Pipeline
    - dictionary yang menyimpan predictor
    - objek hybrid yang memiliki predictor/model/xgb_model.
    """
    if hasattr(model, "predict"):
        return model

    if isinstance(model, dict):
        preferred = [
            "model",
            "predictor",
            "xgb_model",
            "xgboost_model",
            "svr_model",
            "regressor",
            "estimator",
        ]

        for key in preferred:
            value = model.get(key)
            if value is not None and hasattr(value, "predict"):
                return value

        for value in model.values():
            if hasattr(value, "predict"):
                return value

    for attr in [
        "model",
        "predictor",
        "xgb_model",
        "xgboost_model",
        "svr_model",
        "regressor",
        "estimator",
    ]:
        value = get_attr(model, attr)
        if value is not None and hasattr(value, "predict"):
            return value

    return model


def model_expected_count(model, scaler):
    for obj in [scaler, unwrap_predictor(model), model]:
        value = get_attr(obj, "n_features_in_")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def model_feature_names(model, scaler):
    for obj in [scaler, unwrap_predictor(model), model]:
        names = get_attr(obj, "feature_names_in_")
        if names is not None:
            try:
                return [str(x) for x in names]
            except Exception:
                pass
    return None


# ============================================================
# DATA PREPARATION
# ============================================================

def validate_columns(df):
    required = ["Tanggal"] + BASE_FEATURES
    return [c for c in required if c not in df.columns]


def prepare_dataset(df):
    data = df.copy()

    data["Tanggal"] = pd.to_datetime(
        data["Tanggal"],
        errors="coerce",
    )

    for col in BASE_FEATURES:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    if "CH" in data.columns:
        data["CH"] = pd.to_numeric(
            data["CH"],
            errors="coerce",
        )
    else:
        data["CH"] = np.nan

    data = data.dropna(subset=["Tanggal"]).copy()

    # Penting: lag dibuat setelah tanggal diurutkan.
    data = data.sort_values("Tanggal").reset_index(drop=True)

    duplicate_dates = int(
        data["Tanggal"].duplicated(keep=False).sum()
    )

    # Lag-1.
    for col in BASE_FEATURES:
        data[f"{col}_lag1"] = data[col].shift(1)

    data["CH_lag1"] = data["CH"].shift(1)

    return data, duplicate_dates


# ============================================================
# FEATURE SELECTION
# ============================================================

def candidate_feature_sets(data):
    """
    Beberapa struktur yang mungkin digunakan saat training.
    FULL_FEATURES = 31 fitur:
    15 fitur asli + 15 lag-1 + CH_lag1.
    """
    candidates = [
        ("full_31", FULL_FEATURES),
        ("base_plus_lag_30", BASE_FEATURES + LAG_FEATURES),
        ("base_15", BASE_FEATURES),
        ("base_plus_chlag_16", BASE_FEATURES + ["CH_lag1"]),
    ]

    return [
        (name, features)
        for name, features in candidates
        if all(col in data.columns for col in features)
    ]


def choose_features(data, model_name):
    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]

    expected_names = model_feature_names(model, scaler)
    expected_count = model_expected_count(model, scaler)

    candidates = candidate_feature_sets(data)

    # Prioritas tertinggi: nama fitur yang tersimpan pada scaler/model.
    if expected_names:
        if all(col in data.columns for col in expected_names):
            return expected_names, "feature_names_in_"

        raise ValueError(
            "Model/scaler menyimpan nama fitur yang tidak tersedia pada dataset. "
            f"Fitur yang diminta: {expected_names}"
        )

    # Jika hanya jumlah fitur yang diketahui, cocokkan otomatis.
    if expected_count is not None:
        for label, features in candidates:
            if len(features) == expected_count:
                return features, f"n_features_in_={expected_count}"

        available = [f"{label}: {len(features)} fitur" for label, features in candidates]

        raise ValueError(
            f"Model/scaler membutuhkan {expected_count} fitur, "
            "tetapi struktur fitur yang tersedia tidak cocok. "
            f"Kandidat: {', '.join(available)}"
        )

    # Fallback paling sesuai dengan kode training lama:
    # 15 fitur + lag-1 + CH_lag1.
    return FULL_FEATURES, "fallback_full_31"


# ============================================================
# PREDICTION
# ============================================================

def predict_date(data, model_name, selected_date):
    selected_date = pd.Timestamp(selected_date)

    matches = data.index[
        data["Tanggal"] == selected_date
    ].tolist()

    if not matches:
        raise ValueError(
            f"Tanggal {selected_date.strftime('%d-%m-%Y')} tidak ditemukan."
        )

    idx = matches[0]

    features, feature_source = choose_features(
        data,
        model_name,
    )

    row = data.loc[[idx], features].copy()

    missing = row.columns[row.isna().any()].tolist()

    if missing:
        raise ValueError(
            "Data fitur untuk tanggal tersebut belum lengkap: "
            + ", ".join(missing)
        )

    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]
    predictor = unwrap_predictor(model)

    # Gunakan DataFrame jika scaler menyimpan feature_names_in_.
    # Ini menghindari mismatch nama fitur pada sklearn.
    scaler_names = get_attr(scaler, "feature_names_in_")

    if scaler_names is not None:
        scaler_input = row[
            [str(x) for x in scaler_names]
        ]
    else:
        scaler_input = row

    try:
        X_scaled = scaler.transform(scaler_input)
    except Exception as exc:
        raise RuntimeError(
            "Scaling gagal.\n\n"
            f"Model: {model_name}\n"
            f"Fitur yang dipakai: {features}\n"
            f"Jumlah fitur: {len(features)}\n"
            f"Jumlah fitur scaler: {get_attr(scaler, 'n_features_in_')}\n"
            f"Error asli: {exc}"
        ) from exc

    try:
        prediction = predictor.predict(X_scaled)
    except Exception as exc:
        raise RuntimeError(
            "Model gagal melakukan prediksi setelah scaling.\n\n"
            f"Model: {model_name}\n"
            f"Predictor: {type(predictor).__name__}\n"
            f"Shape input setelah scaling: {getattr(X_scaled, 'shape', None)}\n"
            f"Jumlah fitur model: {get_attr(predictor, 'n_features_in_')}\n"
            f"Error asli: {exc}"
        ) from exc

    prediction = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return prediction, data.loc[idx], features, feature_source


# ============================================================
# EVALUATION HELPERS
# ============================================================

def calculate_metrics(result):
    actual = pd.to_numeric(
        result["Actual"],
        errors="coerce",
    )

    pred = pd.to_numeric(
        result["Prediction"],
        errors="coerce",
    )

    valid = actual.notna() & pred.notna()

    actual = actual[valid]
    pred = pred[valid]

    rmse = np.sqrt(
        mean_squared_error(actual, pred)
    )

    mae = mean_absolute_error(
        actual,
        pred,
    )

    nonzero = actual != 0

    if nonzero.any():
        mape = (
            np.mean(
                np.abs(
                    (actual[nonzero] - pred[nonzero])
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
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Evaluation",
        )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SESSION
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


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
        ],
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
                Sistem prediksi intensitas curah hujan harian menggunakan
                model Hybrid TabNet dengan XGBoost dan SVR. Pilih tanggal
                yang ingin diprediksi dan lihat hasil intensitas hujan
                dalam satuan milimeter (mm).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sistem Prediksi")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>📁 Dataset</h3>
                <p>
                    Upload dataset bersih Excel yang berisi tanggal dan
                    parameter meteorologi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>🤖 Model</h3>
                <p>
                    Tersedia Hybrid TabNet-XGBoost, versi Extreme,
                    dan Hybrid TabNet-SVR.
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
                    Evaluasi model dilengkapi metrik dan visualisasi
                    hasil prediksi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cara Menggunakan")

    a, b, c = st.columns(3)

    with a:
        st.info("**01 — Upload**\n\nMasukkan dataset Excel.")

    with b:
        st.info("**02 — Pilih Tanggal**\n\nTentukan tanggal yang ingin diprediksi.")

    with c:
        st.info("**03 — Prediksi**\n\nLihat intensitas hujan dalam mm.")

    st.success(
        "Silakan masuk ke menu **Prediction** untuk memulai."
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
                Upload dataset, pilih model, pilih tanggal, kemudian
                sistem akan menampilkan intensitas curah hujan.
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
    )

    if uploaded is not None:

        try:
            raw_df = pd.read_excel(uploaded)
        except Exception as exc:
            st.error("File Excel tidak dapat dibaca.")
            st.exception(exc)
            st.stop()

        missing_columns = validate_columns(raw_df)

        if missing_columns:
            st.error("Kolom wajib tidak ditemukan:")
            st.write(missing_columns)
            st.stop()

        data, duplicate_count = prepare_dataset(raw_df)

        if duplicate_count:
            st.warning(
                f"Ditemukan {duplicate_count} baris dengan tanggal duplikat. "
                "Untuk tanggal yang sama, sistem menggunakan baris pertama."
            )

        st.success(
            f"Dataset berhasil dimuat — {len(data):,} baris."
        )

        with st.expander("Preview Dataset"):
            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        # Tentukan tanggal yang benar-benar memiliki fitur lengkap
        try:
            features_for_date, _ = choose_features(
                data,
                model_name,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        valid_mask = data[features_for_date].notna().all(axis=1)

        valid_dates = (
            data.loc[valid_mask, "Tanggal"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        if not valid_dates:
            st.error(
                "Tidak ada tanggal yang memiliki data lengkap "
                "untuk fitur model."
            )
            st.stop()

        st.markdown("### Pilih Tanggal Prediksi")

        selected_date = st.selectbox(
            "Tanggal",
            valid_dates,
            format_func=lambda x:
                pd.Timestamp(x).strftime("%d-%m-%Y"),
        )

        if st.button(
            "🔮 Prediksi Intensitas Hujan",
            type="primary",
            use_container_width=True,
        ):

            try:
                prediction, selected_row, used_features, feature_source = (
                    predict_date(
                        data,
                        model_name,
                        selected_date,
                    )
                )

                st.session_state.prediction_result = {
                    "date": pd.Timestamp(selected_date),
                    "model": model_name,
                    "prediction": prediction,
                    "actual": selected_row["CH"],
                    "features": used_features,
                    "feature_source": feature_source,
                }

            except Exception as exc:
                st.session_state.prediction_result = None
                st.error("Prediksi tidak berhasil.")
                st.exception(exc)

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
                            {result["date"].strftime("%d-%m-%Y")}
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

                with st.expander("Informasi Pemrosesan"):
                    st.write(
                        f"Struktur fitur: **{result['feature_source']}**"
                    )
                    st.write(
                        f"Jumlah fitur: **{len(result['features'])}**"
                    )
                    st.write(result["features"])

                # Grafik konteks ±7 hari
                selected_idx = data.index[
                    data["Tanggal"] == pd.Timestamp(selected_date)
                ][0]

                start = max(0, selected_idx - 7)
                end = min(len(data), selected_idx + 8)

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
                Evaluasi performa model menggunakan data pengujian,
                dilengkapi metrik dan visualisasi kesalahan prediksi.
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

    st.markdown("### Download Hasil Evaluasi")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="hasil_evaluasi.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=to_excel(result),
            file_name="hasil_evaluasi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()

    st.markdown("### Prediction Result")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # 1. Actual vs Prediction
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
        hovermode="x unified",
        legend_title="",
    )

    fig1.update_traces(line=dict(width=2))

    st.plotly_chart(
        fig1,
        use_container_width=True,
    )

    # 2. Scatter
    st.markdown("### 2. Actual vs Predicted")

    scatter = result.dropna(
        subset=["Actual", "Prediction"]
    ).copy()

    fig2 = px.scatter(
        scatter,
        x="Actual",
        y="Prediction",
        opacity=0.7,
        template="plotly_white",
        labels={
            "Actual": "Actual Rainfall (mm)",
            "Prediction": "Predicted Rainfall (mm)",
        },
    )

    if not scatter.empty:
        low = min(
            scatter["Actual"].min(),
            scatter["Prediction"].min(),
        )

        high = max(
            scatter["Actual"].max(),
            scatter["Prediction"].max(),
        )

        fig2.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line=dict(dash="dash"),
        )

    fig2.update_layout(height=500)

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    # 3. Residual
    st.markdown("### 3. Residual Error")

    residual = result.copy()

    residual["Residual"] = (
        residual["Actual"]
        - residual["Prediction"]
    )

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

    # 4. Histogram
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
