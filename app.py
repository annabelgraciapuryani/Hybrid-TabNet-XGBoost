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
# PAGE CONFIG
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
# MODEL CONFIGURATION
# Mengikuti struktur fitur dari notebook Colab.
# ============================================================

MODEL_OPTIONS = [
    "Hybrid TabNet-XGBoost",
    "Hybrid TabNet-XGBoost (Extreme)",
    "Hybrid TabNet-SVR",
]

MODEL_CONFIG = {
    "Hybrid TabNet-XGBoost": {
        "model_file": "hybrid_xgb.pkl",
        "scaler_file": None,
        "features": [
            "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1",
            "500_lag1", "KI_lag1", "850_lag4", "LI_lag1",
            "SI_lag1", "TT_lag3", "CAPE_lag3", "Height_lag1",
            "LFC_lag1", "Press0_lag7", "Temp0_lag7", "CH_lag1",
            "month", "month_sin", "month_cos",
        ],
    },

    "Hybrid TabNet-XGBoost (Extreme)": {
        "model_file": "hybrid_extreme.pkl",
        "scaler_file": None,
        "features": [
            "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1",
            "500_lag1", "KI_lag1", "850_lag1", "LI_lag1",
            "CCL_lag1", "SI_lag1", "CIN_lag1", "CAPE_lag1",
            "Press0_lag7", "BOYDEN_lag1", "Temp0_lag7", "CH_lag1",
            "month", "month_sin", "month_cos",
        ],
    },

    "Hybrid TabNet-SVR": {
        "model_file": "hybrid_svr.pkl",
        "scaler_file": "scaler_svr.pkl",
        "features": [
            "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1",
            "500_lag1", "KI_lag1", "850_lag4", "LI_lag1",
            "SI_lag1", "TT_lag3", "CIN_lag1", "Height_lag1",
            "Press0_lag7", "KO_lag1", "Temp0_lag7", "CH_lag1",
            "month", "month_sin", "month_cos",
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

    for name, config in MODEL_CONFIG.items():
        model_path = MODEL_DIR / config["model_file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}"
            )

        model = joblib.load(model_path)

        scaler = None
        if config["scaler_file"] is not None:
            scaler_path = MODEL_DIR / config["scaler_file"]

            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler tidak ditemukan: {scaler_path}"
                )

            scaler = joblib.load(scaler_path)

        loaded[name] = {
            "model": model,
            "scaler": scaler,
        }

    return loaded


@st.cache_data
def load_evaluation():
    files = {
        "Hybrid TabNet-XGBoost": "hasil_evaluasi.csv",
        "Hybrid TabNet-XGBoost (Extreme)": "hasil_evaluasi_ektrem.csv",
        "Hybrid TabNet-SVR": "hasil_evaluasi_svr.csv",
    }

    loaded = {}

    for name, filename in files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"File evaluasi tidak ditemukan: {path}"
            )

        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        loaded[name] = df

    return loaded


try:
    MODELS = load_models()
    EVALUATION_DATA = load_evaluation()
except Exception as exc:
    st.error("Model atau data evaluasi tidak dapat dimuat.")
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


def prepare_prediction_data(df):
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    if "Tanggal" not in data.columns:
        raise ValueError(
            "Kolom 'Tanggal' tidak ditemukan pada dataset."
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

    data = data.sort_values(
        "Tanggal"
    ).reset_index(drop=True)

    duplicate_count = int(
        data["Tanggal"].duplicated(keep=False).sum()
    )

    if "CH" not in data.columns:
        data["CH"] = np.nan

    # Ambil semua base feature yang dibutuhkan oleh tiga model.
    all_features = []
    for config in MODEL_CONFIG.values():
        all_features.extend(config["features"])

    base_features = set()

    for feature in all_features:
        base, lag = parse_lag(feature)

        if base is not None:
            base_features.add(base)

    # Buat lag persis sesuai kebutuhan model.
    for feature in sorted(set(all_features)):
        base, lag = parse_lag(feature)

        if base is not None:
            if base in data.columns:
                data[feature] = data[base].shift(lag)
            else:
                data[feature] = np.nan

    # Fitur musiman persis seperti pipeline Colab.
    data["month"] = data["Tanggal"].dt.month

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    return data, duplicate_count


# ============================================================
# MODEL VALIDATION
# ============================================================

def get_model_feature_names(model):
    names = getattr(
        model,
        "feature_names_in_",
        None
    )

    if names is None:
        return None

    return [str(x) for x in names]


def validate_model(model_name):
    model = MODELS[model_name]["model"]
    expected = MODEL_CONFIG[model_name]["features"]

    actual = get_model_feature_names(model)

    # Jika XGBoost menyimpan feature_names_in_, cocokkan.
    if actual is not None and actual != expected:
        raise ValueError(
            "Model .pkl tidak sesuai dengan pipeline Colab.\n\n"
            f"Model meminta:\n{actual}\n\n"
            f"Colab menggunakan:\n{expected}\n\n"
            "Silakan ekspor ulang model dari notebook Colab yang sesuai."
        )

    n_features = getattr(
        model,
        "n_features_in_",
        None
    )

    if n_features is not None and int(n_features) != len(expected):
        raise ValueError(
            f"Jumlah fitur model tidak sesuai. "
            f"Model: {n_features}, Colab: {len(expected)}."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_date(data, model_name, selected_date):
    config = MODEL_CONFIG[model_name]
    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]
    features = config["features"]

    validate_model(model_name)

    selected_date = pd.Timestamp(
        selected_date
    )

    matches = data.index[
        data["Tanggal"] == selected_date
    ].tolist()

    if not matches:
        raise ValueError(
            "Tanggal yang dipilih tidak ditemukan."
        )

    idx = matches[0]

    row = data.loc[
        [idx],
        features
    ].copy()

    missing = row.columns[
        row.isna().any()
    ].tolist()

    if missing:
        raise ValueError(
            "Fitur untuk tanggal tersebut belum lengkap:\n\n"
            + ", ".join(missing)
            + "\n\n"
            "Tanggal tersebut membutuhkan data beberapa hari sebelumnya "
            "sesuai lag model."
        )

    # XGBoost tidak menggunakan scaler pada pipeline Colab.
    if scaler is not None:
        try:
            scaler_names = getattr(
                scaler,
                "feature_names_in_",
                None
            )

            if scaler_names is not None:
                scaler_names = [
                    str(x) for x in scaler_names
                ]
                row_for_scaler = row[
                    scaler_names
                ]
            else:
                row_for_scaler = row

            X_input = scaler.transform(
                row_for_scaler
            )

        except Exception as exc:
            raise RuntimeError(
                "Scaling SVR gagal.\n\n"
                f"Fitur: {features}\n"
                f"Error: {exc}"
            ) from exc
    else:
        X_input = row

    try:
        prediction = model.predict(
            X_input
        )

    except Exception as exc:
        raise RuntimeError(
            f"Prediksi {model_name} gagal.\n\n"
            f"Jumlah fitur: {len(features)}\n"
            f"Fitur: {features}\n"
            f"Error: {exc}"
        ) from exc

    prediction = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return prediction, data.loc[idx]


# ============================================================
# EVALUATION HELPERS
# ============================================================

def normalize_evaluation_columns(df):
    result = df.copy()
    result.columns = [
        str(c).strip()
        for c in result.columns
    ]

    # Cari nama kolom tanggal secara fleksibel.
    date_candidates = [
        "Tanggal",
        "tanggal",
        "Date",
        "date",
        "DATE",
    ]

    date_col = next(
        (
            c for c in date_candidates
            if c in result.columns
        ),
        None
    )

    if date_col is not None:
        result["Tanggal"] = pd.to_datetime(
            result[date_col],
            errors="coerce"
        )
    else:
        # Jika file evaluasi tidak mempunyai tanggal,
        # gunakan nomor observasi agar visualisasi tetap berjalan.
        result["Tanggal"] = np.arange(
            1,
            len(result) + 1
        )

    required = ["Actual", "Prediction"]

    missing = [
        c for c in required
        if c not in result.columns
    ]

    if missing:
        raise ValueError(
            "Kolom evaluasi tidak lengkap. "
            f"Kolom yang tidak ditemukan: {missing}"
        )

    result["Actual"] = pd.to_numeric(
        result["Actual"],
        errors="coerce"
    )

    result["Prediction"] = pd.to_numeric(
        result["Prediction"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["Actual", "Prediction"]
    ).reset_index(drop=True)

    return result


def calculate_metrics(result):
    actual = result["Actual"]
    prediction = result["Prediction"]

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
            * 10
        )
    else:
        mape = np.nan

    return rmse, mae, mape


def dataframe_to_excel(df):
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
                pendekatan Hybrid TabNet dengan XGBoost dan SVR.
                Pilih model dan tanggal untuk memperoleh hasil prediksi
                dalam satuan milimeter (mm).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>📁 Data</h3>
                <p>
                    Gunakan dataset bersih dalam format Excel
                    sebagai data input.
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
                    Tersedia Hybrid TabNet-XGBoost,
                    Extreme, dan Hybrid TabNet-SVR.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>📊 Evaluation</h3>
                <p>
                    Lihat RMSE, MAE, MAPE, scatter,
                    residual, dan distribusi error.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cara Menggunakan")

    a, b, c = st.columns(3)

    with a:
        st.info("**01 — Upload**\n\nUpload dataset Excel.")

    with b:
        st.info("**02 — Pilih Tanggal**\n\nTentukan tanggal prediksi.")

    with c:
        st.info("**03 — Prediksi**\n\nHasil ditampilkan dalam mm.")

    st.success(
        "Silakan buka menu **Prediction** untuk memulai."
    )


# ============================================================
# PREDICTION
# Tidak ada plot sesuai permintaan.
# ============================================================

elif menu == "🌧️ Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Rainfall Prediction</h1>
            <p>
                Pilih model dan tanggal yang ingin diprediksi.
                Hasil akhir ditampilkan sebagai intensitas curah hujan
                dalam satuan milimeter (mm).
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
        help="Gunakan dataset bersih yang digunakan pada penelitian.",
    )

    if uploaded is not None:

        try:
            raw_df = pd.read_excel(
                uploaded
            )

            data, duplicate_count = prepare_prediction_data(
                raw_df
            )

        except Exception as exc:
            st.error(
                "Dataset tidak dapat diproses."
            )
            st.exception(exc)
            st.stop()

        if duplicate_count:
            st.warning(
                f"Ditemukan {duplicate_count} baris dengan tanggal duplikat. "
                "Sistem menggunakan baris pertama untuk tanggal yang sama."
            )

        features = MODEL_CONFIG[
            model_name
        ]["features"]

        # Pastikan semua base variable tersedia.
        required_base = set()

        for feature in features:
            base, lag = parse_lag(feature)

            if base is not None:
                required_base.add(base)

        missing_base = [
            col for col in sorted(required_base)
            if col not in data.columns
        ]

        if missing_base:
            st.error(
                "Variabel yang dibutuhkan model tidak tersedia:"
            )
            st.write(missing_base)
            st.stop()

        st.success(
            f"Dataset berhasil dimuat: "
            f"**{len(data):,} baris**."
        )

        valid_mask = data[
            features
        ].notna().all(axis=1)

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
                "Tidak ada tanggal yang memiliki data lengkap "
                "untuk model yang dipilih."
            )
            st.stop()

        st.markdown(
            "### Pilih Tanggal Prediksi"
        )

        selected_date = st.selectbox(
            "Tanggal",
            valid_dates,
            format_func=lambda x:
                pd.Timestamp(x).strftime(
                    "%d-%m-%Y"
                ),
        )

        if st.button(
            "🔮 Prediksi Intensitas Hujan",
            type="primary",
            use_container_width=True,
        ):

            try:
                prediction, selected_row = predict_date(
                    data,
                    model_name,
                    selected_date,
                )

                st.session_state[
                    "prediction_result"
                ] = {
                    "date": pd.Timestamp(
                        selected_date
                    ),
                    "model": model_name,
                    "prediction": prediction,
                    "actual": selected_row["CH"],
                }

            except Exception as exc:
                st.session_state[
                    "prediction_result"
                ] = None

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
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Prediksi Curah Hujan",
                        f'{result["prediction"]:.2f} mm',
                    )

                with c2:
                    if pd.notna(
                        result["actual"]
                    ):
                        st.metric(
                            "Curah Hujan Aktual",
                            f'{float(result["actual"]):.2f} mm',
                        )
                    else:
                        st.metric(
                            "Curah Hujan Aktual",
                            "Belum tersedia",
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

    try:
        result = normalize_evaluation_columns(
            EVALUATION_DATA[model_name]
        )

        rmse, mae, mape = calculate_metrics(
            result
        )

    except Exception as exc:
        st.error(
            "Data evaluasi tidak dapat diproses."
        )
        st.exception(exc)
        st.stop()

    st.markdown(
        "### Performance Metrics"
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

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

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
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=dataframe_to_excel(
                result
            ),
            file_name="hasil_evaluasi.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        "### Prediction Result — Testing Data"
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # 1. ACTUAL VS PREDICTION
    # --------------------------------------------------------

    st.markdown(
        "### 1. Actual vs Prediction"
    )

    plot_data = result[
        ["Tanggal", "Actual", "Prediction"]
    ].copy()

    plot_data = plot_data.melt(
        id_vars=["Tanggal"],
        value_vars=[
            "Actual",
            "Prediction"
        ],
        var_name="Series",
        value_name="Rainfall",
    )

    fig1 = px.line(
        plot_data,
        x="Tanggal",
        y="Rainfall",
        color="Series",
        markers=False,
        template="plotly_white",
    )

    fig1.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        legend_title="",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2. ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.markdown(
        "### 2. Actual vs Predicted"
    )

    scatter = result[
        ["Actual", "Prediction"]
    ].dropna()

    fig2 = px.scatter(
        scatter,
        x="Actual",
        y="Prediction",
        opacity=0.70,
        template="plotly_white",
        labels={
            "Actual":
                "Actual Rainfall (mm)",
            "Prediction":
                "Predicted Rainfall (mm)",
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
            line=dict(
                dash="dash"
            ),
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
    # 3. RESIDUAL
    # --------------------------------------------------------

    st.markdown(
        "### 3. Residual Error"
    )

    residual = result[
        ["Tanggal", "Actual", "Prediction"]
    ].copy()

    residual["Residual"] = (
        residual["Actual"]
        - residual["Prediction"]
    )

    fig3 = px.line(
        residual,
        x="Tanggal",
        y="Residual",
        template="plotly_white",
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

    st.markdown(
        "### 4. Distribution of Prediction Error"
    )

    fig4 = px.histogram(
        residual,
        x="Residual",
        nbins=30,
        template="plotly_white",
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
