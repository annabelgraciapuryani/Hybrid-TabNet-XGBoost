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

MODEL_OPTIONS = [
    "Hybrid TabNet-XGBoost",
    "Hybrid TabNet-XGBoost (Extreme)",
    "Hybrid TabNet-SVR",
]

# ============================================================
# FITUR FINAL - DIAMBIL DARI OUTPUT COLAB PENGGUNA
# ============================================================

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
            "500_lag1", "KI_lag1", "850_lag1", "CCL_lag1",
            "CAPE_lag1", "Height_lag3", "LFC_lag1", "SWEAT_lag4",
            "Press0_lag7", "MVV_lag2", "Temp0_lag7", "CH_lag1",
            "month", "month_sin", "month_cos",
        ],
    },

    "Hybrid TabNet-SVR": {
        "model_file": "hybrid_svr.pkl",
        "scaler_file": "scaler_svr.pkl",
        "features": [
            "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1",
            "500_lag1", "KI_lag1", "850_lag4", "LI_lag1",
            "SI_lag1", "TT_lag3", "CAPE_lag3", "Height_lag1",
            "LFC_lag1", "Press0_lag7", "Temp0_lag7", "CH_lag1",
            "month", "month_sin", "month_cos",
        ],
    },
}

# Semua base variable yang mungkin diperlukan oleh ketiga model.
ALL_BASE_FEATURES = set()

for config in MODEL_CONFIG.values():
    for feature in config["features"]:
        match = re.fullmatch(
            r"(.+)_lag([1-9][0-9]*)",
            feature
        )
        if match:
            ALL_BASE_FEATURES.add(match.group(1))

ALL_BASE_FEATURES = sorted(ALL_BASE_FEATURES)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5faff 0%, #ffffff 50%, #f6f9fc 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2942 0%, #125174 100%);
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
        font-size: 2.55rem;
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

        if config["scaler_file"]:
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
                f"Data evaluasi tidak ditemukan: {path}"
            )

        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        loaded[name] = df

    return loaded


try:
    MODELS = load_models()
    EVALUATION_DATA = load_evaluation()
except Exception as exc:
    st.error("Aplikasi gagal memuat model atau data evaluasi.")
    st.exception(exc)
    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def parse_lag_feature(feature):
    match = re.fullmatch(
        r"(.+)_lag([1-9][0-9]*)",
        str(feature)
    )

    if not match:
        return None, None

    return match.group(1), int(match.group(2))


def prepare_prediction_data(df):
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

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

    # Konversi seluruh kolom input numerik yang tersedia.
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

    duplicate_count = int(
        data["Tanggal"].duplicated(keep=False).sum()
    )

    # Buat lag berdasarkan union kebutuhan semua model.
    all_model_features = sorted({
        feature
        for config in MODEL_CONFIG.values()
        for feature in config["features"]
    })

    for feature in all_model_features:
        base, lag = parse_lag_feature(feature)

        if base is not None:
            if base not in data.columns:
                data[feature] = np.nan
            else:
                data[feature] = data[base].shift(lag)

    # Fitur musiman sama dengan Colab.
    data["month"] = data["Tanggal"].dt.month
    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    return data, duplicate_count


def get_valid_dates(data, features):
    mask = data[features].notna().all(axis=1)

    return (
        data.loc[mask, "Tanggal"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


# ============================================================
# MODEL VALIDATION
# Hanya untuk memberi peringatan jika .pkl benar-benar berbeda.
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

    if actual is not None and actual != expected:
        raise ValueError(
            "Model .pkl tidak sesuai dengan fitur Colab.\n\n"
            f"Model meminta:\n{actual}\n\n"
            f"Colab menggunakan:\n{expected}\n\n"
            "Ekspor ulang model dari notebook yang sama."
        )

    n_features = getattr(
        model,
        "n_features_in_",
        None
    )

    if n_features is not None:
        if int(n_features) != len(expected):
            raise ValueError(
                f"Jumlah fitur model tidak sesuai. "
                f"Model: {n_features}; "
                f"Colab: {len(expected)}."
            )


# ============================================================
# PREDICTION
# ============================================================

def predict_for_date(data, model_name, selected_date):
    config = MODEL_CONFIG[model_name]
    features = config["features"]

    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]

    validate_model(model_name)

    selected_date = pd.Timestamp(selected_date)

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
            "Tanggal tersebut membutuhkan data historis "
            "sesuai lag model."
        )

    # XGBoost / Extreme: langsung ke model.
    if scaler is None:
        X_input = row

    # SVR: gunakan scaler yang memang dibuat untuk svr_features.
    else:
        scaler_names = getattr(
            scaler,
            "feature_names_in_",
            None
        )

        if scaler_names is not None:
            scaler_names = [
                str(x) for x in scaler_names
            ]

            if scaler_names != features:
                raise ValueError(
                    "scaler_svr.pkl tidak sesuai dengan svr_features Colab.\n\n"
                    f"Scaler meminta:\n{scaler_names}\n\n"
                    f"Colab menggunakan:\n{features}\n\n"
                    "Ekspor ulang hybrid_svr.pkl dan scaler_svr.pkl "
                    "dari notebook SVR yang sama."
                )

            X_input = scaler.transform(
                row[scaler_names]
            )
        else:
            X_input = scaler.transform(row)

    try:
        prediction = model.predict(
            X_input
        )
    except Exception as exc:
        raise RuntimeError(
            f"Prediksi {model_name} gagal.\n\n"
            f"Fitur: {features}\n"
            f"Shape input: {getattr(X_input, 'shape', None)}\n"
            f"Error: {exc}"
        ) from exc

    prediction = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return prediction, data.loc[idx]


# ============================================================
# EVALUATION
# ============================================================

def normalize_evaluation(df):
    result = df.copy()
    result.columns = [
        str(c).strip()
        for c in result.columns
    ]

    if "Tanggal" in result.columns:
        result["Tanggal"] = pd.to_datetime(
            result["Tanggal"],
            errors="coerce"
        )
    else:
        # Hindari crash Plotly jika file tidak menyimpan tanggal.
        result["Tanggal"] = np.arange(
            1,
            len(result) + 1
        )

    for col in ["Actual", "Prediction"]:
        if col not in result.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan pada file evaluasi."
            )

        result[col] = pd.to_numeric(
            result[col],
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

    mask = actual != 0

    if mask.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        actual[mask]
                        - prediction[mask]
                    )
                    / actual[mask]
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
                Sistem prediksi intensitas curah hujan harian berbasis
                Hybrid TabNet dengan XGBoost dan SVR.
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
                    Upload dataset dalam format Excel
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
                    Hybrid TabNet-XGBoost,Hybrid TabNet-XGBoost dengan penanganan data
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
                    Evaluasi RMSE, MAE, MAPE.
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
        st.info("**02 — Pilih**\n\nPilih model dan tanggal.")

    with c:
        st.info("**03 — Prediksi**\n\nTertampil intensitas hujan dalam mm.")


# ============================================================
# PREDICTION
# TANPA PLOT
# ============================================================

elif menu == "🌧️ Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Rainfall Prediction</h1>
            <p>
                Pilih model, upload dataset, kemudian pilih tanggal
                yang ingin diprediksi.
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

        # Periksa base variable yang diperlukan.
        required_base = set()

        for feature in features:
            base, lag = parse_lag_feature(feature)

            if base is not None:
                required_base.add(base)

        missing_base = [
            col for col in sorted(required_base)
            if col not in data.columns
        ]

        if missing_base:
            st.error(
                "Variabel input yang dibutuhkan model tidak tersedia:"
            )
            st.write(missing_base)
            st.stop()

        valid_dates = get_valid_dates(
            data,
            features
        )

        if not valid_dates:
            st.error(
                "Tidak ada tanggal yang memiliki seluruh fitur "
                "yang dibutuhkan model."
            )
            st.stop()

        st.success(
            f"Dataset berhasil dimuat: **{len(data):,} baris**."
        )

        st.markdown("### Pilih Tanggal Prediksi")

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
                prediction, selected_row = predict_for_date(
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
                        f'{result["prediction"]:.2f} mm'
                    )

                with c2:
                    if pd.notna(
                        result["actual"]
                    ):
                        st.metric(
                            "Curah Hujan Aktual",
                            f'{float(result["actual"]):.2f} mm'
                        )
                    else:
                        st.metric(
                            "Curah Hujan Aktual",
                            "Belum tersedia"
                        )


# ============================================================
# MODEL EVALUATION
# HANYA SATU VISUALISASI
# ============================================================

elif menu == "📊 Model Evaluation":

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Evaluation</h1>
            <p>
                Evaluasi performa model pada data pengujian.
                Tersedia RMSE, MAE, MAPE, tabel hasil prediksi,
                download data, dan visualisasi Actual vs Prediction.
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
        result = normalize_evaluation(
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

    st.markdown("### Performance Metrics")

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

    # Download tetap di Model Evaluation.
    st.markdown("### Download Hasil Evaluasi")

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

    # ========================================================
    # SATU-SATUNYA GRAFIK
    # ========================================================

    st.markdown("### Actual vs Prediction")

    # Explicit long-form untuk menghindari error Plotly
    # pada y=["Actual", "Prediction"].
    plot_data = result[
        ["Tanggal", "Actual", "Prediction"]
    ].copy()

    if not pd.api.types.is_datetime64_any_dtype(
        plot_data["Tanggal"]
    ):
        plot_data["Tanggal"] = np.arange(
            1,
            len(plot_data) + 1
        )

    plot_data = plot_data.melt(
        id_vars=["Tanggal"],
        value_vars=[
            "Actual",
            "Prediction"
        ],
        var_name="Series",
        value_name="Rainfall",
    )

    plot_data["Rainfall"] = pd.to_numeric(
        plot_data["Rainfall"],
        errors="coerce"
    )

    plot_data = plot_data.dropna(
        subset=["Rainfall"]
    )

    fig = px.line(
        plot_data,
        x="Tanggal",
        y="Rainfall",
        color="Series",
        template="plotly_white",
    )

    fig.update_layout(
        height=520,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        legend_title="",
        hovermode="x unified",
    )

    fig.update_traces(
        line=dict(width=2)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
