import os
import io
import tempfile
import glob
import warnings
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st

# Optional deps used when available
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import requests
except Exception:
    requests = None

# =============================
# Config & Constants
# =============================
APP_TITLE = "MilkCrate – DJ Genre Classifier"
MODELS_DIR = "models"
DEFAULT_MODEL_FILENAME = "model_version3beatport.joblib"  # <— matches the file you committed (~3.8MB)
DEFAULT_LABEL_ENCODER = "label_encoder.pkl"
DEFAULT_SR = 22050

# =============================
# Utility: small helpers
# =============================

def ensure_models_dir() -> str:\n    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def list_models() -> List[str]:
    ensure_models_dir()
    files = []
    for ext in ("*.joblib", "*.pkl", "*.json", "*.ubj", "*.bst"):
        files.extend(sorted(glob.glob(os.path.join(MODELS_DIR, ext))))
    return [os.path.basename(p) for p in files]


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs.github.com/spec" in head
    except Exception:
        return False


# =============================
# Feature Extraction (34 dims)
# =============================

def extract_features_34(path: str, sr: int = DEFAULT_SR, seconds: Optional[float] = None) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required for feature extraction. Add 'librosa' to requirements.txt")

    y, sr = librosa.load(path, sr=sr, mono=True)

    if seconds is not None and seconds > 0:
        y = y[: int(seconds * sr)]

    n_fft = 2048
    hop = 512

    # 13 MFCC means
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop).mean(axis=1)
    # 12 chroma means
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean(axis=1)
    # 7 spectral contrast means
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean(axis=1)
    # 1 zero-crossing rate mean
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop).mean()
    # 1 spectral rolloff mean
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean()

    feat = np.hstack([mfcc, chroma, contrast, [zcr], [rolloff]])
    if feat.shape[0] != 34:
        raise ValueError(f"Feature length {feat.shape[0]} != 34 — check extractor")
    return feat.astype(np.float32)


# =============================
# Loading: Label Encoder
# =============================
@st.cache_resource(show_spinner=False)
def load_label_encoder(le_filename: str = DEFAULT_LABEL_ENCODER):
    ensure_models_dir()
    path = os.path.join(MODELS_DIR, le_filename)
    if not os.path.exists(path):
        st.warning(f"Label encoder '{le_filename}' not found in ./{MODELS_DIR}. Class names will be numeric.")
        return None
    if is_lfs_pointer(path):
        st.error(f"'{le_filename}' looks like a Git LFS pointer. Commit real file or load from URL.")
        st.stop()
    if joblib is None:
        raise RuntimeError("joblib is required. Add 'joblib' to requirements.txt")
    return joblib.load(path)


# =============================
# Loading: Model
# =============================

def _load_model_any(path: str):
    """Try joblib/pickle first. If that fails and XGBoost is available,
    try loading Booster formats (.json/.ubj/.bst). Returns a model-like object.
    """
    if is_lfs_pointer(path):
        raise FileNotFoundError(
            f"'{os.path.basename(path)}' is a Git LFS pointer, not the real model. Replace with real file or use a URL."
        )

    # 1) joblib/pickle path
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception as e:
            # fall through to possible XGBoost booster
            last = e
    else:
        last = RuntimeError("joblib not installed")

    # 2) Native XGBoost Booster
    if xgb is not None and os.path.splitext(path)[1].lower() in {".json", ".ubj", ".bst"}:
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            return booster
        except Exception as e:
            last = e

    raise RuntimeError(f"Could not load model from {path}: {last}")


@st.cache_resource(show_spinner=True)
def load_model(model_filename: str, model_url: Optional[str] = None):
    ensure_models_dir()
    local_path = os.path.join(MODELS_DIR, model_filename)

    if os.path.exists(local_path):
        return _load_model_any(local_path)

    if not model_url:
        # match previous behavior but give actionable info in UI
        raise FileNotFoundError(
            f"Missing model at\n\n{local_path}\n\nProvide a valid file in './{MODELS_DIR}' or a download URL in the sidebar."
        )

    if requests is None:
        raise RuntimeError("'requests' is required to download a model from URL. Add it to requirements.txt.")

    # Download to models dir
    with st.spinner("Downloading model..."):
        r = requests.get(model_url, timeout=300)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return _load_model_any(local_path)


# =============================
# Prediction helpers
# =============================

def predict_with_model(model, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (pred_indices, proba_or_none). Handles sklearn & xgboost.Booster."""
    # Ensure 2D
    if X.ndim == 1:
        X = X[np.newaxis, :]

    # Guard against feature mismatch when possible
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(f"Feature shape mismatch, expected: {expected}, got {X.shape[1]}")

    # sklearn-like (XGBClassifier via scikit API, etc.)
    if hasattr(model, "predict"):
        try:
            y_pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            return y_pred, proba
        except Exception:
            pass

    # xgboost.Booster path
    if xgb is not None and isinstance(model, getattr(xgb, "Booster", ())):
        dmat = xgb.DMatrix(X)
        raw = model.predict(dmat)  # shape: (n, n_classes) for multi-class
        if raw.ndim == 2:
            idx = np.argmax(raw, axis=1)
            return idx, raw
        else:
            # binary case returns shape (n,)
            idx = (raw > 0.5).astype(int)
            proba = np.vstack([1 - raw, raw]).T
            return idx, proba

    # Fallback
    raise RuntimeError("Unsupported model type for prediction.")


# =============================
# UI
# =============================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎛️", layout="centered")
    st.title(APP_TITLE)
    st.caption("Organize untagged music into genres / sub-genres with ML.")

    with st.sidebar:
        st.header("Settings")
        st.write("**Model source**")
        available = list_models()

        model_filename = st.text_input(
            "Model filename (in ./models)",
            value=DEFAULT_MODEL_FILENAME if DEFAULT_MODEL_FILENAME in available or True else (available[0] if available else DEFAULT_MODEL_FILENAME),
            help=f"Put the model file inside './{MODELS_DIR}' or provide a direct download URL below.",
        )
        model_url = st.text_input(
            "Model URL (optional)",
            value="",
            help="If provided, the file will be downloaded to ./models when missing.",
        ).strip() or None

        label_filename = st.text_input(
            "Label encoder filename (in ./models)",
            value=DEFAULT_LABEL_ENCODER,
        )

        analyze_seconds = st.number_input(
            "Analyze first N seconds (0 = full file)",
            min_value=0, max_value=600, value=60, step=5,
        )

        st.divider()
        with st.expander("Debug info"):
            st.write({
                "models_dir": MODELS_DIR,
                "available_files": available,
                "cwd": os.getcwd(),
            })

    # Load model & label encoder
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hush xgboost pickle warnings
            model = load_model(model_filename, model_url)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    label_encoder = load_label_encoder(label_filename)

    st.subheader("Classify a track")
    uploaded = st.file_uploader("Upload an audio file (mp3/wav/flac/m4a/ogg)", type=["mp3","wav","flac","m4a","ogg"], accept_multiple_files=False)

    if uploaded is None:
        st.info("Upload a file to start.")
        return

    # Persist upload to a temp file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    st.write(f"**File:** {uploaded.name}")

    try:
        feat = extract_features_34(tmp_path, sr=DEFAULT_SR, seconds=analyze_seconds if analyze_seconds > 0 else None)
        X = np.atleast_2d(feat)

        # Preview feature shape and expected input
        expected = getattr(model, "n_features_in_", None)
        st.write({"runtime_feature_len": int(X.shape[1]), "model_n_features_in": int(expected) if expected is not None else None})

        y_pred, proba = predict_with_model(model, X)
        pred_idx = int(y_pred[0]) if hasattr(y_pred, "__len__") else int(y_pred)

        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                label = label_encoder.inverse_transform([pred_idx])[0]
            except Exception:
                label = str(pred_idx)
        else:
            label = str(pred_idx)

        st.success(f"Prediction: **{label}**")

        # Show top-k probabilities when available
        if proba is not None and proba.ndim == 2:
            probs = proba[0]
            # Build a display of top classes
            top_k = min(5, probs.shape[0])
            order = np.argsort(probs)[::-1][:top_k]
            top_rows = []
            for i in order:
                name = label_encoder.inverse_transform([i])[0] if (label_encoder is not None and hasattr(label_encoder, "inverse_transform")) else str(i)
                top_rows.append((name, float(probs[i])))
            st.write("Top classes:")
            for name, p in top_rows:
                st.write(f"• {name}: {p:.3f}")

    except ValueError as ve:
        # Common case: feature length mismatch
        st.error(str(ve))
    except Exception as e:
        st.exception(e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
