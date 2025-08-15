from __future__ import annotations

import os
import glob
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
import requests
import streamlit as st

# Optional: only used if a raw xgboost Booster is provided
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import librosa
except Exception:
    st.error("librosa is required. Add 'librosa' to requirements.txt and redeploy.")
    raise

APP_NAME = "MilkCrate – Genre Classifier"
MODELS_DIR = Path("models")
DEFAULT_MODEL_FILENAME = "model_version3beatport.joblib"
DEFAULT_ENCODER_FILENAME = "label_encoder.pkl"
DEFAULT_SR = 22050


def ensure_models_dir() -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR)


def is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs.github.com/spec" in head
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _download_bytes(url: str, timeout: int = 300) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def download_file(url: str, dest: Path) -> None:
    data = _download_bytes(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)


@st.cache_resource(show_spinner=False)
def load_model(model_filename: str, model_url: str = ""):
    """
    Load a scikit-learn/xgboost model from ./models or download from URL if missing.
    """
    ensure_models_dir()
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        if model_url:
            with st.spinner("Downloading model from URL…"):
                download_file(model_url, model_path)
        else:
            available = [Path(p).name for p in glob.glob(str(MODELS_DIR / "*"))]
            raise FileNotFoundError(
                f"Missing model at {model_path} and no URL provided. "
                f"Found in ./models: {available}"
            )

    if is_lfs_pointer(model_path):
        raise FileNotFoundError(
            f"Model file at {model_path} appears to be a Git LFS pointer, not the real artifact."
        )

    # Support .joblib/.pkl via joblib; .json for raw Booster (optional)
    if model_path.suffix.lower() in {".joblib", ".pkl"}:
        return joblib.load(model_path)
    elif model_path.suffix.lower() == ".json" and xgb is not None:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster
    else:
        # Fallback: try joblib anyway
        return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_label_encoder(encoder_filename: str = DEFAULT_ENCODER_FILENAME):
    path = MODELS_DIR / encoder_filename
    if not path.exists():
        return None
    if is_lfs_pointer(path):
        st.warning("Label encoder appears to be an LFS pointer; ignoring.")
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def extract_features_34(
    path: str,
    sr: int = DEFAULT_SR,
    seconds: Optional[float] = None,
) -> np.ndarray:
    """
    34-D features: 13 MFCC means + 12 Chroma means + 7 Spectral contrast means + ZCR mean + Rolloff mean.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)

    if seconds is not None and seconds > 0:
        y = y[: int(seconds * sr)]

    n_fft = 2048
    hop = 512

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop).mean(axis=1)  # 13
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean(axis=1)    # 12
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean(axis=1)  # 7
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop).mean()                           # 1
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).mean()     # 1

    feat = np.hstack([mfcc, chroma, contrast, [zcr], [rolloff]])  # -> (34,)
    if feat.shape[0] != 34:
        raise ValueError(f"Feature shape mismatch at extraction time: expected 34, got {feat.shape[0]}")
    return feat.astype(np.float32)


def predict_label(model, X: np.ndarray, label_encoder=None) -> Tuple[str, Optional[np.ndarray]]:
    """
    Return (label, proba_vector_optional).
    Handles sklearn-style estimators and raw xgboost Booster.
    """
    # Raw xgboost Booster case
    if xgb is not None and isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X)
        proba = model.predict(dmat)
        if proba.ndim == 1:
            y = (proba > 0.5).astype(int)
        else:
            y = np.argmax(proba, axis=1)
        label_val = int(y[0])
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                label_val = label_encoder.inverse_transform([label_val])[0]
            except Exception:
                pass
        return str(label_val), proba[0] if isinstance(proba, np.ndarray) else None

    # sklearn-like path
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        y = np.argmax(proba, axis=1)
        label_idx = int(y[0])
        label_val = label_idx
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                label_val = label_encoder.inverse_transform([label_idx])[0]
            except Exception:
                pass
        return str(label_val), proba[0]
    else:
        y = model.predict(X)
        label_val = y[0]
        if isinstance(label_val, (np.floating, np.integer)):
            label_val = int(label_val)
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                label_val = label_encoder.inverse_transform([int(label_val)])[0]
            except Exception:
                pass
        return str(label_val), None


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🎧", layout="centered")
    st.title(APP_NAME)
    st.caption("Classify tracks by genre using your trained model.")

    st.sidebar.header("Model settings")
    model_filename = st.sidebar.text_input(
        "Model filename (in ./models)",
        value=DEFAULT_MODEL_FILENAME,
        help="Place the model file inside the ./models folder.",
    )
    model_url = st.sidebar.text_input(
        "Model URL (optional)",
        value="",
        help="If provided and the file is missing locally, it will be downloaded and cached.",
    )
    encoder_filename = st.sidebar.text_input(
        "Label encoder filename (in ./models)",
        value=DEFAULT_ENCODER_FILENAME,
    )
    seconds = st.sidebar.number_input(
        "Analyze first N seconds (0 = full track)",
        min_value=0,
        max_value=600,
        value=30,
        step=5,
        help="Use 0 to analyze the full file (slower).",
    )
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # Load model + encoder
    try:
        model = load_model(model_filename, model_url)
    except FileNotFoundError as e:
        st.error(str(e))
        with st.expander("Files in ./models"):
            files = sorted(Path("models").glob("*"))
            if files:
                for p in files:
                    st.write(f"- {p.name} ({p.stat().st_size/1024:.1f} KB)")
            else:
                st.write("No files found.")
        st.stop()

    label_encoder = load_label_encoder(encoder_filename)

    expected = getattr(model, "n_features_in_", None)
    if show_debug:
        st.sidebar.write(
            {
                "expected_n_features": int(expected) if expected is not None else None,
                "model_type": type(model).__name__,
                "cwd": os.getcwd(),
            }
        )

    st.subheader("Upload audio")
    uploads = st.file_uploader(
        "Drop audio files here (mp3/wav/flac/ogg/m4a)",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Upload one or more audio files to classify.")
        return

    results = []
    for up in uploads:
        try:
            # Save to a temp file so librosa can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{up.name}") as tmp:
                tmp.write(up.read())
                tmp_path = tmp.name

            feat = extract_features_34(tmp_path, sr=DEFAULT_SR, seconds=seconds if seconds > 0 else None)
            X = np.atleast_2d(feat)

            if expected is not None and X.shape[1] != int(expected):
                st.error(f"Feature shape mismatch for {up.name}: expected {expected}, got {X.shape[1]}")
                continue

            label, proba = predict_label(model, X, label_encoder)

            results.append((up.name, label, proba.max().item() if isinstance(proba, np.ndarray) else None))
            st.success(
                f"**{up.name}** → **{label}**"
                + (f" (conf {proba.max():.2f})" if isinstance(proba, np.ndarray) else "")
            )

        except Exception as e:
            st.error(f"Failed to process {up.name}: {e}")

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if results:
        st.subheader("Results")
        try:
            import pandas as pd  # optional
            df = pd.DataFrame(results, columns=["file", "predicted_genre", "confidence"])
            st.dataframe(df, use_container_width=True)
        except Exception:
            for fname, label, conf in results:
                st.write(f"- {fname} → {label}" + (f" (conf {conf:.2f})" if conf is not None else ""))


if __name__ == "__main__":
    main()
