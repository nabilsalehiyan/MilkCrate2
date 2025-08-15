# app.py
# MilkCrate (Beatport) — Streamlit app
# - Robust imports (no importlib file-location hacks)
# - Loads model from local file OR downloads from a provided URL
# - Classifies uploaded audio files and can export organized ZIP
# - Uses @st.cache_resource to keep deploys snappy on Streamlit Cloud

from __future__ import annotations

import io
import os
import zipfile
import shutil
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Optional heavy imports wrapped so the app still boots with a friendly hint
try:
    import librosa
except Exception as e:
    librosa = None
    _librosa_err = e

# ---------------------------
# Configuration
# ---------------------------

ROOT = pathlib.Path(__file__).parent.resolve()
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default model file & a placeholder URL (change this to your real Release URL)
DEFAULT_MODEL_NAME = "model_version3beatport.joblib"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# You can set this as an env var on Streamlit Cloud (Secrets) or edit here.
DEFAULT_MODEL_URL = os.environ.get(
    "MILKCRATE_MODEL_URL",
    # Replace this with your actual GitHub Release asset URL:
    "https://github.com/nabilsalehiyan/milkcrate2/releases/download/v1/model_version3beatport.joblib"
)

# If your model expects a specific sample rate / duration for features:
TARGET_SR = 22050
DEFAULT_MAX_DURATION_S = 60  # cap analysis to first N seconds to speed up


# ---------------------------
# UI helpers / Types
# ---------------------------

st.set_page_config(page_title="MilkCrate (Beatport Model)", page_icon="🍼", layout="wide")

@dataclass
class Prediction:
    filename: str
    label: str
    confidence: float


# ---------------------------
# Model loading
# ---------------------------

def _download_file(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)

@st.cache_resource(show_spinner=True)
def load_model(model_path: pathlib.Path, fallback_url: Optional[str]) -> object:
    """
    Loads a joblib model. If not present and a URL is provided, downloads it.
    """
    if not model_path.exists():
        if fallback_url:
            st.info(f"Model not found at `{model_path}` — attempting download…")
            try:
                _download_file(fallback_url, model_path)
            except Exception as e:
                st.error(
                    "Couldn't download the model. "
                    "Set MILKCRATE_MODEL_URL in environment/secrets or place the file locally."
                )
                raise
        else:
            raise FileNotFoundError(f"Missing model at {model_path} and no URL provided.")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path} — is it a valid joblib file?")
        raise
    return model


# ---------------------------
# Feature extraction
# ---------------------------

def load_audio_bytes_to_mono(
    data: bytes,
    sr: int = TARGET_SR,
    max_duration_s: int = DEFAULT_MAX_DURATION_S
) -> np.ndarray:
    if librosa is None:
        raise RuntimeError(
            "librosa is not available. Add 'librosa' and 'soundfile' to requirements.txt "
            f"(import error: {_librosa_err})"
        )
    # librosa.load accepts file-like via soundfile backend
    with io.BytesIO(data) as bio:
        y, _ = librosa.load(bio, sr=sr, mono=True, duration=max_duration_s)
    return y

def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Adjust to match the features your model was trained on.
    Here: log-mel spectrogram summary (means + stds).
    """
    if librosa is None:
        raise RuntimeError("librosa not available for feature extraction.")
    # Log-mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S + 1e-10)
    # Summaries (mean & std over time) -> shape (2 * n_mels,)
    feat_mean = S_db.mean(axis=1)
    feat_std = S_db.std(axis=1)
    feat = np.concatenate([feat_mean, feat_std], axis=0).astype(np.float32)
    # Reshape to (1, n_features) for scikit-learn style models
    return feat.reshape(1, -1)


# ---------------------------
# Inference
# ---------------------------

def predict_one(model: object, filename: str, raw_bytes: bytes, duration_s: int) -> Prediction:
    y = load_audio_bytes_to_mono(raw_bytes, sr=TARGET_SR, max_duration_s=duration_s)
    X = extract_features(y, sr=TARGET_SR)

    # Try scikit-learn API: predict_proba if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        label = _label_from_model(model, idx)
        conf = float(proba[idx])
    else:
        # Fallback to predict only
        pred = model.predict(X)
        # pred might be index or label depending on how it was saved
        if isinstance(pred[0], (int, np.integer)):
            label = _label_from_model(model, int(pred[0]))
        else:
            label = str(pred[0])
        conf = 1.0  # unknown

    return Prediction(filename=filename, label=label, confidence=conf)

def _label_from_model(model: object, idx: int) -> str:
    """
    Resolve a human-readable label from the model. Adjust to your training pipeline.
    Commonly stored as model.classes_ in sklearn classifiers.
    """
    if hasattr(model, "classes_"):
        classes = getattr(model, "classes_")
        try:
            return str(classes[idx])
        except Exception:
            pass
    # Fallback: just return index
    return f"class_{idx}"


# ---------------------------
# Streamlit UI
# ---------------------------

def sidebar_controls() -> Tuple[pathlib.Path, Optional[str], bool, int]:
    st.sidebar.header("⚙️ Settings")

    model_path_str = st.sidebar.text_input(
        "Model filename (in ./models)",
        value=str(DEFAULT_MODEL_PATH.name)
    )
    model_path = (MODELS_DIR / model_path_str).resolve()

    model_url = st.sidebar.text_input(
        "Model URL (optional; used if file is missing)",
        value=str(DEFAULT_MODEL_URL or "")
    ).strip()
    if model_url == "":
        model_url = None

    duration_s = st.sidebar.number_input(
        "Analyze up to first N seconds",
        min_value=5, max_value=300, value=DEFAULT_MAX_DURATION_S, step=5
    )

    debug = st.sidebar.toggle("Debug mode", value=False, help="Show repository root contents and env info.")
    return model_path, model_url, debug, int(duration_s)

def main():
    st.title("🍼 MilkCrate — Beatport Classifier")
    st.caption("Organize your music by genres/sub-genres using your ML model.")

    model_path, model_url, debug, duration_s = sidebar_controls()

    if debug:
        st.write("**Repo root**:", str(ROOT))
        st.write("**Root entries**:", [p.name for p in ROOT.iterdir()])
        st.write("**Models dir**:", str(MODELS_DIR))
        st.write("**Env MILKCRATE_MODEL_URL**:", os.environ.get("MILKCRATE_MODEL_URL"))

    # Load model (cached)
    with st.spinner("Loading model…"):
        model = load_model(model_path, model_url)

    st.success(f"Model loaded: `{model_path.name}`")

    st.subheader("1) Upload audio files")
    files = st.file_uploader(
        "Drag & drop or browse",
        type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
        accept_multiple_files=True
    )

    organize_zip = st.checkbox("Create ZIP organized by predicted genres", value=False)

    results: List[Prediction] = []

    if files:
        st.subheader("2) Results")
        progress = st.progress(0.0)
        tmp_for_zip = pathlib.Path(st.session_state.get("_mc_tmp_dir", str(ROOT / ".tmp_uploads")))
        tmp_for_zip.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(files, start=1):
            data = f.read()
            try:
                pred = predict_one(model, f.name, data, duration_s=duration_s)
                results.append(pred)
            except Exception as e:
                st.error(f"Failed to process `{f.name}`: {e}")
            finally:
                # store the raw file if we might need to zip later
                if organize_zip:
                    out = tmp_for_zip / f.name
                    with open(out, "wb") as w:
                        w.write(data)

            progress.progress(i / len(files))

        if results:
            df = pd.DataFrame([{"file": r.filename, "predicted_genre": r.label, "confidence": r.confidence} for r in results])
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv, "milkcrate_predictions.csv", "text/csv")

            if organize_zip:
                # Build a ZIP with subfolders per predicted label
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    for r in results:
                        src = tmp_for_zip / r.filename
                        if not src.exists():
                            continue
                        arcname = f"{r.label}/{r.filename}"
                        z.write(src, arcname)
                zip_buf.seek(0)
                st.download_button("Download organized ZIP", zip_buf, "milkcrate_organized.zip", "application/zip")

                # cleanup temp files
                try:
                    shutil.rmtree(tmp_for_zip, ignore_errors=True)
                except Exception:
                    pass

        else:
            st.info("No predictions produced yet. Upload supported audio files to begin.")

    st.divider()
    with st.expander("Having trouble?"):
        st.markdown(
            """
- Ensure your model file is present at `./models/` or set a **Model URL** in the sidebar.
- If you trained with different features than the default log-mel summary, update `extract_features(...)` accordingly.
- To speed up first-time loads, host the model on GitHub Releases and provide the asset URL via `MILKCRATE_MODEL_URL`.
- Debug mode (sidebar) shows what files the app can actually see in Streamlit Cloud.
            """
        )

if __name__ == "__main__":
    main()
