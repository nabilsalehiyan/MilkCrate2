# app.py
# Streamlit app for MilkCrate (Beatport model). Adds "Option A" verification sidebar.
# - Reads models/config.json to pick the active model
# - Loads joblib model
# - Tries to load a LabelEncoder or JSON map to convert indices → genre names
# - Batch audio upload, predicts per file
# - Sidebar shows which model file is loaded + metadata so you can verify Beatport is in use

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional deps for audio → features
import soundfile as sf
import librosa


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MilkCrate — Beatport Classifier",
    layout="wide",
    page_icon="🎚️",
)

st.title("Upload audio")

st.caption("Drop audio files here (**mp3/wav/flac/ogg/m4a**). Limit ~200MB per file.")


# -----------------------------
# Paths & config
# -----------------------------
MODELS_DIR = Path("models")
CONFIG_PATH = MODELS_DIR / "config.json"

# Fallbacks in case user hasn't set config yet
DEFAULT_ACTIVE_KEY = "version3beatport"
DEFAULT_MODEL_FILE = "model_version3beatport.joblib"

# Where we might keep a JSON label map like: { "0": "tech_house", ... }
DEFAULT_JSON_MAP = Path("artifacts/beatport/genre_id_to_name.json")


# -----------------------------
# Utilities
# -----------------------------
def _read_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    # fallback
    return {
        "active_model": DEFAULT_ACTIVE_KEY,
        "versions": {DEFAULT_ACTIVE_KEY: DEFAULT_MODEL_FILE},
    }


@st.cache_resource(show_spinner=False)
def load_model_and_meta() -> Tuple[object, Path, dict]:
    """Load model from config and try to read sidecar metadata."""
    cfg = _read_config()
    active_key = cfg.get("active_model", DEFAULT_ACTIVE_KEY)
    rel = cfg.get("versions", {}).get(active_key, DEFAULT_MODEL_FILE)
    model_path = MODELS_DIR / rel

    # Load model
    model = joblib.load(model_path)

    # Optional metadata next to model (e.g., model_version3beatport.meta.json)
    meta_path = model_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    # Always include a few useful fields
    meta.setdefault("dataset", "unknown")
    meta.setdefault("dataset_version", "unknown")
    meta.setdefault("num_classes", int(getattr(model, "n_classes_", 0) or len(getattr(model, "classes_", [])) or 0))
    meta.setdefault("model_file", str(model_path.name))
    meta.setdefault("active_key", active_key)

    return model, model_path, meta


@st.cache_resource(show_spinner=False)
def load_label_map(model_path: Path) -> Dict[int, str]:
    """
    Try multiple ways to get index→name mapping:
      1) Saved LabelEncoder next to the model (e.g., label_encoder_version3beatport.joblib)
      2) A JSON file mapping {"0": "tech_house", ...} at artifacts/beatport/genre_id_to_name.json
      3) If model.classes_ are strings, just return identity map by index
    """
    # 1) Try a LabelEncoder
    le_candidates = [
        model_path.with_name(model_path.name.replace("model_", "label_encoder_")),
        MODELS_DIR / "label_encoder_version3beatport.joblib",
        MODELS_DIR / "label_encoder.joblib",
    ]
    for p in le_candidates:
        if p.exists():
            try:
                le = joblib.load(p)
                # If le.classes_ are names, map idx -> name
                return {i: str(name) for i, name in enumerate(le.classes_)}
            except Exception:
                pass

    # 2) Try a JSON map (string keys)
    if DEFAULT_JSON_MAP.exists():
        try:
            d = json.loads(DEFAULT_JSON_MAP.read_text())
            # ensure int keys
            return {int(k): str(v) for k, v in d.items()}
        except Exception:
            pass

    # 3) If model has string classes_, map by index. We can't access model here, so do it outside when possible.
    return {}


def to_genre_name(idx: int, label_map: Dict[int, str], model_classes=None) -> str:
    """Convert numeric class index to a readable name."""
    # Prefer explicit label_map
    if label_map and idx in label_map:
        return label_map[idx]

    # If model.classes_ exist and are strings, use them
    if model_classes is not None and len(model_classes) > 0 and isinstance(model_classes[0], str):
        try:
            return str(model_classes[idx])
        except Exception:
            pass

    # Fallback: raw index
    return str(idx)


def format_conf(p: float) -> str:
    return f"{p:.3f}"


# -----------------------------
# Audio → Features
# NOTE: We don't know your exact training features. This extractor creates
# a reasonable fixed-size vector, then pads/truncates to the model's expected
# n_features_in_ so predictions will run.
# -----------------------------
def compute_features(
    audio: np.ndarray,
    sr: int,
    target_dim: Optional[int],
) -> np.ndarray:
    """
    Produce a feature vector and pad/trim to target_dim if provided.
    """
    # Safeguards
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Short signals: pad
    if len(audio) < sr:
        pad = sr - len(audio)
        audio = np.pad(audio, (0, pad))

    # Compute a mix of common descriptors
    feats: List[float] = []

    # MFCC (20) mean + std
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())

    # Chroma (12) mean + std
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(np.mean(chroma, axis=1).tolist())
    feats.extend(np.std(chroma, axis=1).tolist())

    # Spectral contrast (7) mean + std
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feats.extend(np.mean(contrast, axis=1).tolist())
    feats.extend(np.std(contrast, axis=1).tolist())

    # Tonnetz (6) mean + std
    y_harm = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    feats.extend(np.mean(tonnetz, axis=1).tolist())
    feats.extend(np.std(tonnetz, axis=1).tolist())

    # Zero crossing rate mean + std
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))

    # Tempo (1)
    try:
        tempo = librosa.beat.tempo(y=audio, sr=sr, aggregate=None)
        feats.append(float(np.median(tempo)))
    except Exception:
        feats.append(0.0)

    vec = np.array(feats, dtype=np.float32)

    # Conform to model's expected dimension if we know it
    if target_dim is not None and target_dim > 0:
        if vec.shape[0] < target_dim:
            vec = np.pad(vec, (0, target_dim - vec.shape[0]))
        elif vec.shape[0] > target_dim:
            vec = vec[:target_dim]

    return vec


def read_audio_from_upload(file) -> Tuple[np.ndarray, int]:
    """Read audio bytes from an uploaded file into mono float32 array + sr."""
    raw = file.read()
    try:
        # soundfile can decode a bunch of formats
        data, sr = sf.read(io.BytesIO(raw), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)
    except Exception:
        # fallback: librosa
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


# -----------------------------
# Load model, label map
# -----------------------------
with st.spinner("Loading model..."):
    model, model_path, meta = load_model_and_meta()

# Try to load a label map
label_map = load_label_map(model_path)

# Model classes (may be ints or strings)
model_classes = getattr(model, "classes_", None)

# For padding features to the right length
n_features_in = getattr(model, "n_features_in_", None)
if isinstance(n_features_in, (list, tuple)):
    n_features_in = int(n_features_in[0])
elif isinstance(n_features_in, np.ndarray):
    n_features_in = int(n_features_in.item()) if n_features_in.size == 1 else int(n_features_in[0])
elif n_features_in is not None:
    n_features_in = int(n_features_in)


# -----------------------------
# Sidebar: "Option A" verification info
# -----------------------------
st.sidebar.header("Model status ✅")
st.sidebar.success(f"Active model: **{meta.get('active_key', 'unknown')}**")
st.sidebar.write(f"File: `{meta.get('model_file', 'unknown')}`")
try:
    mtime = time.ctime((model_path).stat().st_mtime)
    st.sidebar.write(f"Modified: {mtime}")
except Exception:
    pass

st.sidebar.write("---")
st.sidebar.subheader("Metadata")
st.sidebar.write(f"Dataset: **{meta.get('dataset', 'unknown')}**")
st.sidebar.write(f"Dataset version: **{meta.get('dataset_version', 'unknown')}**")
st.sidebar.write(f"Num classes: **{meta.get('num_classes', 0)}**")

if model_classes is not None:
    # Show a glimpse of classes
    try:
        if len(model_classes) > 0:
            preview = list(model_classes[:10])
            st.sidebar.caption("classes_ (first 10):")
            st.sidebar.code(repr(preview))
    except Exception:
        pass

# Label map information
if label_map:
    st.sidebar.info(f"Label map loaded: {len(label_map)} entries")
    # Show a small sample
    kv = list(label_map.items())[:10]
    st.sidebar.caption("label_map (first 10):")
    st.sidebar.code(repr(kv))
else:
    st.sidebar.warning(
        "No label map found. Predictions will show numeric class indices. "
        "To display names, save a LabelEncoder next to the model or add "
        "`artifacts/beatport/genre_id_to_name.json`."
    )


# -----------------------------
# Main UI: upload → batch predict
# -----------------------------
files = st.file_uploader(
    "Drag and drop files here",
    type=["mp3", "wav", "flac", "ogg", "m4a"],
    accept_multiple_files=True,
)

if not files:
    st.stop()

rows = []
log_items = []

for idx, file in enumerate(files):
    try:
        y, sr = read_audio_from_upload(file)
        vec = compute_features(y, sr, target_dim=n_features_in)
        X = vec.reshape(1, -1)

        # Predict
        pred_idx = model.predict(X)[0]
        # Handle numpy types
        try:
            pred_idx_int = int(pred_idx)
        except Exception:
            # if model already returns string label
            pred_idx_int = None

        # Confidence (top-1 proba if available)
        conf = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                if pred_idx_int is not None and 0 <= pred_idx_int < len(proba):
                    conf = float(proba[pred_idx_int])
                else:
                    conf = float(np.max(proba))
            except Exception:
                conf = None

        # Human-readable name
        pred_name = None
        if pred_idx_int is not None:
            pred_name = to_genre_name(pred_idx_int, label_map, model_classes=model_classes)
        else:
            # if model returns string labels directly
            pred_name = str(pred_idx)

        rows.append(
            {
                "file": file.name,
                "predicted_genre": pred_name,
                "confidence": conf if conf is not None else np.nan,
            }
        )

        # Live line items
        conf_txt = f"(conf {conf:.2f})" if conf is not None else ""
        st.success(f"{file.name} → {pred_name} {conf_txt}")

    except Exception as e:
        st.error(f"{file.name}: error during prediction: {e}")
        rows.append({"file": file.name, "predicted_genre": "ERROR", "confidence": np.nan})

st.markdown("## Results")
df = pd.DataFrame(rows)
# Order columns
df = df[["file", "predicted_genre", "confidence"]]
st.dataframe(df, use_container_width=True)
