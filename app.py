# app.py — MilkCrate (audio/video → genre folders → ZIP) + Diagnostics
# Build: 2025-08-18-audio-only + feature-alignment
# - Accepts audio (wav, mp3, flac, ogg, opus, m4a, aac, wma, aiff, aif, aifc)
# - Accepts video (mp4, m4v, mov, webm, mkv) and extracts audio
# - Extracts features with librosa, predicts with sklearn model + LabelEncoder
# - Groups ORIGINAL uploads into folders named by predicted genre; offers ZIP download
# - Diagnostics panel to check feature coverage vs. model expectations

import io
import os
import re
import json
import zipfile
import unicodedata
import warnings
import tempfile
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# decoding / features
import soundfile as sf              # libsndfile (wav/flac/ogg/aiff)
import librosa                      # decoding + features (uses audioread/ffmpeg)
import audioread                    # backend for mp3/m4a/etc
from moviepy.editor import AudioFileClip  # fallback for video containers

warnings.filterwarnings("ignore")

# --------- Config (changeable in sidebar) ---------
DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"   # ~46MB, committed to repo
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"      # 23-class encoder
EXPECTED_FEATURES_JSON = "artifacts/beatport201611_feature_columns.json"

TARGET_SR_DEFAULT = 22050
MAX_ANALYZE_SECONDS_DEFAULT = 120
TOP_K_DEFAULT = 5

SUPPORTED_AUDIO = {
    "wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "aac", "wma", "aiff", "aif", "aifc"
}
SUPPORTED_VIDEO = {"mp4", "m4v", "mov", "webm", "mkv"}

st.set_page_config(page_title="MilkCrate • Audio → Genre ZIP", layout="wide")

# --------- Cache loaders ---------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found: {path}")
        st.stop()
    return joblib.load(path)

# --------- Utils ---------
def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "audio")
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^\w\-.]+", "_", base).strip("._")
    return base or "audio"

def load_expected_features_fallback() -> List[str]:
    try:
        if os.path.exists(EXPECTED_FEATURES_JSON):
            with open(EXPECTED_FEATURES_JSON, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols
    except Exception:
        pass
    return []

def pad2(n: int) -> str:
    return f"{n:02d}"

def add_aliases_to_match_expected(feats: Dict[str, float], expected: List[str]) -> Dict[str, float]:
    """
    Add alternate keys for common naming conventions (only if the alias is in expected).
    E.g., mfcc_1_mean -> mfcc_01_mean, spec_centroid_mean -> spectral_centroid_mean, etc.
    """
    if not expected:
        return feats
    out = dict(feats)

    expected_set = set(expected)

    # Basic synonyms
    synonyms = {
        "zcr_mean": "zero_crossing_rate_mean",
        "zcr_std": "zero_crossing_rate_std",
        "spec_centroid_mean": "spectral_centroid_mean",
        "spec_centroid_std": "spectral_centroid_std",
        "spec_bw_mean": "spectral_bandwidth_mean",
        "spec_bw_std": "spectral_bandwidth_std",
        "spec_rolloff_mean": "spectral_rolloff_mean",
        "spec_rolloff_std": "spectral_rolloff_std",
        "duration_s": "duration",
        "chroma_mean": "chroma_stft_mean",
        "chroma_std": "chroma_stft_std",
        "tempo_mean": "bpm_mean",
        "tempo_std": "bpm_std",
        "rms_mean": "rms_energy_mean",
        "rms_std": "rms_energy_std",
    }
    for src, alias in synonyms.items():
        if src in feats and alias in expected_set:
            out[alias] = feats[src]

    # MFCC / chroma / contrast / tonnetz: add zero-padded variants
    for i in range(1, 21):  # mfcc 1..20
        k1m = f"mfcc_{i}_mean"; k1s = f"mfcc_{i}_std"
        k2m = f"mfcc_{pad2(i)}_mean"; k2s = f"mfcc_{pad2(i)}_std"
        if k1m in feats and k2m in expected_set: out[k2m] = feats[k1m]
        if k1s in feats and k2s in expected_set: out[k2s] = feats[k1s]
    for i in range(1, 13):  # chroma 1..12
        k1m = f"chroma_{i:01d}_mean"; k1s = f"chroma_{i:01d}_std"
        k2m = f"chroma_{pad2(i)}_mean"; k2s = f"chroma_{pad2(i)}_std"
        if k1m in feats and k2m in expected_set: out[k2m] = feats[k1m]
        if k1s in feats and k2s in expected_set: out[k2s] = feats[k1s]
    for i in range(1, 11):  # contrast 1..10 (safe upper bound)
        k1m = f"contrast_{i}_mean"; k1s = f"contrast_{i}_std"
        k2m = f"contrast_{pad2(i)}_mean"; k2s = f"contrast_{pad2(i)}_std"
        if k1m in feats and k2m in expected_set: out[k2m] = feats[k1m]
        if k1s in feats and k2s in expected_set: out[k2s] = feats[k1s]
    for i in range(1, 7):   # tonnetz 1..6
        k1m = f"tonnetz_{i}_mean"; k1s = f"tonnetz_{i}_std"
        k2m = f"tonnetz_{pad2(i)}_mean"; k2s = f"tonnetz_{pad2(i)}_std"
        if k1m in feats and k2m in expected_set: out[k2m] = feats[k1m]
        if k1s in feats and k2s in expected_set: out[k2s] = feats[k1s]

    return out

def align_columns_to_model(X: pd.DataFrame, model):
    names = list(getattr(model, "feature_names_in_", []))
    if not names:
        names = load_expected_features_fallback()
    if names:
        missing = [c for c in names if c not in X.columns]
        if missing:
            st.warning(f"Input missing {len(missing)} of {len(names)} expected columns. First few: {missing[:10]}")
        X = X.reindex(columns=names)
    return X

def get_display_names(model, encoder):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None, None
    arr = np.array(classes)
    if np.issubdtype(arr.dtype, np.number):
        try:
            names = encoder.inverse_transform(arr.astype(int))
        except Exception:
            names = arr.astype(str)
        return arr, names
    return arr, arr

# --------- Feature extraction ---------
def extract_features_array(y: np.ndarray, sr: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if y is None or len(y) == 0:
        return feats

    feats["duration_s"] = float(len(y) / sr)

    # Time-domain
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    feats["rms_mean"] = float(np.mean(rms)); feats["rms_std"] = float(np.std(rms))
    feats["zcr_mean"] = float(np.mean(zcr)); feats["zcr_std"] = float(np.std(zcr))

    # Spectral
    S = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=512))
    feats["spec_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    feats["spec_centroid_std"]  = float(np.std(librosa.feature.spectral_centroid(S=S, sr=sr)))
    feats["spec_bw_mean"]       = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    feats["spec_bw_std"]        = float(np.std(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    feats["spec_rolloff_mean"]  = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    feats["spec_rolloff_std"]   = float(np.std(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))

    # Tempo
    try:
        tempo_seq = librosa.beat.tempo(y=y, sr=sr, hop_length=512, aggregate=None)
        feats["tempo_mean"] = float(np.mean(tempo_seq)) if tempo_seq.size else np.nan
        feats["tempo_std"]  = float(np.std(tempo_seq)) if tempo_seq.size else np.nan
    except Exception:
        feats["tempo_mean"] = np.nan; feats["tempo_std"] = np.nan

    # Chroma
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    feats["chroma_mean"] = float(np.mean(chroma)); feats["chroma_std"] = float(np.std(chroma))
    for i in range(min(12, chroma.shape[0])):
        feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i+1}_std"]  = float(np.std(chroma[i]))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(contrast.shape[0]):
            feats[f"contrast_{i+1}_mean"] = float(np.mean(contrast[i]))
            feats[f"contrast_{i+1}_std"]  = float(np.std(contrast[i]))
    except Exception:
        pass

    # Tonnetz
    try:
        y_h = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_h, sr=sr)
        for i in range(tonnetz.shape[0]):
            feats[f"tonnetz_{i+1}_mean"] = float(np.mean(tonnetz[i]))
            feats[f"tonnetz_{i+1}_std"]  = float(np.std(tonnetz[i]))
    except Exception:
        pass

    return feats

def load_audio_any(raw: bytes, ext: str, target_sr: int, mono: bool = True, max_secs: int = 120) -> Tuple[np.ndarray, int]:
    # 1) try libsndfile via soundfile (wav/flac/ogg/aiff)
    try:
        with io.BytesIO(raw) as bio:
            data, sr = sf.read(bio, dtype="float32", always_2d=False)
        if data.ndim == 2 and mono:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if max_secs and len(data) > max_secs * sr:
            data = data[: max_secs * sr]
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 2) try librosa/audioread via temp file (mp3/m4a/etc.)
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
            tmp.write(raw); tmp.flush()
            y, sr = librosa.load(tmp.name, sr=target_sr, mono=mono, duration=max_secs)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 3) if video container, extract audio with moviepy then decode
    if ext.lower() in {"mp4", "m4v", "mov", "webm", "mkv"}:
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as vtmp, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wtmp:
                vtmp.write(raw); vtmp.flush()
                clip = AudioFileClip(vtmp.name)
                clip.write_audiofile(wtmp.name, fps=target_sr, nbytes=2, codec="pcm_s16le", logger=None)
                clip.close()
                y, sr = librosa.load(wtmp.name, sr=target_sr, mono=mono, duration=max_secs)
            return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), sr
        except Exception:
            pass

    raise ValueError("Unable to decode file. (For mp4/m4a, ffmpeg support may be required.)")

def features_from_bytes(raw: bytes, ext: str, target_sr: int, max_secs: int, expected_cols: List[str]) -> Dict[str, float]:
    y, sr = load_audio_any(raw, ext=ext, target_sr=target_sr, mono=True, max_secs=max_secs)
    base = extract_features_array(y, sr)
    # Add alias keys only if they help match expected columns
    return add_aliases_to_match_expected(base, expected_cols)

# --------- Prediction ---------
def predict_dataframe(model, encoder, X: pd.DataFrame, top_k: int = 5):
    X = align_columns_to_model(X, model)
    y_pred = model.predict(X)
    # map numeric codes -> human labels
    labels = encoder.inverse_transform(y_pred.astype(int)) if np.issubdtype(np.array(y_pred).dtype, np.number) else y_pred.astype(str)
    out = pd.DataFrame({"pred_idx": y_pred, "pred_label": labels})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes, names = get_display_names(model, encoder)
        order = np.argsort(proba, axis=1)[:, ::-1][:, :min(top_k, proba.shape[1])]
        out["top_labels"] = [[names[i] for i in row] for row in order]
        out["top_probs"]  = [[float(proba[r, i]) for i in row] for r, row in enumerate(order)]
    return out

def build_zip_by_genre(rows: List[Tuple[str, str, bytes]], preds_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for genre, fname, fbytes in rows:
            zf.writestr(f"{sanitize_filename(genre or 'Unknown')}/{sanitize_filename(fname)}", fbytes)
        zf.writestr("predictions.csv", preds_df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# --------- UI ---------
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18 • audio-only + feature-alignment")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr    = st.selectbox("Target sample rate", [22050, 44100], index=0)
    top_k        = st.number_input("Top-K probabilities", min_value=1, max_value=10, value=TOP_K_DEFAULT, step=1)
    max_secs     = st.number_input("Analyze up to (seconds)", min_value=10, max_value=600, value=MAX_ANALYZE_SECONDS_DEFAULT, step=10)

model = load_model(model_path)
encoder = load_encoder(encoder_path)

with st.expander("🔎 Debug: label map"):
    classes, names = get_display_names(model, encoder)
    if classes is not None:
        st.dataframe(pd.DataFrame({"class_code": classes, "label": names}), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Upload files (audio or video)")

uploaded = st.file_uploader(
    "Drop as many files as you want",
    type=sorted(list(SUPPORTED_AUDIO | SUPPORTED_VIDEO)),
    accept_multiple_files=True
)

if uploaded:
    expected_cols = list(getattr(model, "feature_names_in_", [])) or load_expected_features_fallback()

    items = []      # (name, raw, ext)
    feat_rows = []  # dicts
    progress = st.progress(0)
    for i, f in enumerate(uploaded, start=1):
        raw = f.read()
        name = f.name
        ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            feats = features_from_bytes(raw, ext=ext, target_sr=int(target_sr), max_secs=int(max_secs), expected_cols=expected_cols)
            feats["file_name"] = name
            items.append((name, raw, ext))
            feat_rows.append(feats)
        except Exception as e:
            st.error(f"❌ {name}: {e}")
        progress.progress(i / len(uploaded))
    progress.empty()

    if not feat_rows:
        st.error("No decodable files were uploaded.")
        st.stop()

    df = pd.DataFrame(feat_rows).fillna(np.nan)
    file_names = df.pop("file_name").tolist()

    # --- Diagnostics: feature coverage ---
    coverage = None
    with st.expander("🧪 Diagnostics: feature alignment & variability"):
        if expected_cols:
            present = [c for c in expected_cols if c in df.columns]
            missing = [c for c in expected_cols if c not in df.columns]
            coverage = len(present) / len(expected_cols)
            st.write(f"Expected features: {len(expected_cols)}")
            st.write(f"Present in extracted DF: {len(present)}")
            st.write(f"Missing: {len(missing)}")
            st.progress(coverage)
            if missing:
                st.caption("First 50 missing feature names:")
                st.code("\n".join(missing[:50]))
            # Per-file variability (are inputs collapsing?)
            if present:
                show_cols = present[:20]
                st.write("Per-file feature stats (first 20 overlapping columns):")
                try:
                    st.dataframe(df[show_cols].astype(float).describe().loc[["mean","std"]])
                except Exception:
                    st.dataframe(df[show_cols].describe().loc[["mean","std"]])
        else:
            st.info("Model has no feature_names_in_, and no JSON fallback was found.")

    try:
        preds = predict_dataframe(model, encoder, df, top_k=int(top_k))
        preds.insert(0, "file_name", file_names)
        st.markdown("**Predictions**")
        st.dataframe(preds, use_container_width=True)

        package_rows = []
        for (orig_name, raw, _), (_, prow) in zip(items, preds.iterrows()):
            package_rows.append((str(prow["pred_label"]), orig_name, raw))

        cols = ["file_name", "pred_label", "pred_idx"]
        if "top_labels" in preds.columns: cols += ["top_labels", "top_probs"]
        zip_bytes = build_zip_by_genre(package_rows, preds[cols])

        st.download_button(
            "⬇️ Download ZIP (organized by predicted genre)",
            data=zip_bytes,
            file_name="milkcrate_genres.zip",
            mime="application/zip",
            use_container_width=True
        )

        if preds["pred_label"].nunique(dropna=False) == 1:
            if coverage is not None and coverage < 0.7:
                st.warning("All predictions are the same and feature coverage is low. This strongly suggests a train/inference feature mismatch. The Diagnostics panel lists missing columns.")
            else:
                st.warning("All predictions are the same. Check class balance or feature mismatch in Diagnostics.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
else:
    st.info("Upload audio (mp3/wav/…) or video (mp4/mov/…); originals will be organized into genre folders and bundled as a ZIP.")
