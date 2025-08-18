# app.py — MilkCrate (audio/video files → genre folders → ZIP)
# Build: 2025-08-18-audio-only
# - Accepts audio (wav, mp3, flac, ogg, opus, m4a, aac, wma, aiff, aif, aifc)
# - Accepts video containers (mp4, m4v, mov, webm, mkv) and extracts audio
# - Extracts features with librosa; predicts with sklearn model + LabelEncoder
# - Groups ORIGINAL uploads into folders by predicted genre; lets you download a ZIP

import os, io, re, zipfile, unicodedata, warnings, tempfile
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# audio/decoding
import soundfile as sf          # libsndfile decoder (wav/flac/ogg/aiff)
import librosa                  # decoding + features (uses audioread/ffmpeg)
import audioread                # backend for mp3/m4a/etc
from moviepy.editor import AudioFileClip  # fallback for video containers

warnings.filterwarnings("ignore")

# --------- Config (can be changed in the sidebar) ---------
DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"   # ~46MB, committed
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"      # 23-class encoder
TARGET_SR_DEFAULT = 22050
MAX_ANALYZE_SECONDS_DEFAULT = 120
TOP_K_DEFAULT = 5

SUPPORTED_AUDIO = {
    "wav","mp3","flac","ogg","oga","opus","m4a","aac","wma","aiff","aif","aifc"
}
SUPPORTED_VIDEO = {"mp4","m4v","mov","webm","mkv"}

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
    base = os.path.basename(name)
    base = unicodedata.normalize("NFKD", base).encode("ascii","ignore").decode("ascii")
    base = re.sub(r"[^\w\-.]+","_", base).strip("._")
    return base or "audio"

def align_columns_to_model(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            st.warning(f"Missing {len(missing)} expected columns; first few: {missing[:10]}")
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
def extract_features_array(y: np.ndarray, sr: int) -> Dict[str,float]:
    feats: Dict[str,float] = {}
    if y is None or len(y) == 0:
        return feats

    feats["duration_s"] = float(len(y)/sr)

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
        feats[f"chroma_{i+1:02d}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i+1:02d}_std"]  = float(np.std(chroma[i]))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1:02d}_std"]  = float(np.std(mfcc[i]))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(contrast.shape[0]):
            feats[f"contrast_{i+1:02d}_mean"] = float(np.mean(contrast[i]))
            feats[f"contrast_{i+1:02d}_std"]  = float(np.std(contrast[i]))
    except Exception:
        pass

    # Tonnetz
    try:
        y_h = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_h, sr=sr)
        for i in range(tonnetz.shape[0]):
            feats[f"tonnetz_{i+1:02d}_mean"] = float(np.mean(tonnetz[i]))
            feats[f"tonnetz_{i+1:02d}_std"]  = float(np.std(tonnetz[i]))
    except Exception:
        pass

    return feats

def load_audio_any(raw: bytes, ext: str, target_sr: int, mono: bool = True, max_secs: int = 120) -> Tuple[np.ndarray,int]:
    # 1) try libsndfile via soundfile
    try:
        with io.BytesIO(raw) as bio:
            data, sr = sf.read(bio, dtype="float32", always_2d=False)
        if data.ndim == 2 and mono:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if max_secs and len(data) > max_secs*sr:
            data = data[:max_secs*sr]
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 2) try librosa/audioread via temp file (handles mp3/m4a, etc.)
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
            tmp.write(raw); tmp.flush()
            y, sr = librosa.load(tmp.name, sr=target_sr, mono=mono, duration=max_secs)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 3) if video, extract audio with moviepy then decode
    if ext.lower() in {"mp4","m4v","mov","webm","mkv"}:
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

def features_from_bytes(raw: bytes, ext: str, target_sr: int, max_secs: int) -> Dict[str,float]:
    y, sr = load_audio_any(raw, ext=ext, target_sr=target_sr, mono=True, max_secs=max_secs)
    return extract_features_array(y, sr)

# --------- Prediction ---------
def predict_dataframe(model, encoder, X: pd.DataFrame, top_k: int = 5):
    X = align_columns_to_model(X, model)
    y_pred = model.predict(X)
    if np.issubdtype(np.array(y_pred).dtype, np.number):
        labels = encoder.inverse_transform(y_pred.astype(int))
    else:
        labels = y_pred.astype(str)
    out = pd.DataFrame({"pred_idx": y_pred, "pred_label": labels})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes, names = get_display_names(model, encoder)
        order = np.argsort(proba, axis=1)[:, ::-1][:, :min(top_k, proba.shape[1])]
        out["top_labels"] = [[names[i] for i in row] for row in order]
        out["top_probs"]  = [[float(proba[r, i]) for i in row] for r, row in enumerate(order)]
    return out

def build_zip_by_genre(rows: List[Tuple[str,str,bytes]], preds_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for genre, fname, fbytes in rows:
            zf.writestr(f"{sanitize_filename(genre or 'Unknown')}/{sanitize_filename(fname)}", fbytes)
        zf.writestr("predictions.csv", preds_df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# --------- UI ---------
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18-audio-only")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr    = st.number_input("Target sample rate", min_value=8000, max_value=48000, value=TARGET_SR_DEFAULT, step=1000)
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
    items = []      # (name, raw, ext)
    feat_rows = []  # list of dicts
    progress = st.progress(0)
    for i, f in enumerate(uploaded, start=1):
        raw = f.read()
        name = f.name
        ext  = name.split(".")[-1].lower() if "." in name else ""
        try:
            feats = features_from_bytes(raw, ext=ext, target_sr=int(target_sr), max_secs=int(max_secs))
            feats["file_name"] = name
            items.append((name, raw, ext))
            feat_rows.append(feats)
        except Exception as e:
            st.error(f"❌ {name}: {e}")
        progress.progress(i/len(uploaded))
    progress.empty()

    if not feat_rows:
        st.error("No decodable files were uploaded.")
        st.stop()

    df = pd.DataFrame(feat_rows).fillna(np.nan)
    file_names = df.pop("file_name").tolist()

    try:
        preds = predict_dataframe(model, encoder, df, top_k=int(top_k))
        preds.insert(0, "file_name", file_names)
        st.markdown("**Predictions**")
        st.dataframe(preds, use_container_width=True)

        package_rows = []
        for (orig_name, raw, _), (_, prow) in zip(items, preds.iterrows()):
            package_rows.append((str(prow["pred_label"]), orig_name, raw))

        zip_bytes = build_zip_by_genre(
            package_rows,
            preds[["file_name", "pred_label", "pred_idx"] + (["top_labels","top_probs"] if "top_labels" in preds.columns else [])]
        )

        st.download_button(
            "⬇️ Download ZIP (organized by predicted genre)",
            data=zip_bytes,
            file_name="milkcrate_genres.zip",
            mime="application/zip",
            use_container_width=True
        )

        if preds["pred_label"].nunique(dropna=False) == 1:
            st.warning("All predictions in this batch are the same. Check class balance or train/inference feature mismatch.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
else:
    st.info("Upload audio (mp3/wav/…) or video (mp4/mov/…); originals will be organized into genre folders and bundled as a ZIP.")
PY
