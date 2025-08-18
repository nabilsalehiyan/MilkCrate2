# app.py — MilkCrate (audio/video → genre ZIP) with Beatport-92 feature mapping
# This extractor emits columns like: "1-zcrm", "2-energym", "3-energyentropym", "9-mfccs1m", ...
# so your model won't collapse to a single class.

import io, os, re, json, zipfile, unicodedata, warnings, tempfile
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

# ---------- Config ----------
DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
EXPECTED_FEATURES_JSON = "artifacts/beatport201611_feature_columns.json"

TARGET_SR_DEFAULT = 22050
MAX_ANALYZE_SECONDS_DEFAULT = 120
TOP_K_DEFAULT = 5

SUPPORTED_AUDIO = {"wav","mp3","flac","ogg","oga","opus","m4a","aac","wma","aiff","aif","aifc"}
SUPPORTED_VIDEO = {"mp4","m4v","mov","webm","mkv"}

st.set_page_config(page_title="MilkCrate • Audio → Genre ZIP", layout="wide")

# ---------- Loaders ----------
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

def load_expected_features() -> List[str]:
    # Prefer model.feature_names_in_; fall back to JSON list (exact order matters)
    names = list(getattr(model, "feature_names_in_", []))
    if names:
        return list(names)
    if os.path.exists(EXPECTED_FEATURES_JSON):
        try:
            with open(EXPECTED_FEATURES_JSON, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols
        except Exception:
            pass
    return []

# ---------- Utils ----------
def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "audio")
    base = unicodedata.normalize("NFKD", base).encode("ascii","ignore").decode("ascii")
    base = re.sub(r"[^\w\-.]+","_", base).strip("._")
    return base or "audio"

def align_columns_to_expected(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(f"Input missing {len(missing)} of {len(expected)} expected columns. First few: {missing[:10]}")
    return df.reindex(columns=expected)

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

# ---------- Low-level feature helpers (framewise) ----------
def frame_rms(y, hop=512, n_fft=2048):
    return librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

def frame_zcr(y, hop=512):
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

def stft_mag(y, n_fft=2048, hop=512):
    return np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop))

def spectral_centroid_series(S, sr):
    return librosa.feature.spectral_centroid(S=S, sr=sr)[0]

def spectral_bandwidth_series(S, sr):
    return librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

def spectral_rolloff_series(S, sr, roll=0.85):
    return librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll)[0]

def spectral_entropy_series(S, eps=1e-10):
    # entropy per frame over frequency bins
    P = S / (S.sum(axis=0, keepdims=True) + eps)
    ent = -np.sum(P * np.log(P + eps), axis=0)  # nats
    return ent

def spectral_flux_series(S):
    # positive flux between consecutive frames, L2-normed
    Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-10)
    d = np.diff(Sn, axis=1)
    flux = np.sqrt((np.clip(d, 0, None)**2).sum(axis=0))
    # align length with other framewise series by padding one NaN at start
    return np.concatenate([[np.nan], flux])

def energy_entropy_series(y, frame_len=2048, hop=512, subframes=10, eps=1e-12):
    # For each analysis frame, split it into 'subframes' segments and compute entropy of sub-energies
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T  # shape: (n_frames, frame_len)
    ent = []
    seg_len = frame_len // subframes
    for fr in frames:
        # split into equal segments
        segs = [fr[i*seg_len:(i+1)*seg_len] for i in range(subframes)]
        energies = np.array([np.sum(s**2) for s in segs], dtype=float)
        p = energies / (energies.sum() + eps)
        ent.append(-np.sum(p * np.log(p + eps)))
    return np.array(ent)

# ---------- Beatport-92 row builder ----------
def build_beatport92_row(y: np.ndarray, sr: int, expected_cols: List[str]) -> Dict[str, float]:
    """
    Populate EXACT expected keys. We parse names like '1-zcrm', '2-energym', '9-mfccs1m', '33-mfccs1s', '57-chromas1m', etc.
    If a key is unknown, we set NaN so the model's imputer can handle it.
    """
    row: Dict[str, float] = {}

    # Precompute primitives
    hop = 512
    n_fft = 2048
    S = stft_mag(y, n_fft=n_fft, hop=hop)
    zcr = frame_zcr(y, hop=hop)
    rms = frame_rms(y, hop=hop, n_fft=n_fft)
    sc = spectral_centroid_series(S, sr)
    sbw = spectral_bandwidth_series(S, sr)
    sro = spectral_rolloff_series(S, sr, 0.85)
    sent = spectral_entropy_series(S)
    sflux = spectral_flux_series(S)

    # MFCCs (20)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Chroma STFT (12)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    # Energy entropy (framewise)
    eent = energy_entropy_series(y, frame_len=n_fft, hop=hop, subframes=10)

    # Helper to fetch mean/std from a series, robust to NaNs
    def stat(series, which: str):
        s = np.asarray(series, dtype=float)
        if which == "m":
            return float(np.nanmean(s))
        return float(np.nanstd(s))

    # Map keys
    for key in expected_cols:
        # Strip the numeric prefix "NN-"
        name = re.sub(r"^\d+-", "", key)

        # Mean/Std suffix handling (final char m/s)
        m_or_s = None
        if name.endswith("m"):
            m_or_s = "m"; base = name[:-1]
        elif name.endswith("s"):
            m_or_s = "s"; base = name[:-1]
        else:
            base = name

        val = np.nan  # default if unknown

        # --- Single-feature bases ---
        if base == "zcr" and m_or_s:
            val = stat(zcr, m_or_s)
        elif base == "energy" and m_or_s:
            val = stat(rms, m_or_s)  # using RMS as energy proxy
        elif base == "energyentropy" and m_or_s:
            val = stat(eent, m_or_s)
        elif base == "spectralcentroid" and m_or_s:
            val = stat(sc, m_or_s)
        elif base == "spectralspread" and m_or_s:
            val = stat(sbw, m_or_s)
        elif base == "spectralentropy" and m_or_s:
            val = stat(sent, m_or_s)
        elif base == "spectralflux" and m_or_s:
            val = stat(sflux, m_or_s)
        elif base == "spectralrolloff" and m_or_s:
            val = stat(sro, m_or_s)

        # --- MFCCs: mfccs{1..20}{m|s} ---
        elif base.startswith("mfccs") and m_or_s:
            m = re.match(r"mfccs(\d+)$", base)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < mfcc.shape[0]:
                    val = stat(mfcc[idx], m_or_s)

        # --- Chroma: chromas{1..12}{m|s} ---
        elif base.startswith("chromas") and m_or_s:
            m = re.match(r"chromas(\d+)$", base)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < chroma.shape[0]:
                    val = stat(chroma[idx], m_or_s)

        row[key] = float(val) if np.isfinite(val) else np.nan

    return row

# ---------- Decode audio or extract from video ----------
def load_audio_any(raw: bytes, ext: str, target_sr: int, mono: bool = True, max_secs: int = 120) -> Tuple[np.ndarray, int]:
    # 1) soundfile (wav/flac/ogg/aiff)
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

    # 2) librosa/audioread via temp file (mp3/m4a/etc.)
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
            tmp.write(raw); tmp.flush()
            y, sr = librosa.load(tmp.name, sr=target_sr, mono=mono, duration=max_secs)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 3) video containers via moviepy
    if ext.lower() in SUPPORTED_VIDEO:
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

# ---------- Prediction ----------
def predict_dataframe(model, encoder, X: pd.DataFrame, top_k: int = 5):
    expected = load_expected_features()
    X = align_columns_to_expected(X, expected) if expected else X
    y_pred = model.predict(X)
    labels = encoder.inverse_transform(y_pred.astype(int)) if np.issubdtype(np.array(y_pred).dtype, np.number) else y_pred.astype(str)
    out = pd.DataFrame({"pred_idx": y_pred, "pred_label": labels})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes, names = get_display_names(model, encoder)
        order = np.argsort(proba, axis=1)[:, ::-1][:, :min(top_k, proba.shape[1])]
        out["top_labels"] = [[names[i] for i in row] for row in order]
        out["top_probs"]  = [[float(proba[r, i]) for i in row] for r, row in enumerate(order)]
    return out

def build_zip_by_genre(rows, preds_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for genre, fname, fbytes in rows:
            zf.writestr(f"{sanitize_filename(genre or 'Unknown')}/{sanitize_filename(fname)}", fbytes)
        zf.writestr("predictions.csv", preds_df.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ---------- UI ----------
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18 • Beatport-92 feature mapping")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr    = st.selectbox("Target sample rate", [22050, 44100], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, TOP_K_DEFAULT, 1)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, MAX_ANALYZE_SECONDS_DEFAULT, 10)

model = load_model(model_path)
encoder = load_encoder(encoder_path)
expected_cols = load_expected_features()

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
    rows = []       # feature dicts
    progress = st.progress(0)
    for i, f in enumerate(uploaded, start=1):
        raw = f.read()
        name = f.name
        ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            y, sr = load_audio_any(raw, ext=ext, target_sr=int(target_sr), mono=True, max_secs=int(max_secs))
            # Build a row EXACTLY for expected columns (92 keys). Unknown keys -> NaN
            if not expected_cols:
                st.error("No expected feature list found on model or JSON; cannot build Beatport-92 features.")
                st.stop()
            row = build_beatport92_row(y, sr, expected_cols)
            row["file_name"] = name
            items.append((name, raw, ext))
            rows.append(row)
        except Exception as e:
            st.error(f"❌ {name}: {e}")
        progress.progress(i / len(uploaded))
    progress.empty()

    if not rows:
        st.error("No decodable files were uploaded.")
        st.stop()

    df = pd.DataFrame(rows).fillna(np.nan)
    file_names = df.pop("file_name").tolist()

    # Diagnostics
    with st.expander("🧪 Diagnostics: feature alignment"):
        if expected_cols:
            present = [c for c in expected_cols if c in df.columns and df[c].notna().any()]
            missing = [c for c in expected_cols if c not in df.columns]  # should be 0 now
            coverage = len(present) / max(1, len(expected_cols))
            st.write(f"Expected features: {len(expected_cols)}")
            st.write(f"Present (non-NaN) in DF: {len(present)}")
            st.write(f"Missing keys: {len(missing)}")
            st.progress(coverage)
            if missing:
                st.code("\n".join(missing[:50]))

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
            st.warning("All predictions are the same. If coverage is high, this may be class imbalance in training.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
else:
    st.info("Upload audio (mp3/wav/…) or video (mp4/mov/…); originals will be organized into genre folders and bundled as a ZIP.")
