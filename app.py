# app.py — MilkCrate (audio/video → genre folders → ZIP) | Beatport-92 full + robust name parsing
# Build: 2025-08-18 • beatport92-full-v2

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
st.set_page_config(page_title="MilkCrate • Audio → Genre ZIP", layout="wide")

# ---------- Config ----------
DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
EXPECTED_FEATURES_JSON = "artifacts/beatport201611_feature_columns.json"

TARGET_SR_DEFAULT = 22050
MAX_ANALYZE_SECONDS_DEFAULT = 120
TOP_K_DEFAULT = 5

SUPPORTED_AUDIO = {"wav","mp3","flac","ogg","oga","opus","m4a","aac","wma","aiff","aif","aifc"}
SUPPORTED_VIDEO = {"mp4","m4v","mov","webm","mkv"}

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

_model_feature_names: List[str] = []

def load_expected_features() -> List[str]:
    if _model_feature_names:
        return list(_model_feature_names)
    if os.path.exists(EXPECTED_FEATURES_JSON):
        try:
            with open(EXPECTED_FEATURES_JSON, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols
        except Exception:
            pass
    st.error("No expected feature list found on model or JSON; cannot build Beatport-92 features.")
    st.stop()

# ---------- Utils ----------
def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "audio")
    base = unicodedata.normalize("NFKD", base).encode("ascii","ignore").decode("ascii")
    base = re.sub(r"[^\w\-.]+","_", base).strip("._")
    return base or "audio"

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

# ---------- Framewise primitives ----------
def stft_mag(y, n_fft=2048, hop=512):
    return np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop))

def series_zcr(y, hop=512):
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

def series_rms(y, hop=512, n_fft=2048):
    return librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

def series_centroid(S, sr):
    return librosa.feature.spectral_centroid(S=S, sr=sr)[0]

def series_bandwidth(S, sr):
    return librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

def series_rolloff(S, sr, roll=0.85):
    return librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll)[0]

def series_spec_entropy(S, eps=1e-10):
    P = S / (S.sum(axis=0, keepdims=True) + eps)
    return (-np.sum(P * np.log(P + eps), axis=0))

def series_spec_flux(S):
    Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-10)
    d = np.diff(Sn, axis=1)
    flux = np.sqrt((np.clip(d, 0, None)**2).sum(axis=0))
    return np.concatenate([[np.nan], flux])

def series_energy_entropy(y, frame_len=2048, hop=512, subframes=10, eps=1e-12):
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T
    ent = []
    seg_len = frame_len // subframes
    for fr in frames:
        segs = [fr[i*seg_len:(i+1)*seg_len] for i in range(subframes)]
        energies = np.array([np.sum(s**2) for s in segs], dtype=float)
        p = energies / (energies.sum() + eps)
        ent.append(-np.sum(p * np.log(p + eps)))
    return np.array(ent)

def safe_stat(series: np.ndarray, which: str) -> float:
    s = np.asarray(series, dtype=float)
    if which in ("m", "mean"):
        return float(np.nanmean(s)) if s.size else np.nan
    else:
        return float(np.nanstd(s)) if s.size else np.nan

# ---------- Beatport-92 row builder (robust name parsing) ----------
def build_beatport92_row(y: np.ndarray, sr: int, expected_cols: List[str]) -> Dict[str, float]:
    """
    Handles:
      - zcr, energy, energyentropy, spectral{centroid,spread,entropy,flux,rolloff} + {m|s}
      - MFCC:      mfccs{N}{m|s|std}
      - ΔMFCC:     dmfccs{N}{…} / deltamfccs{N}{…} / mfccs{N}d{…}
      - Chroma:    chromas{N}{…} / chroma{N}{…} / chromagram{N}{…}
    Any unknown key -> NaN (imputer can handle a few).
    """
    row: Dict[str, float] = {}

    # Precompute primitives
    hop = 512
    n_fft = 2048
    S = stft_mag(y, n_fft=n_fft, hop=hop)
    zcr = series_zcr(y, hop=hop)
    rms = series_rms(y, hop=hop, n_fft=n_fft)
    sc = series_centroid(S, sr)
    sbw = series_bandwidth(S, sr)
    sro = series_rolloff(S, sr, 0.85)
    sent = series_spec_entropy(S)
    sflux = series_spec_flux(S)
    eent = series_energy_entropy(y, frame_len=n_fft, hop=hop, subframes=10)

    # MFCC + ΔMFCC (20 by default; training often used 13, that's fine)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)   # (20, T)
    dmfcc = librosa.feature.delta(mfcc, order=1)         # (20, T)

    # Chroma (12, T)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    for key in expected_cols:
        # strip numeric prefix "NN-"
        base = re.sub(r"^\d+-", "", key)

        # detect mean/std suffix
        stat_suffix = None
        if base.endswith(("std", "STD")):
            stat_suffix = "std"; base = base[:-3]
        elif base.endswith("m"):
            stat_suffix = "m"; base = base[:-1]
        elif base.endswith("s"):
            stat_suffix = "s"; base = base[:-1]

        val = np.nan
        b = base.lower().replace("_","").replace("-","")

        # core groups
        if b == "zcr" and stat_suffix:
            val = safe_stat(zcr, "m" if stat_suffix=="m" else "s")
        elif b == "energy" and stat_suffix:
            val = safe_stat(rms, "m" if stat_suffix=="m" else "s")
        elif b in ("energyentropy","energyent") and stat_suffix:
            val = safe_stat(eent, "m" if stat_suffix=="m" else "s")
        elif b in ("spectralcentroid","speccentroid") and stat_suffix:
            val = safe_stat(sc, "m" if stat_suffix=="m" else "s")
        elif b in ("spectralspread","specspread") and stat_suffix:
            val = safe_stat(sbw, "m" if stat_suffix=="m" else "s")
        elif b in ("spectralentropy","specentropy") and stat_suffix:
            val = safe_stat(sent, "m" if stat_suffix=="m" else "s")
        elif b in ("spectralflux","specflux") and stat_suffix:
            val = safe_stat(sflux, "m" if stat_suffix=="m" else "s")
        elif b in ("spectralrolloff","specrolloff") and stat_suffix:
            val = safe_stat(sro, "m" if stat_suffix=="m" else "s")

        else:
            # MFCC variants
            m_dm1 = re.match(r"(?:dmfccs|deltamfccs)(\d+)$", b)   # dmfccs1, deltamfccs1
            m_dm2 = re.match(r"mfccs(\d+)d$", b)                  # mfccs1d
            m_mf  = re.match(r"mfccs(\d+)$", b)                   # mfccs1

            # Chroma variants
            m_ch  = re.match(r"(?:chromas|chroma|chromagram|chromagrams)(\d+)$", b)

            if (m_dm1 or m_dm2) and stat_suffix:  # ΔMFCC
                idx = int((m_dm1 or m_dm2).group(1)) - 1
                if 0 <= idx < dmfcc.shape[0]:
                    val = safe_stat(dmfcc[idx], "m" if stat_suffix=="m" else "s")

            elif m_mf and stat_suffix:  # MFCC
                idx = int(m_mf.group(1)) - 1
                if 0 <= idx < mfcc.shape[0]:
                    val = safe_stat(mfcc[idx], "m" if stat_suffix=="m" else "s")

            elif m_ch and stat_suffix:  # Chroma
                idx = int(m_ch.group(1)) - 1
                if 0 <= idx < chroma.shape[0]:
                    val = safe_stat(chroma[idx], "m" if stat_suffix=="m" else "s")

        row[key] = float(val) if np.isfinite(val) else np.nan

    return row

# ---------- Decode audio or extract from video ----------
def load_audio_any(raw: bytes, ext: str, target_sr: int, mono: bool = True, max_secs: int = 120) -> Tuple[np.ndarray, int]:
    # 1) soundfile
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

    # 2) librosa/audioread via temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
            tmp.write(raw); tmp.flush()
            y, sr = librosa.load(tmp.name, sr=target_sr, mono=mono, duration=max_secs)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), sr
    except Exception:
        pass

    # 3) video via moviepy
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

# ---------- Prediction helpers ----------
def predict_dataframe(model, encoder, X: pd.DataFrame, top_k: int = 5):
    expected = load_expected_features()
    X = X.reindex(columns=expected)
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
st.caption("Build 2025-08-18 • beatport92-full-v2 (ΔMFCC+Chroma parsing)")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr    = st.selectbox("Target sample rate", [22050, 44100], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, TOP_K_DEFAULT, 1)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, MAX_ANALYZE_SECONDS_DEFAULT, 10)

model = load_model(model_path)
encoder = load_encoder(encoder_path)
_model_feature_names = list(getattr(model, "feature_names_in_", []))  # capture for load_expected_features()

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
    expected_cols = load_expected_features()
    items = []
    rows = []
    progress = st.progress(0)

    for i, f in enumerate(uploaded, start=1):
        raw = f.read()
        name = f.name
        ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            y, sr = load_audio_any(raw, ext=ext, target_sr=int(target_sr), mono=True, max_secs=int(max_secs))
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

    # ---- Diagnostics: coverage per group ----
    def present_count(regex: str) -> int:
        pat = re.compile(regex)
        cols = [c for c in expected_cols if pat.search(re.sub(r'^\d+-','', c))]
        return int(np.sum([df[c].notna().any() for c in cols])) if cols else 0

    present_any = int(np.sum([df[c].notna().any() for c in expected_cols]))
    with st.expander("🧪 Diagnostics: feature alignment"):
        st.write(f"Expected features: {len(expected_cols)}")
        st.write(f"Present (non-NaN) in DF: {present_any}")
        st.write(f"Missing keys: 0")
        st.progress(present_any / max(1, len(expected_cols)))
        st.write({
            "core (zcr/energy/entropy/spec*)": present_count(r'^(zcr|energy(entropy)?|spectral(centroid|spread|entropy|flux|rolloff))(m|s|std)?$'),
            "MFCC":   present_count(r'^mfccs\d+(m|s|std)?$'),
            "ΔMFCC":  present_count(r'^(?:dmfccs|deltamfccs|mfccs\d+d)(m|s|std)?$'),
            "Chroma": present_count(r'^(?:chromas|chroma|chromagram|chromagrams)\d+(m|s|std)?$'),
        })

    try:
        preds = predict_dataframe(model, encoder, df, top_k=int(top_k))
        preds.insert(0, "file_name", file_names)
        st.markdown("**Predictions**")
        st.dataframe(preds, use_container_width=True)

        package_rows = []
        for (orig_name, raw, _), (_, prow) in zip(items, preds.iterrows()):
            package_rows.append((str(prow["pred_label"]), orig_name, raw))

        cols = ["file_name", "pred_label", "pred_idx"]
        if "top_labels" in preds.columns:
            cols += ["top_labels", "top_probs"]
        zip_bytes = build_zip_by_genre(package_rows, preds[cols])

        st.download_button(
            "⬇️ Download ZIP (organized by predicted genre)",
            data=zip_bytes,
            file_name="milkcrate_genres.zip",
            mime="application/zip",
            use_container_width=True
        )

        if preds["pred_label"].nunique(dropna=False) == 1 and present_any < len(expected_cols):
            st.warning("All predictions are the same and many features are NaN. If this persists, share the Diagnostics numbers so we can refine parsing further.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
else:
    st.info("Upload audio (mp3/wav/…) or video (mp4/mov/…); originals will be organized into genre folders and bundled as a ZIP.")
