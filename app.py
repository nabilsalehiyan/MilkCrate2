# app.py — MilkCrate (audio/video → genre folders → ZIP)
# Build: 2025-08-18 • MFCC13 + chromavector + chromadeviation + core • RNG pickle shim v2

import io, os, re, json, zipfile, unicodedata, warnings, tempfile
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# decoding / features
import soundfile as sf              # wav/flac/ogg/aiff
import librosa                      # decoding + features
import audioread                    # mp3/m4a/etc backend
from moviepy.editor import AudioFileClip  # video → audio

warnings.filterwarnings("ignore")
st.set_page_config(page_title="MilkCrate • Audio → Genre ZIP", layout="wide")

# ---------- Config ----------
DEFAULT_MODEL_PATH = "artifacts/model_rf.joblib"
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
EXPECTED_FEATURES_JSON = "artifacts/beatport201611_feature_columns.json"  # optional fallback

TARGET_SR_DEFAULT = 22050
MAX_ANALYZE_SECONDS_DEFAULT = 120
TOP_K_DEFAULT = 5

SUPPORTED_AUDIO = {"wav","mp3","flac","ogg","oga","opus","m4a","aac","wma","aiff","aif","aifc"}
SUPPORTED_VIDEO = {"mp4","m4v","mov","webm","mkv"}

# ---------- NumPy RNG pickle compat (strong shim) ----------
def _numpy_rng_pickle_shim():
    """
    Some artifacts were saved when NumPy referenced private RNG modules
    like numpy.random._pcg64. Newer builds may not recognize those paths
    and raise: "<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module".
    This shim:
      1) creates alias modules for old paths, and
      2) patches numpy.random._pickle.__bit_generator_ctor to map legacy classes.
    """
    import sys, types, numpy as _np
    npr = _np.random

    # 1) Provide legacy private modules with current classes
    mapping = {
        "numpy.random._pcg64":   ("PCG64", "BitGenerator"),
        "numpy.random._mt19937": ("MT19937", "BitGenerator"),
        "numpy.random._philox":  ("Philox", "BitGenerator"),
        "numpy.random._sfc64":   ("SFC64", "BitGenerator"),
    }
    for mod, names in mapping.items():
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            for name in names:
                if hasattr(npr, name):
                    setattr(m, name, getattr(npr, name))
            sys.modules[mod] = m

    # 2) Patch constructor used during unpickling to accept legacy module names
    try:
        import numpy.random._pickle as nrp  # type: ignore[attr-defined]
        orig_ctor = getattr(nrp, "__bit_generator_ctor", None)
        if callable(orig_ctor):
            def _compat_ctor(bitgen_cls):
                # bitgen_cls can be a class; str(bitgen_cls) → "<class '...PCG64'>"
                name = getattr(bitgen_cls, "__name__", None) or str(bitgen_cls)
                if "PCG64" in name:   return npr.PCG64
                if "MT19937" in name: return npr.MT19937
                if "Philox" in name or "PHILOX" in name: return npr.Philox
                if "SFC64" in name:   return npr.SFC64
                # Fallback to NumPy's original behavior
                return orig_ctor(bitgen_cls)
            nrp.__bit_generator_ctor = _compat_ctor  # type: ignore[assignment]
    except Exception:
        # If patching fails, we still have the alias modules above.
        pass

# ---------- Loaders ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    _numpy_rng_pickle_shim()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found: {path}")
        st.stop()
    return joblib.load(path)

_model_feature_names: List[str] = []

def load_expected_features() -> List[str]:
    """
    Inference column order: prefer model.feature_names_in_,
    else fall back to artifacts JSON (if present).
    """
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
    st.error("No expected feature list found on model or JSON; cannot build features.")
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

def split_base_and_stat(raw_name: str):
    """
    Returns (base_without_stat, stat) where stat in {'m','s'} or None.
    Accepts suffixes: m/s/std/mean/avg/sd/stdev/variance/var.
    'variance'/'var' map to std ('s').
    """
    s = raw_name.lower().replace("_","").replace("-","")
    for suf in ("mean", "avg", "m"):
        if s.endswith(suf):
            return s[: -len(suf)], "m"
    for suf in ("std", "sd", "stdev", "s", "variance", "var"):
        if s.endswith(suf):
            return s[: -len(suf)], "s"
    return s, None

# ---------- Row builder matching your model's names ----------
def build_feature_row(y: np.ndarray, sr: int, expected_cols: List[str]) -> Dict[str, float]:
    """
    Fills keys exactly as your model expects:
      - Core: zcr, energy, energyentropy, spectral{centroid,spread,entropy,flux,rolloff} + mean/std
      - MFCC: mfccs{1..20}{m|std} (artifact uses 1..13)
      - Chroma: chromavector{1..12}{m|std}, chromadeviation{m|std}
    """
    row: Dict[str, float] = {}

    # Precompute primitives
    hop = 512
    n_fft = 2048
    S = stft_mag(y, n_fft=n_fft, hop=hop)
    zcr = series_zcr(y, hop=hop)
    rms = series_rms(y, hop=hop, n_fft=n_fft)
    sc  = series_centroid(S, sr)
    sbw = series_bandwidth(S, sr)
    sro = series_rolloff(S, sr, 0.85)
    sent = series_spec_entropy(S)
    sflux = series_spec_flux(S)
    eent = series_energy_entropy(y, frame_len=n_fft, hop=hop, subframes=10)

    # MFCCs (compute 20; we'll use indices present in expected names)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)   # (20, T)

    # Chroma vector (12) + deviation across bins per frame
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)     # (12, T)
    chroma_dev_per_frame = np.std(chroma, axis=0)        # (T,)

    for key in expected_cols:
        # strip numeric prefix 'NN-'
        name = re.sub(r"^\d+-", "", key)
        base, which = split_base_and_stat(name)  # which is 'm' or 's'
        out = np.nan
        b = base  # normalized string

        # ----- core groups -----
        if b == "zcr" and which:                          out = safe_stat(zcr, which)
        elif b == "energy" and which:                     out = safe_stat(rms, which)
        elif b in ("energyentropy","energyent") and which: out = safe_stat(eent, which)
        elif b in ("spectralcentroid","speccentroid") and which: out = safe_stat(sc, which)
        elif b in ("spectralspread","specspread") and which:     out = safe_stat(sbw, which)
        elif b in ("spectralentropy","specentropy") and which:   out = safe_stat(sent, which)
        elif b in ("spectralflux","specflux") and which:         out = safe_stat(sflux, which)
        elif b in ("spectralrolloff","specrolloff") and which:   out = safe_stat(sro, which)

        else:
            s = b
            # MFCC index: mfccs{N}
            m_mf = re.match(r"mfccs(\d+)$", s)
            # Chroma vector: chromavector{N}
            m_cv = re.match(r"chromavector(\d+)$", s)
            # Chroma deviation: chromadeviation
            is_cdev = (s == "chromadeviation")

            if m_mf and which:
                i = int(m_mf.group(1)) - 1
                if 0 <= i < mfcc.shape[0]:
                    out = safe_stat(mfcc[i], which)

            elif m_cv and which:
                j = int(m_cv.group(1)) - 1
                if 0 <= j < chroma.shape[0]:
                    out = safe_stat(chroma[j], which)

            elif is_cdev and which:
                out = safe_stat(chroma_dev_per_frame, which)

        row[key] = float(out) if np.isfinite(out) else np.nan

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
st.caption("Build 2025-08-18 • MFCC13 + chromavector + chromadeviation + core • RNG pickle shim v2")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr    = st.selectbox("Target sample rate", [22050, 44100], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, TOP_K_DEFAULT, 1)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, MAX_ANALYZE_SECONDS_DEFAULT, 10)

model = load_model(model_path)
encoder = load_encoder(encoder_path)
# Capture model feature names for expected-order alignment
_model_feature_names = list(getattr(model, "feature_names_in_", []))

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
            row = build_feature_row(y, sr, expected_cols)
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
            "core (zcr/energy/entropy/spec*)": present_count(r'^(zcr|energy(entropy)?|spectral(centroid|spread|entropy|flux|rolloff))(m|s|std|mean|sd|stdev|variance|var)?$'),
            "MFCC":   present_count(r'^mfccs\d+(m|s|std|mean|sd|stdev|variance|var)?$'),
            "Chroma": present_count(r'^(chromavector\d+|chromadeviation)(m|s|std|mean|sd|stdev|variance|var)?$'),
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
            st.warning("All predictions are the same and many features are NaN. Share the Diagnostics numbers if this persists.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
else:
    st.info("Upload audio (mp3/wav/…) or video (mp4/mov/…); originals will be organized into genre folders and bundled as a ZIP.")
