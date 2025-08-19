# app.py — MilkCrate: drop audio/video → genre-organized ZIP
# Build 2025-08-18 • MFCC13 + ΔMFCC13 + chromavector + core • soft RNG shim on-demand

from __future__ import annotations

import io
import os
import re
import sys
import zipfile
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Audio stack
import librosa
import soundfile as sf  # noqa: F401 (librosa may use it)

# ----------------
# Defaults / UI
# ----------------
DEFAULT_MODEL_PATH   = "artifacts/model_version1beatport.joblib"   # small, portable
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
DEFAULT_SR           = 22050
DEFAULT_TOPK         = 5
DEFAULT_MAX_SECS     = 120

st.set_page_config(page_title="MilkCrate — Drop audio/video → genre-organized ZIP", layout="wide")

# ------------------------------------------------------
# Soft NumPy RNG pickle shim (safe; no class replacement)
# ------------------------------------------------------
def _numpy_rng_pickle_shim_soft():
    """
    Best-effort compatibility for legacy NumPy RNG pickles:
    - Provide private legacy modules (e.g., numpy.random._pcg64) pointing at current classes
    - Patch numpy.random._pickle.__bit_generator_ctor
    - Loosen BitGenerator.__setstate__ to accept tuple payloads
    This avoids the C-level size mismatch that happens if you *replace* BitGenerator.
    """
    import sys as _sys, types as _types, numpy as _np
    npr = _np.random

    # 1) Map private legacy modules to current public classes
    mapping = {
        "numpy.random._pcg64":   ("PCG64", "BitGenerator"),
        "numpy.random._mt19937": ("MT19937", "BitGenerator"),
        "numpy.random._philox":  ("Philox", "BitGenerator"),
        "numpy.random._sfc64":   ("SFC64", "BitGenerator"),
    }
    for mod_name, names in mapping.items():
        if mod_name not in _sys.modules:
            m = _types.ModuleType(mod_name)
            for nm in names:
                if hasattr(npr, nm):
                    setattr(m, nm, getattr(npr, nm))
            _sys.modules[mod_name] = m

    # 2) Patch ctor hook used during unpickling
    try:
        import numpy.random._pickle as nrp  # type: ignore[attr-defined]
        orig_ctor = getattr(nrp, "__bit_generator_ctor", None)
        if callable(orig_ctor):
            def _compat_ctor(bitgen_cls):
                nm = getattr(bitgen_cls, "__name__", "") or str(bitgen_cls)
                if "PCG64" in nm:   return npr.PCG64
                if "MT19937" in nm: return npr.MT19937
                if "Philox" in nm:  return npr.Philox
                if "SFC64" in nm:   return npr.SFC64
                return orig_ctor(bitgen_cls)
            nrp.__bit_generator_ctor = _compat_ctor  # type: ignore[assignment]
    except Exception:
        pass

    # 3) Relax BitGenerator.__setstate__ to accept tuple payloads
    try:
        BG = _np.random.bit_generator.BitGenerator  # Cython class
        _orig_setstate = BG.__setstate__
        def _compat_setstate(self, state):
            if isinstance(state, tuple):
                cand = None
                if len(state) == 2 and isinstance(state[1], dict):
                    cand = state[1]
                elif len(state) == 1 and isinstance(state[0], dict):
                    cand = state[0]
                if cand is None:
                    cand = {"state": state}
                try:
                    return _orig_setstate(self, cand)
                except Exception:
                    # fall back to a benign fresh seed; inference doesn't use RNG history
                    try:
                        self.state = _np.random.PCG64(42).state  # type: ignore[attr-defined]
                        return
                    except Exception:
                        pass
            return _orig_setstate(self, state)
        BG.__setstate__ = _compat_setstate  # type: ignore[assignment]
    except Exception:
        pass


# -------------
# Cached I/O
# -------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    # Try without any shim first (works for the v1 RF model)
    try:
        return joblib.load(path)
    except (TypeError, ValueError) as e:
        msg = f"{e}"
        if "BitGenerator" in msg or "bit_generator" in msg or "random._pickle" in msg:
            # Apply soft shim and retry
            _numpy_rng_pickle_shim_soft()
            return joblib.load(path)
        raise

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found: {path}")
        st.stop()
    return joblib.load(path)

# -----------------------
# Audio/video decoding
# -----------------------
def _tmp_from_uploader(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    fd, path = tempfile.mkstemp(prefix="milkcrate_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def load_audio_any(path: str, target_sr: int, max_secs: int) -> Tuple[np.ndarray, int]:
    """
    Load mono audio at target_sr from audio or video containers.
    First try librosa; if that fails, fall back to MoviePy/ffmpeg.
    """
    # librosa path
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_secs)
        return y, sr
    except Exception:
        pass

    # moviepy fallback (optional)
    try:
        from moviepy.editor import AudioFileClip  # lazy import
        clip = AudioFileClip(path)
        dur = min(max_secs, int(clip.duration)) if clip.duration else max_secs
        arr = clip.to_soundarray(fps=target_sr)
        clip.close()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        y = arr.astype(np.float32)
        if dur and len(y) > dur * target_sr:
            y = y[:dur * target_sr]
        return y, target_sr
    except Exception as e:
        raise RuntimeError(f"Could not decode audio: {e}")

# -------------------------
# Feature extraction
# -------------------------
def _safe_mean_std(X: np.ndarray) -> Tuple[float, float]:
    if X.size == 0 or np.all(~np.isfinite(X)):
        return float("nan"), float("nan")
    return float(np.nanmean(X)), float(np.nanstd(X))

def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Superset: core + MFCC13 + ΔMFCC13 + chroma(12) + chroma deviation,
    with mean/std for each. Names match model columns after stripping numeric prefixes.
    """
    n_fft = 2048
    hop = 512
    feats: Dict[str, float] = {}

    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) + 1e-12

    # Core features
    zcr   = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).squeeze()
    rms   = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    sc    = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sbw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sroll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()

    # spectral entropy per frame
    P = (S / S.sum(axis=0, keepdims=True)).clip(min=1e-12)
    sent = (-P * np.log2(P)).sum(axis=0)

    # energy entropy over frames
    Er = (rms ** 2).astype(np.float64)
    Er /= (Er.sum() + 1e-12)
    eent = -np.where(Er > 0, Er * np.log2(Er), 0.0)

    # spectral flux
    dS = np.diff(S, axis=1)
    sflux = np.sqrt((dS * dS).mean(axis=0))
    sflux = np.pad(sflux, (1, 0), mode="constant")

    for name, arr in [
        ("zcrm", zcr),
        ("energym", rms),
        ("spectralcentroidm", sc),
        ("spectralspreadm", sbw),
        ("spectralrolloffm", sroll),
    ]:
        m, s = _safe_mean_std(arr)
        feats[name] = m
        feats[name.replace("m", "std", 1) if name.endswith("m") else f"{name}std"] = s

    m, s = _safe_mean_std(eent);  feats["energyentropym"]    = m; feats["energyentropystd"] = s
    m, s = _safe_mean_std(sent);  feats["spectralentropym"]  = m; feats["spectralentropystd"] = s
    m, s = _safe_mean_std(sflux); feats["spectralfluxm"]     = m; feats["spectralfluxstd"] = s

    # MFCC 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    for i in range(13):
        m, s = _safe_mean_std(mfcc[i])
        feats[f"mfccs{i+1}m"]  = m
        feats[f"mfccs{i+1}std"] = s

    # ΔMFCC 13
    dmfcc = librosa.feature.delta(mfcc, order=1)
    for i in range(13):
        m, s = _safe_mean_std(dmfcc[i])
        feats[f"amfccs{i+1}m"]  = m
        feats[f"amfccs{i+1}std"] = s

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i in range(12):
        m, s = _safe_mean_std(chroma[i])
        feats[f"chromavector{i+1}m"]  = m
        feats[f"chromavector{i+1}std"] = s
    ch_dev = chroma.std(axis=0)
    m, s = _safe_mean_std(ch_dev)
    feats["chromadeviationm"]   = m
    feats["chromadeviationstd"] = s

    return feats

# -------------------------
# Alignment & prediction
# -------------------------
_PREFIX_RE = re.compile(r"^\d+-")
def _strip_prefix(c: str) -> str: return _PREFIX_RE.sub("", c)

def align_features_for_model(feat_dict: Dict[str, float], model_cols: List[str]) -> pd.DataFrame:
    row = {}
    for col in model_cols:
        key = _strip_prefix(col)
        row[col] = feat_dict.get(key, np.nan)
    return pd.DataFrame([row], columns=model_cols)

def predict_one(path: str, model, encoder, sr: int, max_secs: int, model_cols: List[str], top_k: int):
    y, _ = load_audio_any(path, sr, max_secs)
    feats = extract_features(y, sr)
    X = align_features_for_model(feats, model_cols)
    pred_idx = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        k = min(top_k, len(proba))
        idxs = np.argsort(proba)[::-1][:k]
        top_probs = [float(proba[i]) for i in idxs]
        human = encoder.inverse_transform(np.asarray(idxs, int))
        top_labels = [str(x) for x in human]
    else:
        top_labels = [str(encoder.inverse_transform([pred_idx])[0])]
        top_probs  = [1.0]

    pred_label = str(encoder.inverse_transform([pred_idx])[0])
    return os.path.basename(path), pred_idx, pred_label, top_labels, top_probs, feats

def make_zip(rows: List[Tuple[str, str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, basename, data in rows:
            safe = re.sub(r"[^\w\-\s.]", "_", label)
            zf.writestr(f"{safe}/{basename}", data)
    mem.seek(0)
    return mem.read()

# -----------
# UI
# -----------
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18 • MFCC13 + ΔMFCC13 + chromavector + core • soft RNG shim on-demand")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path",   value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    tgt_sr       = st.selectbox("Target sample rate", options=[22050, 32000, 44100, 48000], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, DEFAULT_TOPK)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, DEFAULT_MAX_SECS)

# Load resources
model   = load_model(model_path)
encoder = load_encoder(encoder_path)
model_cols: List[str] = list(getattr(model, "feature_names_in_", []))

# Upload & predict
st.subheader("Upload audio/video files")
uploaded = st.file_uploader(
    "Drag and drop files here",
    type=None,
    accept_multiple_files=True,
    help="Any common audio/video: mp3, wav, aiff, m4a, mp4, mov, mkv, ogg, opus, flac, webm, etc."
)

if uploaded:
    tmp_paths: List[str] = []
    file_bytes: Dict[str, bytes] = {}
    for up in uploaded:
        p = _tmp_from_uploader(up)
        tmp_paths.append(p)
        file_bytes[p] = up.getvalue()

    rows = []
    zip_rows = []
    diag_groups = {"core": 0, "MFCC": 0, "ΔMFCC": 0, "Chroma": 0}

    with st.spinner("Analyzing…"):
        for p in tmp_paths:
            try:
                base, pred_idx, pred_label, top_labels, top_probs, feats = predict_one(
                    p, model, encoder, int(tgt_sr), int(max_secs), model_cols, int(top_k)
                )
                rows.append({
                    "file_name": base,
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                    "top_labels": ", ".join(top_labels),
                    "top_probs": ", ".join(f"{x:.6f}" for x in top_probs),
                })
                zip_rows.append((pred_label, base, file_bytes[p]))
                # crude feature group counts
                for k in feats:
                    lk = k.lower()
                    if lk.startswith(("zcr","energy","spectral")): diag_groups["core"] += 1
                    elif lk.startswith("mfccs"): diag_groups["MFCC"] += 1
                    elif lk.startswith("amfccs"): diag_groups["ΔMFCC"] += 1
                    elif lk.startswith(("chroma","chromavector","chromadev")): diag_groups["Chroma"] += 1
            except Exception as e:
                rows.append({
                    "file_name": os.path.basename(p),
                    "pred_idx": -1,
                    "pred_label": f"ERROR: {e}",
                    "top_labels": "",
                    "top_probs": "",
                })

    with st.expander("📝 Diagnostics: feature alignment", expanded=True):
        if model_cols:
            st.write(f"Expected features: {len(model_cols)}")
            # quick coverage check on last file
            y, _ = load_audio_any(tmp_paths[-1], int(tgt_sr), int(max_secs))
            last = extract_features(y, int(tgt_sr))
            present = sum(np.isfinite(last.get(re.sub(r'^\\d+-','', c), np.nan)) for c in model_cols)
            st.write(f"Present (non-NaN) in DF: {present}")
            st.write("By group (non-NaN counts):")
            st.json(diag_groups)
        else:
            st.write("Model did not expose feature_names_in_; alignment assumed.")

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Build ZIP (exclude rows with errors)
    ok_rows = [(r["pred_label"], r["file_name"], file_bytes[p])
               for r, p in zip(rows, tmp_paths) if not r["pred_label"].startswith("ERROR:")]
    if ok_rows:
        zbytes = make_zip(ok_rows)
        st.download_button(
            "⬇️ Download ZIP (organized by predicted genre)",
            data=zbytes,
            file_name="milkcrate_by_genre.zip",
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("Upload one or more audio/video files to get predictions and a genre-organized ZIP.")
