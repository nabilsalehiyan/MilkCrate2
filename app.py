# app.py — MilkCrate: drop audio/video → genre-organized ZIP
# Build 2025-08-18 • MFCC13 + ΔMFCC13 + chromavector + core • RNG pickle shim v3

from __future__ import annotations

import io
import os
import re
import sys
import json
import math
import time
import shutil
import zipfile
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Librosa stack for audio features
import librosa
import soundfile as sf


# =========================
# Defaults & UI parameters
# =========================
DEFAULT_MODEL_PATH   = "artifacts/model_version1beatport.joblib"   # small, portable
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
DEFAULT_SR           = 22050
DEFAULT_TOPK         = 5
DEFAULT_MAX_SECS     = 120

AUDIO_EXTS = {
    ".wav",".mp3",".mp4",".m4a",".aac",".aif",".aiff",".aifc",
    ".flac",".m4v",".mov",".webm",".wma",".ogg",".oga",".opus",
    ".mkv",".mpg",".mpeg",".aac",".caf",".mpga",".weba",".wav"
}

st.set_page_config(page_title="MilkCrate — Drop audio/video → genre-organized ZIP", layout="wide")

# ===========================================
# NumPy RNG pickle hardening (v3: robust)
# ===========================================
def _rng_pickle_shim_v3():
    """
    Shadow NumPy's Cython BitGenerator with a Python stub during unpickle, and
    expose shim classes for legacy private modules (PCG64, MT19937, Philox, SFC64).
    This avoids crashes like:
      TypeError: descriptor '__setstate__' for '...BitGenerator' doesn't apply to a 'tuple'
    """
    import sys as _sys, types as _types, numpy as _np
    npr = _np.random

    class _ShimBitGen:
        def __init__(self, *a, **k): self.state = {}
        def __setstate__(self, state):
            # Accept older tuple payloads:
            if isinstance(state, tuple):
                cand = None
                if len(state) == 2 and isinstance(state[1], dict):
                    cand = state[1]
                elif len(state) == 1 and isinstance(state[0], dict):
                    cand = state[0]
                if cand is None:
                    cand = {"state": state}
                self.state = cand
                return
            # dict-like
            self.state = state
        def __getstate__(self): return self.state

    class _ShimPCG64(_ShimBitGen): pass
    class _ShimMT19937(_ShimBitGen): pass
    class _ShimPhilox(_ShimBitGen): pass
    class _ShimSFC64(_ShimBitGen): pass

    # 1) Shadow the Cython BitGenerator
    try:
        import numpy.random.bit_generator as bg
        bg.BitGenerator = _ShimBitGen  # type: ignore
    except Exception:
        pass

    # 2) Provide private legacy modules with shim classes
    for suffix, cls in [("_pcg64", _ShimPCG64),
                        ("_mt19937", _ShimMT19937),
                        ("_philox",  _ShimPhilox),
                        ("_sfc64",   _ShimSFC64)]:
        fullname = f"numpy.random.{suffix}"
        mod = _sys.modules.get(fullname)
        if mod is None:
            mod = _types.ModuleType(fullname)
            _sys.modules[fullname] = mod
        setattr(mod, cls.__name__[6:], cls)  # expose 'PCG64', etc.

    # 3) Patch numpy’s unpickle ctor hook (best-effort)
    try:
        import numpy.random._pickle as nrp  # type: ignore
        def _ctor(bitgen_cls):
            name = getattr(bitgen_cls, "__name__", str(bitgen_cls))
            return {
                "PCG64":  _ShimPCG64,
                "MT19937":_ShimMT19937,
                "Philox": _ShimPhilox,
                "SFC64":  _ShimSFC64,
            }.get(name, _ShimBitGen)
        nrp.__bit_generator_ctor = _ctor  # type: ignore
    except Exception:
        pass


# ==================
# Cached loaders
# ==================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    _rng_pickle_shim_v3()  # <-- critical: call before joblib.load
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found: {path}")
        st.stop()
    return joblib.load(path)


# =================================
# Robust audio/video file loading
# =================================
def _tmp_from_uploader(uploaded) -> str:
    """Write an uploaded file to a real temp path and return it."""
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    fd, path = tempfile.mkstemp(prefix="milkcrate_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def load_audio_any(path_or_bytes: str | bytes, target_sr: int, max_secs: int) -> Tuple[np.ndarray, int]:
    """
    Load audio mono at target_sr from an audio or video file.
    Tries librosa/soundfile first. If that fails (e.g., some video containers),
    falls back to MoviePy (ffmpeg).
    """
    # a) if bytes, write to temp
    cleanup = None
    if isinstance(path_or_bytes, bytes):
        fd, tmp = tempfile.mkstemp(prefix="milkcrate_buf_", suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(path_or_bytes)
        path = tmp
        cleanup = path
    else:
        path = path_or_bytes

    y, sr = None, None
    # First attempt: librosa
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_secs)
    except Exception:
        # Fallback: MoviePy (handles many containers)
        try:
            from moviepy.editor import AudioFileClip  # lazy import
            clip = AudioFileClip(path)
            # Duration cap
            dur = min(max_secs, int(clip.duration)) if clip.duration else max_secs
            # Pull audio as mono
            arr = clip.to_soundarray(fps=target_sr)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            y = arr.astype(np.float32)
            sr = target_sr
            if dur and len(y) > dur * sr:
                y = y[:dur * sr]
            clip.close()
        except Exception as e:
            if cleanup and os.path.exists(cleanup):
                os.remove(cleanup)
            raise e

    if cleanup and os.path.exists(cleanup):
        os.remove(cleanup)

    if y is None or sr is None:
        raise RuntimeError("Could not decode audio from file.")
    return y, sr


# ===============================
# Feature extraction utilities
# ===============================
def _safe_mean_std(X: np.ndarray) -> Tuple[float,float]:
    if X.size == 0 or np.all(~np.isfinite(X)):
        return float("nan"), float("nan")
    return float(np.nanmean(X)), float(np.nanstd(X))

def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute a superset of features to cover multiple model variants.
    Names intentionally exclude numeric prefixes; we align later.

    Groups:
      - core: zcr, energy (rms), energyentropy, spectralcentroid, spectralspread(bandwidth),
              spectralentropy, spectralflux, spectralrolloff    (means + stds)
      - MFCC 13: means + stds
      - ΔMFCC 13: means + stds
      - Chroma (12): means + stds, plus chromadeviation mean/std
    """
    n_fft = 2048
    hop = 512

    feats = {}

    # STFT magnitude
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) + 1e-12

    # Core time/freq features
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).squeeze()
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sroll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    # spectral entropy per frame
    P = (S / S.sum(axis=0, keepdims=True)).clip(min=1e-12)
    sent = (-P * np.log2(P)).sum(axis=0)
    # energy entropy per frame (Shannon of RMS distribution across frames)
    Er = (rms ** 2).astype(np.float64)
    Er /= (Er.sum() + 1e-12)
    eent = -np.where(Er > 0, Er * np.log2(Er), 0.0)

    # spectral flux (frame-to-frame change)
    dS = np.diff(S, axis=1)
    sflux = np.sqrt((dS * dS).mean(axis=0))
    sflux = np.pad(sflux, (1, 0), mode="constant")  # align length

    for name, arr in [
        ("zcrm", zcr),
        ("energym", rms),                 # NOTE: 'm' suffix retained in name for alignment mapping
        ("spectralcentroidm", sc),
        ("spectralspreadm", sbw),
        ("spectralrolloffm", sroll),
    ]:
        m, s = _safe_mean_std(arr)
        feats[name] = m
        feats[name.replace("m", "std", 1) if name.endswith("m") else f"{name}std"] = s

    m, s = _safe_mean_std(eent)
    feats["energyentropym"] = m
    feats["energyentropystd"] = s

    m, s = _safe_mean_std(sent)
    feats["spectralentropym"] = m
    feats["spectralentropystd"] = s

    m, s = _safe_mean_std(sflux)
    feats["spectralfluxm"] = m
    feats["spectralfluxstd"] = s

    # MFCCs (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    for i in range(13):
        m, s = _safe_mean_std(mfcc[i])
        feats[f"mfccs{i+1}m"] = m
        feats[f"mfccs{i+1}std"] = s

    # ΔMFCCs (first order)
    dmfcc = librosa.feature.delta(mfcc, order=1)
    for i in range(13):
        m, s = _safe_mean_std(dmfcc[i])
        feats[f"amfccs{i+1}m"] = m
        feats[f"amfccs{i+1}std"] = s

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i in range(12):
        m, s = _safe_mean_std(chroma[i])
        feats[f"chromavector{i+1}m"] = m
        feats[f"chromavector{i+1}std"] = s

    # Chroma deviation (per-frame spread across 12 bins)
    ch_dev = chroma.std(axis=0)
    m, s = _safe_mean_std(ch_dev)
    feats["chromadeviationm"] = m
    feats["chromadeviationstd"] = s

    return feats


# ==========================
# Alignment & prediction
# ==========================
_PREFIX_RE = re.compile(r"^\d+-")

def _strip_numeric_prefix(col: str) -> str:
    """Remove leading 'NN-' numeric prefix if present."""
    return _PREFIX_RE.sub("", col)

def align_features_for_model(feat_dict: Dict[str, float], model_cols: List[str]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with columns ordered as model expects.
    We match by stripping numeric prefixes from model columns.
    Unknown columns become NaN (the model's pipeline imputer can handle them).
    """
    aligned = {}
    for col in model_cols:
        key = _strip_numeric_prefix(col)
        aligned[col] = feat_dict.get(key, np.nan)
    return pd.DataFrame([aligned], columns=model_cols)


def predict_one_path(
    path: str, model, encoder, target_sr: int, max_secs: int, model_cols: List[str], top_k: int
) -> Tuple[str, int, str, List[str], List[float], Dict[str, float]]:
    """
    Returns: (basename, pred_idx, pred_label, top_labels, top_probs, feat_dict_used)
    """
    y, sr = load_audio_any(path, target_sr, max_secs)
    feats = extract_features(y, sr)
    X = align_features_for_model(feats, model_cols)

    # prediction
    pred_idx = int(model.predict(X)[0])

    # top-k
    top_labels, top_probs = [], []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        top_k = min(top_k, len(proba))
        idxs = np.argsort(proba)[::-1][:top_k]
        top_probs = [float(proba[i]) for i in idxs]
        # classes_ are numeric codes; inverse_transform to names
        human = encoder.inverse_transform(np.asarray(idxs, dtype=int))
        top_labels = [str(x) for x in human]
    else:
        top_labels = [str(encoder.inverse_transform([pred_idx])[0])]
        top_probs = [1.0]

    pred_label = str(encoder.inverse_transform([pred_idx])[0])
    return os.path.basename(path), pred_idx, pred_label, top_labels, top_probs, feats


def make_zip_by_genre(rows: List[Tuple[str, str, bytes]]) -> bytes:
    """
    rows: list of (pred_label, original_basename, file_bytes)
    returns: zip bytes with folder per predicted label
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for label, basename, data in rows:
            safe_label = re.sub(r"[^\w\-\s.]", "_", label)
            arcname = f"{safe_label}/{basename}"
            zf.writestr(arcname, data)
    mem.seek(0)
    return mem.read()


# =========
# UI
# =========
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18 • MFCC13 + ΔMFCC13 + chromavector + chromadeviation + core • RNG pickle shim v3")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    tgt_sr       = st.selectbox("Target sample rate", options=[22050, 32000, 44100, 48000], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, DEFAULT_TOPK)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, DEFAULT_MAX_SECS)

# Load model/encoder
model   = load_model(model_path)
encoder = load_encoder(encoder_path)

# Capture model feature names for expected-order alignment
_model_feature_names: List[str] = list(getattr(model, "feature_names_in_", []))

# Upload area
st.subheader("Upload audio/video files")
uploaded = st.file_uploader(
    "Drag and drop files here",
    type=None,  # accept anything; we will try to decode audio track
    accept_multiple_files=True,
    help="Any common audio/video file is supported (mp3, wav, aiff, m4a, mp4, mov, mkv, ogg, opus, flac, etc.)."
)

if uploaded:
    work_dir = tempfile.mkdtemp(prefix="milkcrate_batch_")
    file_paths: List[str] = []
    file_bytes: Dict[str, bytes] = {}

    # Persist uploads
    for up in uploaded:
        p = _tmp_from_uploader(up)
        file_paths.append(p)
        file_bytes[p] = up.getvalue()

    # Predict
    records = []
    zip_rows = []
    diag_counts = {"core": 0, "MFCC": 0, "ΔMFCC": 0, "Chroma": 0}
    ok = 0

    with st.spinner("Analyzing files…"):
        for p in file_paths:
            try:
                base, pred_idx, pred_label, top_labels, top_probs, feats = predict_one_path(
                    p, model, encoder, tgt_sr, int(max_secs), _model_feature_names, int(top_k)
                )
                ok += 1
                records.append({
                    "file_name": base,
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                    "top_labels": ", ".join(top_labels),
                    "top_probs": ", ".join(f"{x:.6f}" for x in top_probs),
                })
                zip_rows.append((pred_label, base, file_bytes[p]))

                # crude group diagnostics
                for k in feats:
                    lk = k.lower()
                    if lk.startswith(("zcr","energy","spectral")): diag_counts["core"] += 1
                    elif lk.startswith("mfccs"): diag_counts["MFCC"] += 1
                    elif lk.startswith("amfccs"): diag_counts["ΔMFCC"] += 1
                    elif lk.startswith(("chroma","chromavector","chromadev")): diag_counts["Chroma"] += 1
            except Exception as e:
                records.append({
                    "file_name": os.path.basename(p),
                    "pred_idx": -1,
                    "pred_label": f"ERROR: {e}",
                    "top_labels": "",
                    "top_probs": "",
                })

    # Diagnostics: feature alignment
    with st.expander("📝 Diagnostics: feature alignment", expanded=True):
        st.write(f"Expected features: {_model_feature_names and len(_model_feature_names) or 'unknown'}")
        if _model_feature_names:
            # compute coverage on the last file’s feats (approximate overview)
            last_feats = extract_features(*load_audio_any(file_paths[-1], tgt_sr, int(max_secs)))
            present = sum(
                np.isfinite(last_feats.get(_strip_numeric_prefix(c), np.nan)) for c in _model_feature_names
            )
            st.write(f"Present (non-NaN) in DF: {present}")
            missing_keys = [
                _strip_numeric_prefix(c)
                for c in _model_feature_names
                if not np.isfinite(last_feats.get(_strip_numeric_prefix(c), np.nan))
            ]
            st.write(f"Missing keys: {len(missing_keys)}")
            if len(missing_keys) > 0:
                st.code("\n".join(missing_keys[:40]), language="text")
            st.write("By group (non-NaN counts):")
            st.json(diag_counts)

    # Results table
    df = pd.DataFrame.from_records(records)
    st.subheader("Predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download ZIP
    zip_bytes = make_zip_by_genre([(label, name, data) for (label, name, data) in
                                   [(r["pred_label"], r["file_name"], file_bytes[p]) for r, p in zip(records, file_paths)]
                                   if not r["pred_label"].startswith("ERROR:")])

    st.download_button(
        "⬇️ Download ZIP (organized by predicted genre)",
        data=zip_bytes,
        file_name="milkcrate_by_genre.zip",
        mime="application/zip",
        use_container_width=True,
        disabled=(ok == 0),
    )

else:
    st.info("Upload one or more audio/video files to get predictions, organized by genre in a downloadable ZIP.")
