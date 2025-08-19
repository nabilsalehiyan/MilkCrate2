# app.py — MilkCrate: drop audio/video → genre-organized ZIP
# Build 2025-08-18
# - RNG-safe unpickler for NumPy BitGenerator pickles
# - 92-col feature set with aliasing
# - Audio/video uploads, batch prediction, ZIP-by-genre export
# - Beats loudness band ratios (24) + BPM/danceability/tempo histogram + beats_loudness.mean/std

from __future__ import annotations

import io
import os
import re
import zipfile
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# audio
import librosa
import soundfile as sf  # noqa: F401

# ================= Defaults =================
DEFAULT_MODEL_PATH   = "artifacts/model_version1beatport.joblib"
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
DEFAULT_SR           = 22050
DEFAULT_TOPK         = 5
DEFAULT_MAX_SECS     = 120

st.set_page_config(page_title="MilkCrate — Drop audio/video → genre-organized ZIP", layout="wide")


# ================= RNG-safe loader =================
def _safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        msg = str(e)
        needs = any(k in msg for k in (
            "BitGenerator", "bit_generator", "numpy.random._pickle",
            "is not a known BitGenerator", "size changed", "__setstate__"
        ))
        if not needs:
            raise

        from joblib.numpy_pickle import NumpyUnpickler

        class _GenStub:
            def __init__(self, *a, **k): self.state = {}
            def __setstate__(self, state):
                if isinstance(state, tuple):
                    if len(state) == 2 and isinstance(state[1], dict): state = state[1]
                    elif len(state) == 1 and isinstance(state[0], dict): state = state[0]
                    else: state = {"state": state}
                self.state = state
            def __getstate__(self): return self.state
            def random(self, *a, **k): return 0.5
            def randint(self, *a, **k): return 0

        def _noop(*_a, **_k): return _GenStub()

        class _ShimUnpickler(NumpyUnpickler):
            def find_class(self, module, name):
                if module == "numpy.random._pickle" and name in {
                    "__generator_ctor", "__randomstate_ctor", "__bit_generator_ctor"
                }:
                    return _noop
                if (module, name) in {
                    ("numpy.random._generator", "Generator"),
                    ("numpy.random.mtrand", "RandomState"),
                    ("numpy.random.bit_generator", "BitGenerator"),
                }:
                    return _GenStub
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _ShimUnpickler(path, f, mmap_mode=None).load()


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    return _safe_joblib_load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found: {path}")
        st.stop()
    return joblib.load(path)


# ================= Audio/video decode =================
def _tmp_from_uploader(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".bin"
    fd, path = tempfile.mkstemp(prefix="milkcrate_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def load_audio_any(path: str, target_sr: int, max_secs: int) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_secs)
        return y, sr
    except Exception:
        pass
    try:
        from moviepy.editor import AudioFileClip
        clip = AudioFileClip(path)
        dur = min(max_secs, int(clip.duration)) if clip.duration else max_secs
        arr = clip.to_soundarray(fps=target_sr)
        clip.close()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        y = arr.astype(np.float32)
        if dur and len(y) > dur * target_sr:
            y = y[: dur * target_sr]
        return y, target_sr
    except Exception as e:
        raise RuntimeError(f"Could not decode audio: {e}")


# ================= Feature helpers =================
def _safe_mean_std(X: np.ndarray) -> Tuple[float, float]:
    if X.size == 0 or np.all(~np.isfinite(X)):
        return float("nan"), float("nan")
    return float(np.nanmean(X)), float(np.nanstd(X))

def _beat_boundaries(y: np.ndarray, sr: int, hop: int, n_frames: int) -> List[Tuple[int, int]]:
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
        beats = np.asarray(beats, int)
        if beats.size >= 2:
            starts = beats
            ends = np.r_[beats[1:], n_frames]
            pairs = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
            if pairs:
                return pairs
    except Exception:
        pass
    # fallback ~0.5s
    step = int(round(0.5 * sr / hop))
    starts = np.arange(0, n_frames, step, dtype=int)
    ends = np.r_[starts[1:], n_frames]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]

def _compute_beats_loudness_band_ratio(y: np.ndarray, sr: int, S_power: np.ndarray, hop: int) -> Dict[str, float]:
    n_fft = 2048
    n_frames = S_power.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyq = sr / 2.0
    f_min = 20.0
    edges = np.geomspace(f_min, nyq, num=13)
    band_bins = []
    for j in range(12):
        lo, hi = edges[j], edges[j+1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2.0))])
        band_bins.append(idx)

    beats = _beat_boundaries(y, sr, hop, n_frames)
    eps = 1e-10
    full_power = S_power.sum(axis=0) + eps

    ratios = [[] for _ in range(12)]
    for s, e in beats:
        s = max(0, min(s, n_frames-1))
        e = max(s+1, min(e, n_frames))
        denom = float(full_power[s:e].mean())
        if not np.isfinite(denom) or denom <= 0:
            continue
        for j in range(12):
            band_pow = S_power[band_bins[j], s:e].mean()
            ratios[j].append(float(band_pow / denom))

    feats = {}
    for j in range(12):
        arr = np.asarray(ratios[j], float)
        if arr.size == 0:
            m, s = float("nan"), float("nan")
        else:
            m, s = float(np.nanmean(arr)), float(np.nanstd(arr))
        feats[f"beats_loudness_band_ratio.mean{j+1}"]  = m
        feats[f"beats_loudness_band_ratio.stdev{j+1}"] = s
    return feats

def _compute_beats_loudness_stats(y: np.ndarray, sr: int, hop: int, n_frames: int) -> Dict[str, float]:
    """Per-beat RMS → mean & std; names match Essentia style."""
    beats = _beat_boundaries(y, sr, hop, n_frames)
    if not beats:
        return {"beats_loudness.mean": float("nan"), "beats_loudness.stdev": float("nan")}
    n_fft = 2048
    # frame-wise RMS to aggregate inside beats
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    vals = []
    for s, e in beats:
        s = max(0, min(s, len(rms)-1))
        e = max(s+1, min(e, len(rms)))
        vals.append(float(np.nanmean(rms[s:e])))
    arr = np.asarray(vals, float)
    m, s = _safe_mean_std(arr)
    return {"beats_loudness.mean": m, "beats_loudness.stdev": s}

def _tempo_histogram_feats(y: np.ndarray, sr: int, hop: int) -> Dict[str, float]:
    """
    Approximations of Essentia-style tempo histogram features using librosa tempogram:
      - bpm (1st peak bpm)
      - bpmconf (normalized height of 1st peak)
      - bpm_histogram_first_peak_bpm / _weight
      - bpm_histogram_second_peak_bpm / _weight / _spread
      - danceability (same as 1st-peak normalized weight)
      - bpmessentia (alias target; we also map it back in alias table)
    """
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if oenv.size == 0:
        return {k: float("nan") for k in [
            "bpm", "bpmconf", "bpm_histogram_first_peak_bpm", "bpm_histogram_first_peak_weight",
            "bpm_histogram_second_peak_bpm", "bpm_histogram_second_peak_weight", "bpm_histogram_second_peak_spread",
            "danceability"
        ]}
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop)  # shape: lags x frames
    tg_mean = tg.mean(axis=1)  # aggregate over time
    tempi = librosa.tempo_frequencies(tg_mean.shape[0], sr=sr, hop_length=hop)  # bpm grid

    # Focus on 60..200 BPM (EDM-ish). If none, broaden.
    mask = (tempi >= 60) & (tempi <= 200)
    if not np.any(mask):
        mask = (tempi >= 30) & (tempi <= 240)
    t_vals = tempi[mask]
    h_vals = tg_mean[mask]
    if h_vals.size == 0 or np.all(~np.isfinite(h_vals)):
        return {k: float("nan") for k in [
            "bpm", "bpmconf", "bpm_histogram_first_peak_bpm", "bpm_histogram_first_peak_weight",
            "bpm_histogram_second_peak_bpm", "bpm_histogram_second_peak_weight", "bpm_histogram_second_peak_spread",
            "danceability"
        ]}

    # Normalize to [0,1]
    h_vals = np.maximum(h_vals, 0)
    if float(h_vals.max()) > 0:
        h_norm = h_vals / float(h_vals.max())
    else:
        h_norm = h_vals

    # Get top-2 peaks (by value)
    order = np.argsort(h_vals)[::-1]
    i1 = int(order[0])
    bpm1 = float(t_vals[i1])
    w1 = float(h_norm[i1])

    i2 = int(order[1]) if order.size > 1 else i1
    bpm2 = float(t_vals[i2])
    w2 = float(h_norm[i2])

    # Local spread around 2nd peak: weighted std in ±3 bins
    lo = max(0, i2 - 3); hi = min(len(t_vals), i2 + 4)
    t_win = t_vals[lo:hi]
    w_win = h_norm[lo:hi]
    if w_win.sum() > 0:
        mu = np.sum(t_win * w_win) / np.sum(w_win)
        spread = float(np.sqrt(np.sum(w_win * (t_win - mu) ** 2) / np.sum(w_win)))
    else:
        spread = float("nan")

    return {
        "bpm": bpm1,
        "bpmconf": w1,
        "bpm_histogram_first_peak_bpm": bpm1,
        "bpm_histogram_first_peak_weight": w1,
        "bpm_histogram_second_peak_bpm": bpm2,
        "bpm_histogram_second_peak_weight": w2,
        "bpm_histogram_second_peak_spread": spread,
        "danceability": w1,  # proxy: strength of dominant periodicity
    }


# ================= Extraction (92 cols) =================
def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    n_fft = 2048
    hop = 512
    feats: Dict[str, float] = {}

    # STFT & power
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) + 1e-12
    S_power = (S * S)

    # --- core basics
    zcr   = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).squeeze()
    rms   = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    sc    = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sbw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sroll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()

    P = (S / S.sum(axis=0, keepdims=True)).clip(min=1e-12)
    sent = (-P * np.log2(P)).sum(axis=0)

    Er = (rms ** 2).astype(np.float64); Er /= (Er.sum() + 1e-12)
    eent = -np.where(Er > 0, Er * np.log2(Er), 0.0)

    dS = np.diff(S, axis=1)
    sflux = np.sqrt((dS * dS).mean(axis=0))
    sflux = np.pad(sflux, (1,0), mode="constant")

    for name, arr in [
        ("zcrm", zcr), ("energym", rms),
        ("spectralcentroidm", sc),
        ("spectralspreadm", sbw),
        ("spectralrolloffm", sroll),
    ]:
        m, s = _safe_mean_std(arr)
        feats[name] = m
        feats[name.replace("m", "std", 1) if name.endswith("m") else f"{name}std"] = s

    m, s = _safe_mean_std(eent);  feats["energyentropym"]    = m; feats["energyentropystd"]   = s
    m, s = _safe_mean_std(sent);  feats["spectralentropym"]  = m; feats["spectralentropystd"] = s
    m, s = _safe_mean_std(sflux); feats["spectralfluxm"]     = m; feats["spectralfluxstd"]    = s

    # --- MFCC & ΔMFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    for i in range(13):
        m, s = _safe_mean_std(mfcc[i]); feats[f"mfccs{i+1}m"]   = m; feats[f"mfccs{i+1}std"] = s
    dmfcc = librosa.feature.delta(mfcc, order=1)
    for i in range(13):
        m, s = _safe_mean_std(dmfcc[i]); feats[f"amfccs{i+1}m"] = m; feats[f"amfccs{i+1}std"] = s

    # --- Chroma + deviation
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i in range(12):
        m, s = _safe_mean_std(chroma[i]); feats[f"chromavector{i+1}m"] = m; feats[f"chromavector{i+1}std"] = s
    ch_dev = chroma.std(axis=0); m, s = _safe_mean_std(ch_dev)
    feats["chromadeviationm"] = m; feats["chromadeviationstd"] = s

    # --- Beats loudness band ratios (24)
    feats.update(_compute_beats_loudness_band_ratio(y, sr, S_power=S_power, hop=hop))

    # --- BPM / histogram / danceability
    feats.update(_tempo_histogram_feats(y, sr, hop))

    # --- beats loudness mean/std (Essentia-style names)
    feats.update(_compute_beats_loudness_stats(y, sr, hop, S.shape[1]))

    return feats


# ================= Alignment & prediction =================
_PREFIX_RE = re.compile(r"^\d+-")
def _strip_prefix(c: str) -> str: return _PREFIX_RE.sub("", c)

# ΔMFCC aliases
_DMFFC_ALIASES: Dict[str, str] = {}
for i in range(1, 14):
    _DMFFC_ALIASES[f"delta_mfccs{i}m"]   = f"amfccs{i}m"
    _DMFFC_ALIASES[f"delta_mfccs{i}std"] = f"amfccs{i}std"
    _DMFFC_ALIASES[f"dmfccs{i}m"]        = f"amfccs{i}m"
    _DMFFC_ALIASES[f"dmfccs{i}std"]      = f"amfccs{i}std"
    _DMFFC_ALIASES[f"mfcc_delta{i}m"]    = f"amfccs{i}m"
    _DMFFC_ALIASES[f"mfcc_delta{i}std"]  = f"amfccs{i}std"
    _DMFFC_ALIASES[f"amfcc_{i}m"]        = f"amfccs{i}m"
    _DMFFC_ALIASES[f"amfcc_{i}std"]      = f"amfccs{i}std"

# core aliases
_CORE_ALIASES = {
    "zcr": "zcrm",
    "energy": "energym",
    "spectralflux": "spectralfluxm",
    "spectral_centroidm": "spectralcentroidm",
    "spectral_spreadm": "spectralspreadm",
    "spectral_rolloffm": "spectralrolloffm",
    "energy_entropy_m": "energyentropym",
    "energy_entropy_std": "energyentropystd",
    "spectral_entropy_m": "spectralentropym",
    "spectral_entropy_std": "spectralentropystd",
}

# chroma aliases
_CHROMA_ALIASES: Dict[str, str] = {}
for i in range(1, 13):
    _CHROMA_ALIASES[f"chroma_vector{i}m"]   = f"chromavector{i}m"
    _CHROMA_ALIASES[f"chroma_vector{i}std"] = f"chromavector{i}std"
    _CHROMA_ALIASES[f"chromavector_{i}m"]   = f"chromavector{i}m"
    _CHROMA_ALIASES[f"chromavector_{i}std"] = f"chromavector{i}std"
_CHROMA_ALIASES.update({
    "chroma_deviationm": "chromadeviationm",
    "chroma_deviationstd": "chromadeviationstd",
})

# BPM / Essentia-style aliases
_BPM_ALIASES = {
    "bpmessentia": "bpm",
    "bpm_histogram_first_peak_bpm": "bpm_histogram_first_peak_bpm",
    "bpm_histogram_first_peak_weight": "bpm_histogram_first_peak_weight",
    "bpm_histogram_second_peak_bpm": "bpm_histogram_second_peak_bpm",
    "bpm_histogram_second_peak_weight": "bpm_histogram_second_peak_weight",
    "bpm_histogram_second_peak_spread": "bpm_histogram_second_peak_spread",
    # tolerate typos/variants
    "bpmhistogramfirstpeakbpm": "bpm_histogram_first_peak_bpm",
    "bpmhistogramfirstpeakweight": "bpm_histogram_first_peak_weight",
    "bpmhistogramsecondpeakbpm": "bpm_histogram_second_peak_bpm",
    "bpmhistogramsecondpeakweight": "bpm_histogram_second_peak_weight",
    "bpmhistogramsecondpeakspread": "bpm_histogram_second_peak_spread",
}

def _alias_key(key: str, feats: Dict[str, float]) -> Tuple[str, float]:
    if key in feats and np.isfinite(feats[key]): return key, feats[key]
    if key in _DMFFC_ALIASES:
        k = _DMFFC_ALIASES[key];  return (k, feats[k]) if k in feats and np.isfinite(feats[k]) else (key, np.nan)
    if key in _CORE_ALIASES:
        k = _CORE_ALIASES[key];   return (k, feats[k]) if k in feats and np.isfinite(feats[k]) else (key, np.nan)
    if key in _CHROMA_ALIASES:
        k = _CHROMA_ALIASES[key]; return (k, feats[k]) if k in feats and np.isfinite(feats[k]) else (key, np.nan)
    if key in _BPM_ALIASES:
        k = _BPM_ALIASES[key];    return (k, feats[k]) if k in feats and np.isfinite(feats[k]) else (key, np.nan)
    return key, np.nan

def align_features_for_model(feat_dict: Dict[str, float], model_cols: List[str]) -> pd.DataFrame:
    row = {}
    for col in model_cols:
        key = _strip_prefix(col)
        if key in feat_dict and np.isfinite(feat_dict[key]):
            val = feat_dict[key]
        else:
            _, val = _alias_key(key, feat_dict)
        row[col] = val
    return pd.DataFrame([row], columns=model_cols)

def _group_of(col: str) -> str:
    n = _strip_prefix(col).lower()
    if n.startswith(("amfccs","mfcc_delta","delta_mfcc","dmfccs","amfcc_")): return "ΔMFCC"
    if n.startswith(("mfccs",)) and not n.startswith(("amfccs",)):        return "MFCC"
    if n.startswith(("chroma","chromavector","chroma_vector","chromadev")): return "Chroma"
    return "core"

def predict_one(path: str, model, encoder, sr: int, max_secs: int, model_cols: List[str], top_k: int):
    y, _ = load_audio_any(path, sr, max_secs)
    feats = extract_features(y, sr)
    X = align_features_for_model(feats, model_cols)

    pred_idx = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        k = min(top_k, len(proba))
        idxs = np.argsort(proba)[::-1][:k]
        top_probs  = [float(proba[i]) for i in idxs]
        top_labels = [str(x) for x in encoder.inverse_transform(np.asarray(idxs, int))]
    else:
        top_labels = [str(encoder.inverse_transform([pred_idx])[0])]
        top_probs  = [1.0]
    pred_label = str(encoder.inverse_transform([pred_idx])[0])
    return os.path.basename(path), pred_idx, pred_label, top_labels, top_probs, feats, X


# ================= ZIP helper =================
def make_zip(rows: List[Tuple[str, str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, basename, data in rows:
            safe = re.sub(r"[^\w\-\s.]", "_", label)
            zf.writestr(f"{safe}/{basename}", data)
    mem.seek(0)
    return mem.read()


# ================= UI =================
st.title("🎛️ MilkCrate — Drop audio/video → genre-organized ZIP")
st.caption("Build 2025-08-18 • MFCC13 + ΔMFCC13 + chroma + core + beats-loudness ratios + BPM/danceability • RNG pickle shim")
st.write(f"🔎 Runtime — NumPy: **{np.__version__}**, joblib: **{joblib.__version__}**, pandas: **{pd.__version__}**")

with st.sidebar:
    st.header("Settings")
    model_path   = st.text_input("Model path",   value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    tgt_sr       = st.selectbox("Target sample rate", options=[22050, 32000, 44100, 48000], index=0)
    top_k        = st.number_input("Top-K probabilities", 1, 10, DEFAULT_TOPK)
    max_secs     = st.number_input("Analyze up to (seconds)", 10, 600, DEFAULT_MAX_SECS)

model   = load_model(model_path)
encoder = load_encoder(encoder_path)
model_cols: List[str] = list(getattr(model, "feature_names_in_", []))

with st.expander("📦 Model feature inventory"):
    if model_cols:
        counts = {"core":0, "MFCC":0, "ΔMFCC":0, "Chroma":0}
        for c in model_cols:
            counts[_group_of(c)] += 1
        st.write(f"Total expected by model: **{len(model_cols)}**")
        st.json(counts)
        st.write("First 20 feature names (after numeric prefixes removed):")
        st.write([_strip_prefix(c) for c in model_cols[:20]])
    else:
        st.write("Model has no feature_names_in_; alignment falls back to column order.")

st.subheader("Upload audio/video files")
uploaded = st.file_uploader(
    "Drag and drop files here",
    type=None,
    accept_multiple_files=True,
    help="Any common audio/video: mp3, wav, aiff, flac, ogg, opus, m4a, mp4, mov, mkv, webm…"
)

if uploaded:
    tmp_paths: List[str] = []
    file_bytes: Dict[str, bytes] = {}
    for up in uploaded:
        p = _tmp_from_uploader(up)
        tmp_paths.append(p)
        file_bytes[p] = up.getvalue()

    rows = []
    last_X = None
    last_feats = None

    with st.spinner("Analyzing…"):
        for p in tmp_paths:
            try:
                base, pred_idx, pred_label, top_labels, top_probs, feats, X = predict_one(
                    p, model, encoder, int(tgt_sr), int(max_secs), model_cols, int(top_k)
                )
                last_X = X; last_feats = feats
                rows.append({
                    "file_name": base,
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                    "top_labels": ", ".join(top_labels),
                    "top_probs": ", ".join(f"{x:.6f}" for x in top_probs),
                })
            except Exception as e:
                rows.append({
                    "file_name": os.path.basename(p),
                    "pred_idx": -1,
                    "pred_label": f"ERROR: {e}",
                    "top_labels": "",
                    "top_probs": "",
                })

    with st.expander("📝 Diagnostics: feature alignment", expanded=True):
        if model_cols and last_X is not None:
            mask = np.isfinite(last_X.iloc[0].to_numpy())
            present = int(mask.sum())
            st.write(f"Expected features: {len(model_cols)}")
            st.write(f"Present (non-NaN) in DF: {present}")

            missing_cols = [c for c, ok in zip(model_cols, mask) if not ok]
            if missing_cols:
                show = []
                for c in missing_cols[:30]:
                    key = _strip_prefix(c)
                    alias, _ = _alias_key(key, last_feats or {})
                    have_direct = key in (last_feats or {})
                    have_alias  = alias in (last_feats or {}) and alias != key
                    show.append({
                        "model_col": c,
                        "key": key,
                        "alias_used": alias if have_alias else "",
                        "direct_feat_found": bool(have_direct),
                        "alias_feat_found": bool(have_alias),
                    })
                st.write("Missing columns (first 30) with alias probe:")
                st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)

            by_group = {"core":0, "MFCC":0, "ΔMFCC":0, "Chroma":0}
            for c, ok in zip(model_cols, mask):
                if ok: by_group[_group_of(c)] += 1
            st.write("By group (non-NaN counts from aligned row):")
            st.json(by_group)
        else:
            st.write("No feature_names_in_ or no upload yet.")

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    ok_rows = [(r["pred_label"], r["file_name"], file_bytes[p])
               for r, p in zip(rows, tmp_paths) if not r["pred_label"].startswith("ERROR:")]
    if ok_rows:
        zmem = io.BytesIO()
        with zipfile.ZipFile(zmem, "w", zipfile.ZIP_DEFLATED) as zf:
            for label, base, data in ok_rows:
                safe = re.sub(r"[^\w\-\s.]", "_", label)
                zf.writestr(f"{safe}/{base}", data)
        zmem.seek(0)
        st.download_button(
            "⬇️ Download ZIP (organized by predicted genre)",
            data=zmem.getvalue(),
            file_name="milkcrate_by_genre.zip",
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("Upload one or more audio/video files to get predictions and a genre-organized ZIP.")
