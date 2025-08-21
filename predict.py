#!/usr/bin/env python3
"""
MilkCrate CLI (no Streamlit)

Train, evaluate, and predict genres from audio using the same feature
pipeline as the Streamlit app.

Examples
--------
# Train & evaluate (80/20 split), using a CSV with columns [path, genre]
python predict.py train \
  --csv /path/to/beatport.csv \
  --audio-root /path/to/audio_root \
  --label-col genre \
  --sr 22050 --max-secs 120 \
  --model-out artifacts/model_version1beatport.joblib \
  --encoder-out artifacts/label_encoder.joblib

# Evaluate an existing model on a CSV
python predict.py eval \
  --csv /path/to/beatport_val.csv \
  --audio-root /path/to/audio_root \
  --label-col genre \
  --model artifacts/model_version1beatport.joblib \
  --encoder artifacts/label_encoder.joblib

# Predict for a few files
python predict.py predict \
  --model artifacts/model_version1beatport.joblib \
  --encoder artifacts/label_encoder.joblib \
  --files /path/track1.mp3 /path/track2.aiff \
  --sr 22050 --max-secs 120 --top-k 5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
import tempfile
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
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

# ================= RNG-safe loader (for old pickles) =================
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

# ================= Audio/video decode =================
def load_audio_any(path: str, target_sr: int, max_secs: int) -> Tuple[np.ndarray, int]:
    """Load audio from common audio OR video containers."""
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
    beats = _beat_boundaries(y, sr, hop, n_frames)
    if not beats:
        return {"beats_loudness.mean": float("nan"), "beats_loudness.stdev": float("nan")}
    n_fft = 2048
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
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if oenv.size == 0:
        return {k: float("nan") for k in [
            "bpm", "bpmconf", "bpm_histogram_first_peak_bpm", "bpm_histogram_first_peak_weight",
            "bpm_histogram_second_peak_bpm", "bpm_histogram_second_peak_weight", "bpm_histogram_second_peak_spread",
            "danceability"
        ]}
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop)
    tg_mean = tg.mean(axis=1)
    tempi = librosa.tempo_frequencies(tg_mean.shape[0], sr=sr, hop_length=hop)

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

    h_vals = np.maximum(h_vals, 0)
    h_norm = h_vals / float(h_vals.max()) if float(h_vals.max()) > 0 else h_vals

    order = np.argsort(h_vals)[::-1]
    i1 = int(order[0]); bpm1 = float(t_vals[i1]); w1 = float(h_norm[i1])
    i2 = int(order[1]) if order.size > 1 else i1
    bpm2 = float(t_vals[i2]); w2 = float(h_norm[i2])

    lo = max(0, i2 - 3); hi = min(len(t_vals), i2 + 4)
    t_win = t_vals[lo:hi]; w_win = h_norm[lo:hi]
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
        "danceability": w1,
    }

# ================= Extraction (92+ cols) =================
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

    # --- beats loudness mean/std
    feats.update(_compute_beats_loudness_stats(y, sr, hop, S.shape[1]))

    # --- onset_rate
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        on_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units="frames")
        dur = len(y) / float(sr) if sr else 0.0
        feats["onset_rate"] = float(len(on_frames) / dur) if dur > 0 else float("nan")
    except Exception:
        feats["onset_rate"] = float("nan")

    return feats

# ================= Alignment & prediction helpers =================
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
    "onsetrate": "onset_rate",
}

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

_BPM_ALIASES = {
    "bpmessentia": "bpm",
    "bpm_histogram_first_peak_bpm": "bpm_histogram_first_peak_bpm",
    "bpm_histogram_first_peak_weight": "bpm_histogram_first_peak_weight",
    "bpm_histogram_second_peak_bpm": "bpm_histogram_second_peak_bpm",
    "bpm_histogram_second_peak_weight": "bpm_histogram_second_peak_weight",
    "bpm_histogram_second_peak_spread": "bpm_histogram_second_peak_spread",
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

# ================= Dataset utilities =================
def _guess_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def _resolve_path(p: str, root: Optional[str]) -> str:
    if os.path.isabs(p): return p
    return os.path.join(root, p) if root else p

def _load_row_audio(row_path: str, sr: int, max_secs: int) -> Tuple[np.ndarray, int]:
    return load_audio_any(row_path, sr, max_secs)

def build_feature_table(
    df: pd.DataFrame,
    audio_root: Optional[str],
    path_col: str,
    sr: int,
    max_secs: int,
    cache_parquet: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract features for each row in df. Optionally cache to parquet.
    Returns a DataFrame with one row per item and all feature columns + 'path'.
    """
    if cache_parquet and os.path.exists(cache_parquet):
        return pd.read_parquet(cache_parquet)

    rows = []
    for i, row in df.iterrows():
        p = _resolve_path(str(row[path_col]), audio_root)
        try:
            y, _ = _load_row_audio(p, sr, max_secs)
            feats = extract_features(y, sr)
            feats["path"] = p
            rows.append(feats)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed {p}: {e}\n")

    feat_df = pd.DataFrame(rows)
    # Keep a stable column order: sort columns alphabetically except keep 'path' last
    cols = sorted([c for c in feat_df.columns if c != "path"])
    feat_df = feat_df[cols + ["path"]]
    if cache_parquet:
        os.makedirs(os.path.dirname(cache_parquet), exist_ok=True)
        feat_df.to_parquet(cache_parquet, index=False)
    return feat_df

# ================= Training / Evaluation =================
def _get_model(kind: str = "lgbm", random_state: int = 42):
    kind = (kind or "lgbm").lower()
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )
        except Exception:
            sys.stderr.write("[INFO] LightGBM not available; falling back to RandomForest.\n")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=random_state
    )

def train_and_eval(
    csv_path: str,
    audio_root: Optional[str],
    label_col: Optional[str],
    path_col_candidates: Sequence[str],
    sr: int,
    max_secs: int,
    model_out: str,
    encoder_out: str,
    cache_parquet: Optional[str],
    model_kind: str = "lgbm",
    test_size: float = 0.2,
    random_state: int = 42,
    top_k_eval: int = 3,
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    df = pd.read_csv(csv_path)
    path_col = _guess_col(df, path_col_candidates)
    if path_col is None:
        raise ValueError(f"Could not find a path column. Tried: {path_col_candidates}. Columns: {list(df.columns)}")

    if label_col is None:
        label_col = _guess_col(df, ["label", "genre", "genre_top", "subgenre"])
        if label_col is None:
            raise ValueError("Could not infer label column. Use --label-col to specify.")

    print(f"[INFO] Using path column = '{path_col}', label column = '{label_col}'")

    feat_df = build_feature_table(df[[path_col, label_col]], audio_root, path_col, sr, max_secs, cache_parquet)

    # Align features: use all columns except 'path'
    feature_cols = [c for c in feat_df.columns if c != "path"]
    X_all = feat_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[label_col].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # Model
    model = _get_model(kind=model_kind, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print("\n=== Evaluation (holdout) ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {f1m:.4f}")

    # Top-k accuracy if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        topk = min(top_k_eval, proba.shape[1])
        topk_hits = 0
        for i, p in enumerate(proba):
            idxs = np.argsort(p)[::-1][:topk]
            if y_test[i] in idxs:
                topk_hits += 1
        topk_acc = topk_hits / len(y_test)
        print(f"Top-{topk} Acc: {topk_acc:.4f}")

    print("\n=== Per-class report ===")
    target_names = [str(x) for x in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    os.makedirs("artifacts", exist_ok=True)
    cm_path = os.path.join("artifacts", "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"[INFO] Saved confusion matrix CSV → {cm_path}")

    # Persist artifacts
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(le, encoder_out)
    print(f"[INFO] Saved model → {model_out}")
    print(f"[INFO] Saved label encoder → {encoder_out}")

def eval_only(
    csv_path: str,
    audio_root: Optional[str],
    label_col: Optional[str],
    path_col_candidates: Sequence[str],
    sr: int,
    max_secs: int,
    model_path: str,
    encoder_path: str,
    cache_parquet: Optional[str],
    top_k_eval: int = 3,
):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    df = pd.read_csv(csv_path)
    path_col = _guess_col(df, path_col_candidates)
    if path_col is None:
        raise ValueError(f"Could not find a path column. Tried: {path_col_candidates}. Columns: {list(df.columns)}")

    if label_col is None:
        label_col = _guess_col(df, ["label", "genre", "genre_top", "subgenre"])
        if label_col is None:
            raise ValueError("Could not infer label column. Use --label-col to specify.")

    print(f"[INFO] Using path column = '{path_col}', label column = '{label_col}'")

    feat_df = build_feature_table(df[[path_col, label_col]], audio_root, path_col, sr, max_secs, cache_parquet)

    feature_cols = [c for c in feat_df.columns if c != "path"]
    X_all = feat_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[label_col].values

    # Load model & encoder
    model = _safe_joblib_load(model_path)
    le = joblib.load(encoder_path)
    y_enc = le.transform(y_all)

    # Align columns if model has feature_names_in_
    model_cols = list(getattr(model, "feature_names_in_", []))
    if model_cols:
        # Build a mapping feature name -> index from feat_df
        available = {c: i for i, c in enumerate(feature_cols)}
        X_aligned = np.full((X_all.shape[0], len(model_cols)), np.nan, dtype=np.float32)
        for j, c in enumerate(model_cols):
            key = _strip_prefix(c)
            # direct
            if key in available:
                X_aligned[:, j] = X_all[:, available[key]]
            else:
                # alias
                # (For eval-only, aliases across arrays is cumbersome; for best results, re-extract with the same code.)
                pass
        X_use = X_aligned
    else:
        X_use = X_all

    y_pred = model.predict(X_use)
    acc = accuracy_score(y_enc, y_pred)
    f1m = f1_score(y_enc, y_pred, average="macro")
    print("\n=== Evaluation ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {f1m:.4f}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_use)
        topk = min(top_k_eval, proba.shape[1])
        topk_hits = 0
        for i, p in enumerate(proba):
            idxs = np.argsort(p)[::-1][:topk]
            if y_enc[i] in idxs:
                topk_hits += 1
        topk_acc = topk_hits / len(y_enc)
        print(f"Top-{topk} Acc: {topk_acc:.4f}")

    target_names = [str(x) for x in le.classes_]
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n=== Per-class report ===")
    print(classification_report(y_enc, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_enc, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    os.makedirs("artifacts", exist_ok=True)
    cm_path = os.path.join("artifacts", "confusion_matrix_eval.csv")
    cm_df.to_csv(cm_path)
    print(f"[INFO] Saved confusion matrix CSV → {cm_path}")

# ================= Single-file prediction =================
def predict_files(
    files: Sequence[str],
    model_path: str,
    encoder_path: str,
    sr: int,
    max_secs: int,
    top_k: int
):
    model = _safe_joblib_load(model_path)
    encoder = joblib.load(encoder_path)
    model_cols: List[str] = list(getattr(model, "feature_names_in_", []))  # may be empty

    rows = []
    for fp in files:
        y, _ = load_audio_any(fp, sr, max_secs)
        feats = extract_features(y, sr)

        if model_cols:
            X = align_features_for_model(feats, model_cols)
        else:
            # if model was trained on a plain numpy array without names, sort features by key
            cols = sorted(feats.keys())
            X = pd.DataFrame([[feats[c] for c in cols]], columns=cols)

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

        rows.append({
            "file": fp,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "top_labels": ", ".join(top_labels),
            "top_probs": ", ".join(f"{x:.6f}" for x in top_probs),
        })

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

# ================= CLI =================
def main():
    p = argparse.ArgumentParser(description="MilkCrate CLI (train/eval/predict) — no Streamlit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common defaults
    path_candidates = ["path", "filepath", "file", "audio_path", "track_path", "rel_path"]

    # train
    p_tr = sub.add_parser("train", help="Train a model then evaluate on a holdout split")
    p_tr.add_argument("--csv", required=True, help="CSV with at least audio path and label columns")
    p_tr.add_argument("--audio-root", default=None, help="Root directory to prepend to relative paths")
    p_tr.add_argument("--label-col", default=None, help="Name of label column (default: try genre/label/subgenre)")
    p_tr.add_argument("--sr", type=int, default=DEFAULT_SR)
    p_tr.add_argument("--max-secs", type=int, default=DEFAULT_MAX_SECS)
    p_tr.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    p_tr.add_argument("--encoder-out", default=DEFAULT_ENCODER_PATH)
    p_tr.add_argument("--cache-parquet", default="artifacts/features.parquet", help="Cache extracted features")
    p_tr.add_argument("--model-kind", default="lgbm", choices=["lgbm","rf"])
    p_tr.add_argument("--test-size", type=float, default=0.2)
    p_tr.add_argument("--random-state", type=int, default=42)
    p_tr.add_argument("--top-k-eval", type=int, default=3)

    # eval
    p_ev = sub.add_parser("eval", help="Evaluate an existing model on a CSV")
    p_ev.add_argument("--csv", required=True)
    p_ev.add_argument("--audio-root", default=None)
    p_ev.add_argument("--label-col", default=None)
    p_ev.add_argument("--sr", type=int, default=DEFAULT_SR)
    p_ev.add_argument("--max-secs", type=int, default=DEFAULT_MAX_SECS)
    p_ev.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p_ev.add_argument("--encoder", default=DEFAULT_ENCODER_PATH)
    p_ev.add_argument("--cache-parquet", default=None, help="Optional cache (will be created if missing)")
    p_ev.add_argument("--top-k-eval", type=int, default=3)

    # predict
    p_pr = sub.add_parser("predict", help="Predict labels for given audio files")
    p_pr.add_argument("--files", nargs="+", required=True)
    p_pr.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p_pr.add_argument("--encoder", default=DEFAULT_ENCODER_PATH)
    p_pr.add_argument("--sr", type=int, default=DEFAULT_SR)
    p_pr.add_argument("--max-secs", type=int, default=DEFAULT_MAX_SECS)
    p_pr.add_argument("--top-k", type=int, default=DEFAULT_TOPK)

    args = p.parse_args()

    if args.cmd == "train":
        train_and_eval(
            csv_path=args.csv,
            audio_root=args.audio_root,
            label_col=args.label_col,
            path_col_candidates=path_candidates,
            sr=args.sr,
            max_secs=args.max_secs,
            model_out=args.model_out,
            encoder_out=args.encoder_out,
            cache_parquet=args.cache_parquet,
            model_kind=args.model_kind,
            test_size=args.test_size,
            random_state=args.random_state,
            top_k_eval=args.top_k_eval,
        )
    elif args.cmd == "eval":
        eval_only(
            csv_path=args.csv,
            audio_root=args.audio_root,
            label_col=args.label_col,
            path_col_candidates=path_candidates,
            sr=args.sr,
            max_secs=args.max_secs,
            model_path=args.model,
            encoder_path=args.encoder,
            cache_parquet=args.cache_parquet,
            top_k_eval=args.top_k_eval,
        )
    elif args.cmd == "predict":
        predict_files(
            files=args.files,
            model_path=args.model,
            encoder_path=args.encoder,
            sr=args.sr,
            max_secs=args.max_secs,
            top_k=args.top_k,
        )
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
