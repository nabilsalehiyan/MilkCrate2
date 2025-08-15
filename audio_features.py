from __future__ import annotations
import numpy as np
import pandas as pd
import librosa

FRAME = 2048
HOP = 512
SR = 22050

def _stat(x):
    x = np.asarray(x).ravel()
    return float(np.nanmean(x)), float(np.nanstd(x))

def _spectral_flux(S):
    d = np.diff(S, axis=1)
    d[d < 0] = 0.0
    return np.sum(d, axis=0)

def _bpm_peaks(onset_env, sr, hop_length):
    ac = librosa.autocorrelate(onset_env)
    ac[0] = 0
    if not np.any(ac > 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    lags = np.arange(1, len(ac))
    bpms = 60.0 * sr / (hop_length * lags)
    mask = (bpms >= 50) & (bpms <= 200)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    acm = ac[1:][mask]
    bpmm = bpms[mask]
    idx = np.argsort(acm)[::-1]
    first = idx[0]
    b1 = float(bpmm[first]); w1 = float(acm[first])
    if len(idx) > 1:
        second = idx[1]
        b2 = float(bpmm[second]); w2 = float(acm[second])
    else:
        b2, w2 = np.nan, np.nan
    wsum = (w1 if np.isfinite(w1) else 0.0) + (w2 if np.isfinite(w2) else 0.0)
    w1n = w1/wsum if wsum > 0 else np.nan
    w2n = w2/wsum if wsum > 0 else np.nan
    spread2 = abs(b2 - b1) if np.isfinite(b2) else np.nan
    return b1, w1n, b2, spread2, w2n

def extract_features_from_audio(path: str, sr: int = SR) -> dict:
    y, sr = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / sr
    if dur <= 0:
        raise ValueError("Empty audio")

    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP)) + 1e-9
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    sc = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=HOP)[0]
    sbw = librosa.feature.spectral_bandwidth(S=S, sr=sr, hop_length=HOP)[0]
    sro = librosa.feature.spectral_rolloff(S=S, sr=sr, hop_length=HOP, roll_percent=0.85)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=HOP)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)

    flux = _spectral_flux(S)

    rms_n = rms / (np.sum(rms) + 1e-9)
    ent = -np.sum(rms_n * np.log(rms_n + 1e-12))

    chroma_dev_per_frame = np.std(chroma, axis=0)
    b1, w1, b2, spread2, w2 = _bpm_peaks(onset_env, sr, HOP)

    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
        beat_rms = rms[beats] if len(beats) else rms
        bl_mean, bl_std = _stat(beat_rms)
    except Exception:
        bl_mean, bl_std = np.nan, np.nan

    onset_rate = float(len(onset_frames) / max(dur, 1e-6))

    feats = {}
    m, s = _stat(zcr); feats["1-ZCRm"] = m; feats["35-ZCRstd"] = s
    m, s = _stat(rms); feats["2-Energym"] = m; feats["36-Energystd"] = s
    feats["3-EnergyEntropym"] = float(ent); feats["37-EnergyEntropystd"] = 0.0
    m, s = _stat(sc); feats["4-SpectralCentroidm"] = m; feats["38-SpectralCentroidstd"] = s
    m, s = _stat(sbw); feats["5-SpectralSpreadm"] = m; feats["39-SpectralSpreadstd"] = s
    feats["6-SpectralEntropym"] = float(ent); feats["40-SpectralEntropystd"] = 0.0
    m, s = _stat(flux); feats["7-SpectralFluxm"] = m; feats["41-SpectralFluxstd"] = s
    m, s = _stat(sro); feats["8-SpectralRolloffm"] = m; feats["42-SpectralRolloffstd"] = s
    for idx in range(13):
        m, s = _stat(mfcc[idx])
        feats[f"{9+idx}-MFCCs{idx+1}m"] = m
        feats[f"{43+idx}-MFCCs{idx+1}std"] = s
    for c in range(12):
        m, s = _stat(chroma[c])
        feats[f"{22+c}-ChromaVector{c+1}m"] = m
        feats[f"{56+c}-ChromaVector{c+1}std"] = s
    m, s = _stat(chroma_dev_per_frame)
    feats["34-ChromaDeviationm"] = m
    feats["68-ChromaDeviationstd"] = s
    feats["69-BPM"] = float(tempo)
    feats["70-BPMconf"] = float(np.clip(np.max(onset_env) / (np.mean(onset_env) + 1e-9), 0, 10))
    feats["71-BPMessentia"] = np.nan
    feats["72-bpm_histogram_first_peak_bpm"] = b1
    feats["73-bpm_histogram_first_peak_weight"] = w1
    feats["74-bpm_histogram_second_peak_bpm"] = b2
    feats["75-bpm_histogram_second_peak_spread"] = spread2
    feats["76-bpm_histogram_second_peak_weight"] = w2
    feats["77-danceability"] = np.nan
    feats["78-beats_loudness.mean"] = bl_mean
    feats["79-beats_loudness.stdev"] = bl_std
    feats["80-onset_rate"] = onset_rate
    for i in range(81, 93):
        feats[f"{i}-beats_loudness_band_ratio." + ("mean" if i < 87 else "stdev") + f"{(i-80-1)%6+1}"] = np.nan
    return feats

def features_dataframe_from_audio_paths(paths: list[str]) -> pd.DataFrame:
    rows, names = [], []
    for p in paths:
        try:
            rows.append(extract_features_from_audio(p))
            names.append(p)
        except Exception:
            rows.append({})
            names.append(p)
    df = pd.DataFrame(rows)
    df.insert(0, "source_path", names)
    return df
