"""Acoustic feature extraction.

Frame-level tracks are computed once (``extract_frames``); clip- or window-level
statistics are then derived from any time slice (``summarize``).
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

HOP = 320            # 20 ms at 16 kHz
FRAME = 1024         # 64 ms analysis window
FMIN, FMAX = 65.0, 500.0
SYLLABLES_PER_WORD = 1.45

PROFILE_AXES = [
    ("pitch_level", "Pitch level"),
    ("expressiveness", "Expressiveness"),
    ("loudness", "Loudness"),
    ("dynamics", "Dynamics"),
    ("tempo", "Tempo"),
    ("brightness", "Brightness"),
    ("fluency", "Fluency"),
    ("steadiness", "Steadiness"),
]


def _db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-6))


def extract_frames(y: np.ndarray, sr: int) -> dict:
    """Compute all frame-level tracks on a common 20 ms grid."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    f0, voiced, vprob = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=FRAME, hop_length=HOP
    )
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=FRAME, hop_length=HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0]

    n = min(len(rms), len(f0), len(centroid), len(zcr))
    rms, f0, voiced, vprob = rms[:n], f0[:n], voiced[:n], vprob[:n]
    centroid, rolloff, flatness, bandwidth, zcr = (
        centroid[:n], rolloff[:n], flatness[:n], bandwidth[:n], zcr[:n]
    )
    rms_db = _db(rms)
    # Speech activity: above the noise floor and within 35 dB of the loudest frame.
    noise = np.percentile(rms_db, 10)
    thresh = max(noise + 8.0, rms_db.max() - 35.0)
    active = rms_db > thresh
    voiced = np.asarray(voiced, dtype=bool) & active
    f0 = np.where(voiced, f0, np.nan)

    # Syllable nuclei: peaks of a band-limited (250–2500 Hz) 10 ms energy envelope
    # that fall on voiced speech. Peaks are mapped back onto the 20 ms grid.
    sos = butter(4, [250, 2500], btype="band", fs=sr, output="sos")
    band = librosa.feature.rms(y=sosfiltfilt(sos, y), frame_length=400, hop_length=HOP // 2)[0]
    env = np.convolve(_db(band), np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(env, prominence=2.0, distance=10)
    near_voiced = np.convolve(voiced.astype(float), np.ones(5), mode="same") > 0
    nuclei = np.array([p // 2 for p in peaks if p // 2 < n and active[p // 2] and near_voiced[p // 2]], dtype=int)

    return {
        "sr": sr,
        "hop": HOP,
        "times": np.arange(n) * HOP / sr,
        "rms": rms,
        "rms_db": rms_db,
        "f0": f0,
        "voiced": voiced,
        "vprob": np.nan_to_num(vprob),
        "active": active,
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "bandwidth": bandwidth,
        "zcr": zcr,
        "nuclei": nuclei,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of consecutive True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def pauses(fr: dict, lo: int = 0, hi: int | None = None, min_s: float = 0.25) -> list[tuple[float, float]]:
    """Silent gaps between speech, as (start_s, end_s)."""
    hi = len(fr["active"]) if hi is None else hi
    act = fr["active"][lo:hi]
    idx = np.where(act)[0]
    if len(idx) < 2:
        return []
    first, last = idx[0], idx[-1]
    step = fr["hop"] / fr["sr"]
    out = []
    for s, e in _runs(~act[first:last + 1]):
        if (e - s) * step >= min_s:
            out.append(((lo + first + s) * step, (lo + first + e) * step))
    return out


def _nanstat(fn, x, default=0.0):
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else default


def summarize(fr: dict, t0: float | None = None, t1: float | None = None) -> dict:
    """Statistics for the slice [t0, t1) seconds (whole clip by default)."""
    step = fr["hop"] / fr["sr"]
    n = len(fr["rms"])
    lo = 0 if t0 is None else max(0, int(t0 / step))
    hi = n if t1 is None else min(n, max(lo + 1, int(t1 / step)))
    sl = slice(lo, hi)
    active = fr["active"][sl]
    voiced = fr["voiced"][sl]
    f0 = fr["f0"][sl]
    rms_db = fr["rms_db"][sl]
    dur = (hi - lo) * step
    speech_dur = float(active.sum() * step)

    vf0 = f0[np.isfinite(f0)]
    if len(vf0) >= 3:
        st = 12 * np.log2(vf0 / np.median(vf0))
        pitch_var_st = float(np.std(st))
        periods = 1.0 / vf0
        # only consecutive voiced frames count towards jitter
        cons = np.isfinite(f0[1:]) & np.isfinite(f0[:-1])
        p = 1.0 / f0
        dp = np.abs(np.diff(p))[cons]
        jitter = float(np.mean(dp) / np.mean(periods) * 100) if cons.any() else 0.0
        dr = np.abs(np.diff(rms_db))[np.asarray(voiced[1:] & voiced[:-1])]
        shimmer = float(np.mean(dr)) if len(dr) else 0.0
        p5, p95 = np.percentile(vf0, [5, 95])
    else:
        pitch_var_st = jitter = shimmer = 0.0
        p5 = p95 = 0.0

    act_db = rms_db[active] if active.any() else rms_db
    nuclei = fr["nuclei"]
    n_syll = int(np.sum((nuclei >= lo) & (nuclei < hi)))
    syll_rate = n_syll / speech_dur if speech_dur > 0.3 else 0.0
    gaps = pauses(fr, lo, hi)
    pause_time = sum(e - s for s, e in gaps)
    a_idx = np.where(active)[0]
    span = float((a_idx[-1] - a_idx[0] + 1) * step) if len(a_idx) else 0.0

    def amean(key):
        v = fr[key][sl]
        v = v[active] if active.any() else v
        return float(np.mean(v)) if len(v) else 0.0

    return {
        "duration": round(dur, 2),
        "speech_duration": round(speech_dur, 2),
        "speech_ratio": round(speech_dur / dur, 3) if dur else 0.0,
        "voiced_ratio": round(float(voiced.mean()), 3) if len(voiced) else 0.0,
        "pitch_mean": round(_nanstat(np.mean, f0), 1),
        "pitch_median": round(_nanstat(np.median, f0), 1),
        "pitch_std": round(_nanstat(np.std, f0), 1),
        "pitch_min": round(float(p5), 1),
        "pitch_max": round(float(p95), 1),
        "pitch_range": round(float(p95 - p5), 1),
        "pitch_var_st": round(pitch_var_st, 2),
        "energy_db": round(float(np.mean(act_db)), 1),
        "energy_std_db": round(float(np.std(act_db)), 2),
        "energy_rms": round(float(np.mean(fr["rms"][sl][active])) if active.any() else 0.0, 4),
        "centroid": round(amean("centroid"), 1),
        "rolloff": round(amean("rolloff"), 1),
        "bandwidth": round(amean("bandwidth"), 1),
        "flatness": round(amean("flatness"), 4),
        "zcr": round(amean("zcr"), 4),
        "syllables": n_syll,
        "syllable_rate": round(syll_rate, 2),
        # speaking rate over the whole speaking span (pauses included), like transcript-based wpm
        "wpm": round(n_syll / span * 60 / SYLLABLES_PER_WORD) if span > 0.5 else 0,
        "pause_count": len(gaps),
        "pause_mean": round(float(pause_time) / len(gaps), 2) if gaps else 0.0,
        "pause_ratio": round(float(pause_time) / span, 3) if span else 0.0,
        "jitter_pct": round(jitter, 2),
        "shimmer_db": round(shimmer, 2),
    }


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def profile(s: dict, relative_to: dict | None = None) -> dict:
    """Map statistics to 0–1 voice-profile axes.

    When ``relative_to`` (the whole-clip stats) is given, loudness is measured
    relative to the clip, which removes microphone-gain effects for windows.
    """
    if relative_to is not None:
        loud = 0.5 + (s["energy_db"] - relative_to["energy_db"]) / 16
    else:
        loud = (s["energy_db"] + 40) / 30
    pitch = s["pitch_median"] or (relative_to or {}).get("pitch_median", 0) or 150
    return {
        "pitch_level": _clip01((pitch - 90) / 190),
        "expressiveness": _clip01(s["pitch_var_st"] / 5.5),
        "loudness": _clip01(loud),
        "dynamics": _clip01(s["energy_std_db"] / 12),
        "tempo": _clip01((s["syllable_rate"] - 1.5) / 5.0),
        "brightness": _clip01((s["centroid"] - 700) / 2100),
        "fluency": _clip01(1 - s["pause_ratio"] / 0.45),
        "steadiness": _clip01(1 - (s["jitter_pct"] - 1.0) / 7.0),
    }


def waveform_envelope(y: np.ndarray, sr: int, bins: int = 1200) -> dict:
    n = max(1, len(y) // bins)
    m = len(y) // n
    blocks = y[: m * n].reshape(m, n)
    return {
        "t": (np.arange(m) * n / sr).round(3).tolist(),
        "min": blocks.min(axis=1).round(4).tolist(),
        "max": blocks.max(axis=1).round(4).tolist(),
    }


def mel_spectrogram(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME, hop_length=HOP, n_mels=80, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    times = np.arange(S_db.shape[1]) * HOP / sr
    freqs = librosa.mel_frequencies(n_mels=80, fmax=8000)
    return S_db, times, freqs


def mfcc(y: np.ndarray, sr: int, n: int = 13) -> tuple[np.ndarray, np.ndarray]:
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n, n_fft=FRAME, hop_length=HOP)
    times = np.arange(M.shape[1]) * HOP / sr
    return M, times
