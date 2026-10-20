"""Audio loading and normalisation.

Everything downstream works on mono float32 audio at ``TARGET_SR``.
"""
from __future__ import annotations

import hashlib
import io
import os

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000
MAX_SECONDS = 180          # longer clips are truncated to keep analysis responsive
MIN_SECONDS = 0.5

SUPPORTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "aac", "webm", "mp4"]
MIME_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg",
    "m4a": "audio/mp4", "aac": "audio/aac", "webm": "audio/webm", "mp4": "audio/mp4",
}


def extension(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    return ext if ext in MIME_TYPES else "wav"


class AudioError(ValueError):
    """Raised when audio cannot be decoded or is unusable."""


def audio_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _decode(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    # libsndfile handles wav/flac/ogg/mp3 directly from memory.
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return y, sr
    except Exception:
        pass
    # Fallback: PyAV (bundled FFmpeg, installed with faster-whisper) for M4A/AAC, WebM/Opus, MP4…
    try:
        return _decode_av(data)
    except Exception as exc:
        raise AudioError(
            f"Could not decode '{filename or 'audio'}' — the file is damaged or not a supported audio "
            f"format ({', '.join(SUPPORTED_TYPES).upper()})."
        ) from exc


def _decode_av(data: bytes) -> tuple[np.ndarray, int]:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioError("No audio stream found.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        chunks = []
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise AudioError("No audio samples decoded.")
    return np.concatenate(chunks).astype(np.float32), TARGET_SR


def load_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, int, bool]:
    """Decode bytes to mono float32 at TARGET_SR.

    Returns ``(y, sr, truncated)``.
    """
    if not data:
        raise AudioError("The audio file is empty.")
    y, sr = _decode(data, filename)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = np.nan_to_num(y)
    truncated = len(y) > MAX_SECONDS * sr
    if truncated:
        y = y[: MAX_SECONDS * sr]
    if len(y) < MIN_SECONDS * sr:
        raise AudioError("The recording is too short — please provide at least half a second.")
    if np.max(np.abs(y)) < 1e-4:
        raise AudioError("The recording appears to be silent.")
    return y.astype(np.float32), sr, truncated


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def recording_quality(y: np.ndarray, sr: int) -> dict:
    """Simple microphone-check metrics: level, clipping and an SNR estimate."""
    peak = float(np.max(np.abs(y)))
    frame = int(0.03 * sr)
    n = max(1, len(y) // frame)
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    # floor at -90 dBFS so digital silence doesn't produce absurd SNR values
    rms_db = np.maximum(20 * np.log10(rms + 1e-12), -90.0)
    noise_db = float(np.percentile(rms_db, 10))
    speech_db = float(np.percentile(rms_db, 90))
    clipped = float(np.mean(np.abs(y) > 0.99))
    return {
        "peak_dbfs": round(float(20 * np.log10(peak + 1e-12)), 1),
        "noise_floor_db": round(noise_db, 1),
        "speech_level_db": round(speech_db, 1),
        "snr_db": round(min(speech_db - noise_db, 60.0), 1),   # capped: > 60 dB is effectively noise-free
        "clipping_pct": round(clipped * 100, 2),
    }


def synthetic_demo(seconds: float = 8.0, sr: int = TARGET_SR) -> np.ndarray:
    """A voice-like synthetic signal (glottal pulses + formants + syllable envelope).

    Used only to exercise the pipeline without a microphone; emotion results on it
    are not meaningful.
    """
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 190 + 35 * np.sin(2 * np.pi * 0.35 * t) + 20 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = sum((1 / k) * np.sin(k * phase) for k in range(1, 25))
    # crude formant shaping
    from scipy.signal import lfilter

    y = src
    for fc, bw in [(700, 110), (1220, 120), (2600, 160)]:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        y = lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], y)
    syll = 0.5 * (1 + np.sin(2 * np.pi * 3.8 * t - np.pi / 2)) ** 1.5
    gaps = ((t % 2.6) < 2.1).astype(float)
    env = syll * gaps
    y = y / (np.max(np.abs(y)) + 1e-9)
    y = y * env + 0.004 * rng.standard_normal(len(t))
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
    return y.astype(np.float32)
