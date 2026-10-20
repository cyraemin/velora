"""End-to-end analysis: decode → measure → classify → transcribe → interpret."""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Callable

import numpy as np

from . import audio as A
from . import emotion as E
from . import features as F
from . import interpret as I
from . import transcribe as T


def _clean(arr: np.ndarray, nd: int = 2) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in arr]


def analyze(
    data: bytes,
    name: str,
    source: str,
    *,
    use_emotion: bool = True,
    use_transcript: bool = True,
    whisper_size: str = "base",
    language: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[dict, np.ndarray, int]:
    """Analyse raw audio bytes. Returns (result, mono 16 kHz signal, sr).

    Optional stages (emotion model, transcription) never abort the analysis:
    their failures are recorded in the result instead.
    """
    step = progress or (lambda *_: None)
    step(0.05, "Decoding audio…")
    y, sr, truncated = A.load_audio(data, name)

    step(0.15, "Measuring acoustic features…")
    fr = F.extract_frames(y, sr)
    stats = F.summarize(fr)

    if use_emotion:
        step(0.40, "Running emotion model…")

        def has_speech(a, b):
            return F.summarize(fr, a, b)["speech_duration"] >= 0.25 * (b - a)

        emotion = E.analyze_emotion(y, sr, has_speech)
    else:
        emotion = {"ok": False, "error": "Emotion model disabled in settings.", "disabled": True}

    transcript, transcript_error = None, None
    if use_transcript:
        step(0.65, "Transcribing speech…")
        try:
            transcript = T.transcribe(y, sr, whisper_size, language)
            if transcript is None:
                transcript_error = T.transcriber_status()[1] or "faster-whisper is not installed."
            elif not transcript["segments"]:
                transcript_error = "No speech was recognised."
                transcript = None
        except Exception as exc:
            transcript_error = f"{type(exc).__name__}: {exc}"
    else:
        transcript_error = "Transcription disabled in settings."

    step(0.88, "Composing expressive text…")
    I.annotate_words(fr, stats, transcript)
    step_s = fr["hop"] / sr
    # keep at most ~3000 points per track for storage/plotting
    stride = max(1, len(fr["times"]) // 3000)
    result = {
        "id": uuid.uuid4().hex[:12],
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "source": source,
        "audio_hash": A.audio_hash(data),
        "original": {"filename": name, "ext": A.extension(name), "bytes": len(data)},
        "truncated": truncated,
        "stats": stats,
        "profile": {k: round(v, 3) for k, v in F.profile(stats).items()},
        "quality": A.recording_quality(y, sr),
        "emotion": emotion,
        "transcript": transcript,
        "transcript_error": transcript_error,
        "phrases": I.phrases(fr),
        "pauses": [[round(a, 2), round(b, 2)] for a, b in F.pauses(fr)],
        "tracks": {
            "step": step_s * stride,
            "t": _clean(fr["times"][::stride], 3),
            "f0": _clean(fr["f0"][::stride], 1),
            "rms_db": _clean(fr["rms_db"][::stride], 1),
            "centroid": _clean(fr["centroid"][::stride], 0),
            "active": fr["active"][::stride].astype(int).tolist(),
        },
        "envelope": F.waveform_envelope(y, sr),
    }
    result["insights"] = I.insights(stats, transcript)
    result["interpretation"] = I.interpretation(result)
    step(1.0, "Done")
    return result, y, sr
