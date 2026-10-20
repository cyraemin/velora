"""Speech-to-text with word timestamps via faster-whisper (optional dependency)."""
from __future__ import annotations

import threading

import numpy as np

WHISPER_SIZES = ["base", "tiny"]   # ~145 MB / ~75 MB downloads; larger sizes skipped on purpose


class _Whisper:
    _lock = threading.Lock()
    _models: dict = {}
    _error: str | None = None

    @classmethod
    def available(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    @classmethod
    def get(cls, size: str):
        with cls._lock:
            if size not in cls._models:
                try:
                    from faster_whisper import WhisperModel

                    try:  # use the cached copy without contacting the Hub
                        cls._models[size] = WhisperModel(size, device="cpu", compute_type="int8",
                                                         local_files_only=True)
                    except Exception:  # first use: download (~75–145 MB)
                        cls._models[size] = WhisperModel(size, device="cpu", compute_type="int8")
                    cls._error = None
                except Exception as exc:
                    cls._error = f"{type(exc).__name__}: {exc}"
                    return None
            return cls._models[size]


def transcriber_status() -> tuple[str, str | None]:
    if _Whisper._models:
        return "loaded", None
    if _Whisper._error:
        return "error", _Whisper._error
    return ("available" if _Whisper.available() else "missing"), None


def transcribe(y: np.ndarray, sr: int, size: str = "base", language: str | None = None) -> dict | None:
    """Return {"language", "text", "segments": [{start, end, text, words: [...]}]} or None."""
    if not _Whisper.available():
        return None
    model = _Whisper.get(size)
    if model is None:
        return None
    assert sr == 16_000, "faster-whisper expects 16 kHz audio"
    segments, info = model.transcribe(
        y.astype(np.float32),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    segs = []
    for s in segments:
        words = [
            {
                "start": round(float(w.start), 2),
                "end": round(float(w.end), 2),
                "word": w.word,
                "prob": round(float(w.probability), 3),
            }
            for w in (s.words or [])
        ]
        segs.append(
            {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip(), "words": words}
        )
    text = " ".join(s["text"] for s in segs).strip()
    return {
        "language": info.language,
        "language_prob": round(float(info.language_probability), 3),
        "model": size,
        "text": text,
        "segments": segs,
    }
