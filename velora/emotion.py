"""Speech emotion recognition with a pretrained model.

Model: ``superb/wav2vec2-base-superb-er`` — wav2vec 2.0 (base) fine-tuned on
IEMOCAP for four classes: neutral, happy, angry, sad. It is downloaded on first
use (~380 MB) and runs on the CPU.

Every probability Velora shows comes straight from this model's softmax output.
If the model cannot be loaded, no emotion prediction is produced at all.
"""
from __future__ import annotations

import os
import threading

import numpy as np

# transformers otherwise starts a background "safetensors conversion" thread with network calls for
# repos that ship .bin weights; running it concurrently with inference crashed the process on Windows.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

MODEL_ID = "superb/wav2vec2-base-superb-er"
MODEL_INFO = {
    "id": MODEL_ID,
    "architecture": "wav2vec 2.0 base + classification head",
    "training_data": "IEMOCAP (acted English dyadic sessions)",
    "classes": "neutral, happy, angry, sad",
    "url": f"https://huggingface.co/{MODEL_ID}",
}

# Model label -> display name, in fixed display/colour order.
LABELS = {"hap": "Happy", "neu": "Neutral", "sad": "Sad", "ang": "Angry"}
EMOTIONS = list(LABELS.values())
EMOTION_COLORS = {
    "Happy": "#6E56CF",
    "Neutral": "#2A9D7F",
    "Sad": "#3A86D1",
    "Angry": "#E08A2E",
}
EMOTION_ICONS = {"Happy": "😊", "Neutral": "😐", "Sad": "😔", "Angry": "😠"}

WINDOW_S = 2.0
MAX_WINDOWS = 90


def _local_snapshot() -> str:
    """Path of the cached model; downloads it once if missing. Loading from a local folder means
    no Hugging Face network calls on later runs."""
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.bin", "*.safetensors", "*.txt"]
    try:
        return snapshot_download(MODEL_ID, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download(MODEL_ID, allow_patterns=patterns)


class _Model:
    _lock = threading.Lock()
    _pipe = None
    _error: str | None = None

    @classmethod
    def deps_installed(cls) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._pipe is None and cls._error is None:
                if not cls.deps_installed():
                    cls._error = "torch/transformers are not installed"
                    return None
                try:
                    from transformers import pipeline
                    from transformers.utils import logging as hf_logging

                    hf_logging.disable_progress_bar()   # "Loading weights" bar garbles non-UTF-8 consoles

                    cls._pipe = pipeline("audio-classification", model=_local_snapshot(), device=-1)
                except Exception as exc:  # offline, corrupt cache, incompatible versions…
                    cls._error = f"{type(exc).__name__}: {exc}"
            return cls._pipe

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._pipe, cls._error = None, None


def model_status() -> tuple[str, str | None]:
    """('loaded' | 'error' | 'not loaded' | 'missing', error message)."""
    if _Model._pipe is not None:
        return "loaded", None
    if _Model._error:
        return "error", _Model._error
    return ("not loaded" if _Model.deps_installed() else "missing"), None


def retry_model():
    _Model.reset()


def _predict(clips: list[np.ndarray], sr: int) -> list[dict]:
    pipe = _Model.get()
    if pipe is None:
        raise RuntimeError(_Model._error or "emotion model unavailable")
    inputs = [{"raw": c.astype(np.float32), "sampling_rate": sr} for c in clips]
    out = pipe(inputs, top_k=None, batch_size=8)
    if out and isinstance(out[0], dict):
        out = [out]
    results = []
    for preds in out:
        d = {e: 0.0 for e in EMOTIONS}
        for p in preds:
            if p["label"] in LABELS:
                d[LABELS[p["label"]]] = round(float(p["score"]), 4)
        results.append(d)
    return results


def windows(duration: float) -> list[tuple[float, float]]:
    if duration <= WINDOW_S * 1.5:
        return [(0.0, round(duration, 2))]
    hop = max(1.0, (duration - WINDOW_S) / (MAX_WINDOWS - 1))
    starts = np.arange(0.0, duration - WINDOW_S + 1e-6, hop)
    return [(round(float(s), 2), round(float(s) + WINDOW_S, 2)) for s in starts]


def analyze_emotion(y: np.ndarray, sr: int, speech_mask_fn=None) -> dict:
    """Run the classifier on the whole clip and on sliding 2 s windows.

    ``speech_mask_fn(t0, t1) -> bool`` marks windows that contain speech;
    windows without speech are still classified but flagged so the UI can
    de-emphasise them.

    Returns ``{"ok": False, "error": ...}`` when the model is unavailable.
    """
    duration = len(y) / sr
    try:
        wins = windows(duration)
        clip_input = y if duration <= 30 else None
        clips = [y[int(a * sr): int(b * sr)] for a, b in wins]
        win_probs = _predict(clips, sr)
        if clip_input is not None:
            probs = _predict([clip_input], sr)[0]
            scope = "full clip"
        else:
            # Long recordings: average window outputs (stated explicitly in the UI).
            probs = {e: round(float(np.mean([w[e] for w in win_probs])), 4) for e in EMOTIONS}
            scope = f"mean of {len(wins)} windows"
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}" if not isinstance(exc, RuntimeError) else str(exc)}

    top = max(probs, key=probs.get)
    speech = [bool(speech_mask_fn(a, b)) if speech_mask_fn else True for a, b in wins]
    return {
        "ok": True,
        "model": MODEL_ID,
        "scope": scope,
        "probs": probs,
        "top": top,
        "confidence": probs[top],
        "timeline": {
            "t": [round((a + b) / 2, 2) for a, b in wins],
            "start": [a for a, _ in wins],
            "end": [b for _, b in wins],
            "speech": speech,
            **{e: [w[e] for w in win_probs] for e in EMOTIONS},
        },
    }


def dominant_at(timeline: dict, t0: float, t1: float) -> str | None:
    """Model's most probable class over windows overlapping [t0, t1]."""
    if not timeline or not timeline.get("t"):
        return None
    st, en = np.array(timeline["start"]), np.array(timeline["end"])
    mask = (en > t0) & (st < t1)
    if not mask.any():
        mask = np.zeros(len(st), bool)
        mask[np.argmin(np.abs((st + en) / 2 - (t0 + t1) / 2))] = True
    scores = {e: float(np.mean(np.array(timeline[e])[mask])) for e in EMOTIONS}
    return max(scores, key=scores.get)
