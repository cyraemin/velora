"""Speech emotion recognition with a pretrained model.

Model: ``emotion2vec/emotion2vec_plus_base``, a data2vec-style speech model trained on
~40,000 hours of pseudo-labelled emotional speech in many languages. It predicts
angry, disgusted, fearful, happy, neutral, sad and surprised, plus "other" and
"unknown". It is downloaded on first use (~360 MB) and runs on the CPU via FunASR.

Every probability Velora shows comes straight from this model's softmax output; the
only change is that "other" and "unknown" are reported together as "Other".
If the model cannot be loaded, no emotion prediction is produced at all.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

MODEL_ID = "emotion2vec/emotion2vec_plus_base"
MODEL_INFO = {
    "id": MODEL_ID,
    "architecture": "emotion2vec+ base (data2vec 2.0 backbone + emotion head)",
    "training_data": "~40k hours of pseudo-labelled emotional speech, multilingual",
    "classes": "neutral, happy, sad, angry, fearful, disgusted, surprised, other",
    "size": "~360 MB",
    "url": f"https://huggingface.co/{MODEL_ID}",
}

# Model label (English part of e.g. "生气/angry") -> display name.
LABELS = {
    "neutral": "Neutral", "happy": "Happy", "sad": "Sad", "angry": "Angry",
    "fearful": "Fearful", "disgusted": "Disgusted", "surprised": "Surprised",
    "other": "Other", "<unk>": "Other",
}
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised", "Other"]
EMOTION_COLORS = {
    "Neutral": "#2A9D7F",
    "Happy": "#6E56CF",
    "Sad": "#3A86D1",
    "Angry": "#E08A2E",
    "Fearful": "#C2417A",
    "Disgusted": "#6F8F2E",
    "Surprised": "#C9A227",
    "Other": "#9A96B0",
}

WINDOW_S = 3.0
MIN_HOP_S = 1.5
MAX_WINDOWS = 60
FULL_CLIP_MAX_S = 30.0   # longer clips use the mean of the window outputs


SHORT_NAMES = {MODEL_ID: "emotion2vec+", "superb/wav2vec2-base-superb-er": "wav2vec2 (IEMOCAP)"}


def short_name(emo: dict | None = None) -> str:
    """Compact model name for labels; older analyses may come from a previous model."""
    model = (emo or {}).get("model") or MODEL_ID
    return SHORT_NAMES.get(model, model)


def classes(emo: dict) -> list[str]:
    """Display classes of one result, in display order (older analyses used four classes)."""
    probs = emo.get("probs") or {}
    return [e for e in EMOTIONS if e in probs] + [e for e in probs if e not in EMOTIONS]


def color(emotion: str) -> str:
    return EMOTION_COLORS.get(emotion, "#9A96B0")


def _local_snapshot() -> str:
    """Path of the cached model; downloads it once if missing, so later runs need no network."""
    from huggingface_hub import snapshot_download

    patterns = ["*.yaml", "*.json", "*.pt", "*.txt"]
    try:
        return snapshot_download(MODEL_ID, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download(MODEL_ID, allow_patterns=patterns)


class _Model:
    _lock = threading.Lock()
    _model = None
    _error: str | None = None

    @classmethod
    def deps_installed(cls) -> bool:
        try:
            import funasr  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._model is None and cls._error is None:
                if not cls.deps_installed():
                    cls._error = "torch/funasr are not installed"
                    return None
                try:
                    import functools
                    import types

                    import funasr.train_utils.load_pretrained_model as loader
                    import torch
                    from funasr import AutoModel

                    # FunASR reads the whole checkpoint into RAM and then deep-copies it, so loading briefly
                    # needs over 1 GB. On machines short of memory torch crashed the process instead of raising.
                    # Memory-map the checkpoint (about +2 MB instead of +550 MB) and skip the redundant copy.
                    loader.torch = types.SimpleNamespace(load=functools.partial(torch.load, mmap=True))
                    loader.copy = types.SimpleNamespace(deepcopy=lambda obj: obj)
                    cls._model = AutoModel(model=_local_snapshot(), disable_update=True, disable_pbar=True,
                                           log_level="ERROR")
                except Exception as exc:  # offline on first run, corrupt cache, incompatible versions…
                    cls._error = f"{type(exc).__name__}: {exc}"
            return cls._model

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._model, cls._error = None, None


def model_status() -> tuple[str, str | None]:
    """('loaded' | 'error' | 'not loaded' | 'missing', error message)."""
    if _Model._model is not None:
        return "loaded", None
    if _Model._error:
        return "error", _Model._error
    return ("not loaded" if _Model.deps_installed() else "missing"), None


def retry_model():
    _Model.reset()


def _to_display(labels: list[str], scores: list[float]) -> dict[str, float]:
    d = {e: 0.0 for e in EMOTIONS}
    for label, score in zip(labels, scores):
        name = LABELS.get(label.split("/")[-1].strip())
        if name:
            d[name] += float(score)
    return {e: round(p, 4) for e, p in d.items()}


def _predict(clips: list[np.ndarray], sr: int, progress: Callable[[float], None] | None = None) -> list[dict]:
    model = _Model.get()
    if model is None:
        raise RuntimeError(_Model._error or "emotion model unavailable")
    if sr != 16_000:
        raise ValueError(f"the emotion model expects 16 kHz audio, got {sr} Hz")
    results = []
    for i, clip in enumerate(clips):
        out = model.generate(clip.astype(np.float32), granularity="utterance", extract_embedding=False,
                             disable_pbar=True)[0]
        results.append(_to_display(out["labels"], out["scores"]))
        if progress:
            progress((i + 1) / len(clips))
    return results


def windows(duration: float) -> list[tuple[float, float]]:
    if duration <= WINDOW_S * 1.5:
        return [(0.0, round(duration, 2))]
    hop = max(MIN_HOP_S, (duration - WINDOW_S) / (MAX_WINDOWS - 1))
    starts = np.arange(0.0, duration - WINDOW_S + 1e-6, hop)
    return [(round(float(s), 2), round(float(s) + WINDOW_S, 2)) for s in starts]


def analyze_emotion(y: np.ndarray, sr: int, speech_mask_fn=None,
                    progress: Callable[[float], None] | None = None) -> dict:
    """Run the classifier on the whole clip and on sliding windows for the timeline.

    ``speech_mask_fn(t0, t1) -> bool`` marks windows that contain speech;
    windows without speech are still classified but flagged so the UI can
    de-emphasise them. ``progress(fraction)`` is called as windows finish.

    Returns ``{"ok": False, "error": ...}`` when the model is unavailable.
    """
    duration = len(y) / sr
    try:
        wins = windows(duration)
        clips = [y[int(a * sr): int(b * sr)] for a, b in wins]
        full = duration <= FULL_CLIP_MAX_S and len(wins) > 1
        n = len(clips) + int(full)
        win_probs = _predict(clips, sr, (lambda f: progress(f * len(clips) / n)) if progress else None)
        if full:
            probs = _predict([y], sr)[0]
            scope = "full clip"
        elif len(wins) == 1:
            probs, scope = win_probs[0], "full clip"
        else:
            # Long recordings: average window outputs (stated explicitly in the UI).
            probs = {e: round(float(np.mean([w[e] for w in win_probs])), 4) for e in EMOTIONS}
            scope = f"mean of {len(wins)} windows"
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    top = max(probs, key=probs.get)
    speech = [bool(speech_mask_fn(a, b)) if speech_mask_fn else True for a, b in wins]
    return {
        "ok": True,
        "model": MODEL_ID,
        "scope": scope,
        "window_s": WINDOW_S,
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


def margin(emo: dict) -> float:
    """Gap between the top two classes; small gaps mean the model is unsure."""
    p = sorted(emo["probs"].values(), reverse=True)
    return p[0] - p[1] if len(p) > 1 else p[0]


def dominant_at(timeline: dict, t0: float, t1: float) -> str | None:
    """Model's most probable class over windows overlapping [t0, t1]."""
    if not timeline or not timeline.get("t"):
        return None
    st, en = np.array(timeline["start"]), np.array(timeline["end"])
    mask = (en > t0) & (st < t1)
    if not mask.any():
        mask = np.zeros(len(st), bool)
        mask[np.argmin(np.abs((st + en) / 2 - (t0 + t1) / 2))] = True
    present = [e for e in EMOTIONS if e in timeline] or [k for k in timeline if k not in ("t", "start", "end", "speech")]
    scores = {e: float(np.mean(np.array(timeline[e])[mask])) for e in present}
    return max(scores, key=scores.get)
