"""Textual interpretation and expressive transcription.

Everything here is *derived* — rule-based wording built from measured acoustic
values and from the emotion model's actual output. Nothing is predicted here.
"""
from __future__ import annotations

import html

import numpy as np

from . import features as F
from .emotion import EMOTION_COLORS, dominant_at

PAUSE_MARK_S = 0.35
LONG_PAUSE_S = 0.8


# --------------------------------------------------------------------------- descriptors
def pitch_level_word(hz: float) -> str:
    if hz <= 0:
        return "not detectable"
    if hz < 120:
        return "low"
    if hz < 200:
        return "mid-range"
    return "high"


def intonation_word(st: float) -> str:
    if st < 1.5:
        return "fairly flat"
    if st < 3.0:
        return "moderately varied"
    return "highly varied"


def level_word(db: float) -> str:
    if db < -32:
        return "quiet"
    if db < -20:
        return "moderate"
    return "loud"


def pace_word(wpm: float) -> str:
    if wpm <= 0:
        return "not measurable"
    if wpm < 110:
        return "slow"
    if wpm <= 165:
        return "conversational"
    return "fast"


def speaking_rate(stats: dict, transcript: dict | None) -> tuple[float, str]:
    """Words per minute: from transcript word timings when available, else from syllable nuclei."""
    if transcript and transcript.get("segments"):
        words = [w for s in transcript["segments"] for w in s["words"]]
        if len(words) >= 3:
            span = words[-1]["end"] - words[0]["start"]
            if span > 0.5:
                return round(len(words) / span * 60), "transcript words ÷ speaking span"
    return float(stats["wpm"]), "syllable-count estimate"


def insights(stats: dict, transcript: dict | None) -> list[dict]:
    wpm, wpm_src = speaking_rate(stats, transcript)
    out = [
        {
            "icon": "♪",
            "label": "Pitch (median)",
            "value": f"{stats['pitch_median']:.0f} Hz" if stats["pitch_median"] else "—",
            "note": (
                f"{pitch_level_word(stats['pitch_median']).capitalize()} pitch, "
                f"{intonation_word(stats['pitch_var_st'])} intonation ({stats['pitch_var_st']:.1f} st SD)"
                if stats["pitch_median"] else "No voiced frames detected"
            ),
        },
        {
            "icon": "◉",
            "label": "Energy (RMS)",
            "value": f"{stats['energy_db']:.1f} dBFS",
            "note": f"{level_word(stats['energy_db']).capitalize()} level, {stats['energy_std_db']:.1f} dB variation",
        },
        {
            "icon": "⏱",
            "label": "Speaking rate",
            "value": f"{wpm:.0f} wpm" if wpm else "—",
            "note": f"{pace_word(wpm).capitalize()} pace · {wpm_src}",
        },
        {
            "icon": "◐",
            "label": "Spectral centroid",
            "value": f"{stats['centroid']:.0f} Hz",
            "note": "Brighter timbre" if stats["centroid"] > 1800 else "Warmer, darker timbre",
        },
        {
            "icon": "⏸",
            "label": "Pauses",
            "value": f"{stats['pause_count']}",
            "note": (
                f"avg {stats['pause_mean']:.2f}s · {stats['pause_ratio'] * 100:.0f}% of speaking span"
                if stats["pause_count"] else "Continuous delivery"
            ),
        },
    ]
    return out


def interpretation(result: dict) -> str:
    """A short paragraph that connects measured features with the model's output."""
    s = result["stats"]
    emo = result.get("emotion") or {}
    wpm, _ = speaking_rate(s, result.get("transcript"))
    parts = []
    if emo.get("ok"):
        probs = sorted(emo["probs"].items(), key=lambda kv: -kv[1])
        (top, p1), (second, p2) = probs[0], probs[1]
        certainty = "clearly" if p1 - p2 > 0.4 else ("tentatively" if p1 - p2 < 0.15 else "mainly")
        parts.append(
            f"The emotion model {certainty} classifies this recording as **{top.lower()}** "
            f"({p1 * 100:.0f}%), followed by {second.lower()} ({p2 * 100:.0f}%)."
        )
    else:
        parts.append("No emotion prediction is available because the emotion model could not run.")
    if s["pitch_median"]:
        parts.append(
            f"Your pitch centres around {s['pitch_median']:.0f} Hz with {intonation_word(s['pitch_var_st'])} "
            f"intonation, at a {level_word(s['energy_db'])} speaking level"
        )
    else:
        parts.append(f"Pitch was not reliably detectable; the speaking level is {level_word(s['energy_db'])}")
    if wpm:
        parts[-1] += f" and a {pace_word(wpm)} pace of about {wpm:.0f} words per minute."
    else:
        parts[-1] += "."
    cues = []
    if s["pitch_var_st"] >= 3:
        cues.append("lively intonation")
    elif s["pitch_var_st"] and s["pitch_var_st"] < 1.5:
        cues.append("an even, steady tone")
    if wpm > 165:
        cues.append("a brisk tempo")
    elif 0 < wpm < 110:
        cues.append("an unhurried tempo")
    if s["pause_ratio"] > 0.3:
        cues.append("frequent pauses")
    if s["energy_std_db"] > 9:
        cues.append("strong loudness contrasts")
    if cues:
        parts.append("Acoustically, the delivery is marked by " + ", ".join(cues[:-1])
                     + (" and " if len(cues) > 1 else "") + cues[-1] + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- expressive transcript
def annotate_words(fr: dict, stats: dict, transcript: dict | None) -> None:
    """Attach measured prosody (relative loudness, pitch, contour, preceding gap) to each word."""
    if not transcript:
        return
    step = fr["hop"] / fr["sr"]
    ref_f0 = stats["pitch_median"] or None
    words = [w for seg in transcript["segments"] for w in seg["words"]]

    def span(w):
        lo = int(w["start"] / step)
        return slice(lo, max(lo + 1, int(w["end"] / step)))

    # Loudness of each word relative to the median word, so emphasis is relative to the speaker.
    # 75th percentile is robust to silence that Whisper's word boundaries sometimes include
    word_db = [float(np.percentile(fr["rms_db"][span(w)], 75)) if len(fr["rms_db"][span(w)]) else np.nan for w in words]
    ref_db = float(np.nanmedian(word_db)) if words else 0.0
    prev_end = None
    for w, db in zip(words, word_db):
        f0 = fr["f0"][span(w)]
        vf = f0[np.isfinite(f0)]
        w["rel_db"] = round(db - ref_db, 1) if np.isfinite(db) else 0.0
        w["rel_st"] = round(float(12 * np.log2(np.median(vf) / ref_f0)), 1) if len(vf) >= 2 and ref_f0 else None
        contour = None
        if len(vf) >= 4:
            h = len(vf) // 2
            d = 12 * np.log2(np.median(vf[h:]) / np.median(vf[:h]))
            contour = "rise" if d > 2.5 else ("fall" if d < -2.5 else None)
        w["contour"] = contour
        w["gap_before"] = round(w["start"] - prev_end, 2) if prev_end is not None else 0.0
        prev_end = w["end"]


def phrases(fr: dict, min_pause: float = 0.3) -> list[dict]:
    """Speech phrases separated by measured pauses (used when no transcript exists)."""
    gaps = F.pauses(fr, min_s=min_pause)
    act = np.where(fr["active"])[0]
    if not len(act):
        return []
    step = fr["hop"] / fr["sr"]
    bounds, start = [], act[0] * step
    for s, e in gaps:
        bounds.append((start, s))
        start = e
    bounds.append((start, (act[-1] + 1) * step))
    out = []
    for a, b in bounds:
        st = F.summarize(fr, a, b)
        lo, hi = int(a / step), int(b / step)
        vf = fr["f0"][lo:hi]
        vf = vf[np.isfinite(vf)]
        contour = None
        if len(vf) >= 6:
            h = len(vf) // 2
            d = 12 * np.log2(np.median(vf[h:]) / np.median(vf[:h]))
            contour = "rising" if d > 1.0 else ("falling" if d < -1.0 else "level")
        out.append({
            "start": round(a, 2), "end": round(b, 2),
            "pitch": st["pitch_median"], "energy_db": st["energy_db"], "contour": contour,
        })
    return out


def _chip(emo: str | None) -> str:
    if not emo:
        return ""
    c = EMOTION_COLORS[emo]
    return (f'<span class="vl-chip" title="Emotion model, dominant class for this span">'
            f'<i style="background:{c}"></i>{emo}</span>')


def expressive_html(result: dict) -> str:
    """Transcript with measured prosody markup, rendered as HTML."""
    tr = result.get("transcript")
    emo = result.get("emotion") or {}
    tl = emo.get("timeline") if emo.get("ok") else None
    blocks = []
    if tr and tr.get("segments"):
        for seg in tr["segments"]:
            toks = []
            for w in seg["words"]:
                gap = w.get("gap_before", 0)
                if gap >= LONG_PAUSE_S:
                    toks.append(f'<span class="vl-pause">⏸ {gap:.1f}s</span>')
                elif gap >= PAUSE_MARK_S:
                    toks.append('<span class="vl-pause">…</span>')
                word = html.escape(w["word"].strip())
                cls = []
                if w.get("rel_db", 0) >= 3:
                    cls.append("vl-loud")
                elif w.get("rel_db", 0) <= -5:
                    cls.append("vl-soft")
                if w.get("rel_st") is not None and w["rel_st"] >= 3:
                    cls.append("vl-high")
                mark = {"rise": "↗", "fall": "↘"}.get(w.get("contour"), "")
                tip = f"{w['start']:.2f}–{w['end']:.2f}s · {w.get('rel_db', 0):+.1f} dB"
                if w.get("rel_st") is not None:
                    tip += f" · {w['rel_st']:+.1f} st"
                toks.append(f'<span class="{" ".join(cls)}" title="{tip}">{word}'
                            f'{f"<sup>{mark}</sup>" if mark else ""}</span>')
            dom = dominant_at(tl, seg["start"], seg["end"]) if tl else None
            blocks.append(
                f'<p class="vl-seg"><span class="vl-ts">{seg["start"]:.1f}s</span>{_chip(dom)} '
                + " ".join(toks) + "</p>"
            )
    else:
        for i, ph in enumerate(result.get("phrases", []), 1):
            dom = dominant_at(tl, ph["start"], ph["end"]) if tl else None
            desc = [f"{ph['end'] - ph['start']:.1f}s"]
            if ph["pitch"]:
                desc.append(f"{ph['pitch']:.0f} Hz")
            if ph["contour"]:
                desc.append(f"{ph['contour']} contour")
            desc.append(f"{ph['energy_db']:.0f} dBFS")
            blocks.append(
                f'<p class="vl-seg"><span class="vl-ts">{ph["start"]:.1f}s</span>{_chip(dom)} '
                f'<b>Phrase {i}</b> — {" · ".join(desc)}</p>'
            )
    return "".join(blocks) or "<p class='vl-muted'>No speech detected.</p>"


def expressive_markdown(result: dict) -> str:
    """Plain-text/Markdown rendering of the expressive transcript (for export)."""
    tr = result.get("transcript")
    lines = []
    if tr and tr.get("segments"):
        for seg in tr["segments"]:
            toks = []
            for w in seg["words"]:
                gap = w.get("gap_before", 0)
                if gap >= LONG_PAUSE_S:
                    toks.append(f"[pause {gap:.1f}s]")
                elif gap >= PAUSE_MARK_S:
                    toks.append("…")
                t = w["word"].strip()
                if w.get("rel_db", 0) >= 3:
                    t = f"**{t}**"
                elif w.get("rel_db", 0) <= -5:
                    t = f"_{t}_"
                t += {"rise": "↗", "fall": "↘"}.get(w.get("contour"), "")
                toks.append(t)
            lines.append(f"[{seg['start']:.1f}s] " + " ".join(toks))
    return "\n\n".join(lines)
