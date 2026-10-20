"""Comparison of two analyses: compatibility checks, measured deltas, model outputs,
word-aligned delivery differences, and exportable reports.

All numbers are taken from the two stored analyses; nothing is re-predicted here.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import re
import zipfile

import numpy as np

from . import emotion as E
from . import interpret as I

SAME_TEXT = 0.8      # word-sequence similarity above which we treat the scripts as "the same words"
PARTIAL_TEXT = 0.4


# --------------------------------------------------------------------------- helpers
def _norm(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower())


def words(result: dict) -> list[dict]:
    tr = result.get("transcript")
    if not tr:
        return []
    out = []
    for seg in tr["segments"]:
        for w in seg["words"]:
            n = _norm(w["word"])
            if n:
                out.append({**w, "norm": n, "text": w["word"].strip()})
    return out


def metric_rows(ra: dict, rb: dict) -> list[dict]:
    sa, sb = ra["stats"], rb["stats"]
    wa, _ = I.speaking_rate(sa, ra.get("transcript"))
    wb, _ = I.speaking_rate(sb, rb.get("transcript"))
    spec = [
        ("Median pitch", sa["pitch_median"], sb["pitch_median"], "Hz", 0, True),
        ("Pitch variation", sa["pitch_var_st"], sb["pitch_var_st"], "st", 1, True),
        ("Pitch range (p5–p95)", sa["pitch_range"], sb["pitch_range"], "Hz", 0, True),
        ("Energy (RMS)", sa["energy_db"], sb["energy_db"], "dBFS", 1, False),
        ("Energy variation", sa["energy_std_db"], sb["energy_std_db"], "dB", 1, False),
        ("Speaking rate", wa, wb, "wpm", 0, True),
        ("Articulation rate", sa["syllable_rate"], sb["syllable_rate"], "syll/s", 2, True),
        ("Pauses", sa["pause_count"], sb["pause_count"], "", 0, False),
        ("Pause ratio", sa["pause_ratio"] * 100, sb["pause_ratio"] * 100, "%", 0, False),
        ("Spectral centroid", sa["centroid"], sb["centroid"], "Hz", 0, True),
        ("Speech time", sa["speech_duration"], sb["speech_duration"], "s", 1, True),
        ("Duration", sa["duration"], sb["duration"], "s", 1, True),
    ]
    rows = []
    for name, a, b, unit, nd, rel in spec:
        a, b = float(a or 0), float(b or 0)
        missing = (unit == "Hz" and "pitch" in name.lower() and (not a or not b)) or (unit == "wpm" and (not a or not b))
        rows.append({
            "feature": name, "a": a, "b": b, "unit": unit, "nd": nd,
            "delta": None if missing else b - a,
            "pct": (b - a) / a * 100 if rel and a and not missing else None,
        })
    return rows


def fmt(v: float | None, nd: int, unit: str = "", sign: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return f"{s} {unit}".strip()


# --------------------------------------------------------------------------- checks
def checks(ra: dict, rb: dict) -> list[tuple[str, str]]:
    """(level, message) with level in {'error','warning','info'}."""
    out = []
    if ra["id"] == rb["id"]:
        out.append(("error", "A and B are the same analysis — choose two different recordings."))
    elif ra.get("audio_hash") and ra.get("audio_hash") == rb.get("audio_hash"):
        out.append(("warning", "A and B contain identical audio, so every difference will be zero."))
    for lbl, r in (("A", ra), ("B", rb)):
        s = r["stats"]
        if s["speech_duration"] < 1.0:
            out.append(("warning", f"Recording {lbl} has only {s['speech_duration']:.1f}s of detected speech — "
                                   "its statistics are unreliable."))
        if not s["pitch_median"]:
            out.append(("warning", f"No pitch was detectable in recording {lbl}; pitch comparisons are skipped."))
        if r.get("source") == "demo":
            out.append(("info", f"Recording {lbl} is the synthetic demo signal; its model outputs carry no meaning."))
        if r.get("truncated"):
            out.append(("info", f"Recording {lbl} was longer than 3 minutes and only its first 3 minutes were analysed."))
    da, db = ra["stats"]["duration"], rb["stats"]["duration"]
    if min(da, db) > 0 and max(da, db) / min(da, db) > 3:
        out.append(("warning", f"Durations differ a lot ({da:.1f}s vs {db:.1f}s). Rates and averages remain "
                               "comparable, but counts (pauses, syllables) are not."))
    qa, qb = ra.get("quality", {}), rb.get("quality", {})
    if qa and qb:
        if min(qa["snr_db"], qb["snr_db"]) < 12:
            out.append(("warning", "At least one recording is noisy (estimated SNR < 12 dB); pitch and energy "
                                   "measurements may be affected."))
        if abs(qa["peak_dbfs"] - qb["peak_dbfs"]) > 10:
            out.append(("info", "Recording levels differ by more than 10 dB — loudness differences may reflect "
                                "microphone gain rather than delivery."))
    ea, eb = ra.get("emotion") or {}, rb.get("emotion") or {}
    if not (ea.get("ok") and eb.get("ok")):
        miss = [lbl for lbl, e in (("A", ea), ("B", eb)) if not e.get("ok")]
        out.append(("info", f"No emotion-model output for recording {' and '.join(miss)} — the emotion comparison "
                            "is limited to what is available."))
    elif ea.get("model") != eb.get("model"):
        out.append(("warning", "The two recordings were classified by different emotion models."))
    ta, tb = ra.get("transcript"), rb.get("transcript")
    if not (ta and tb):
        miss = [lbl for lbl, t in (("A", ta), ("B", tb)) if not t]
        out.append(("info", f"No transcript for recording {' and '.join(miss)} — the word-by-word delivery "
                            "comparison needs both."))
    elif ta["language"] != tb["language"]:
        out.append(("warning", f"Different languages detected ({ta['language']} vs {tb['language']}); "
                               "word-level comparison is disabled."))
    return out


# --------------------------------------------------------------------------- words
def align(ra: dict, rb: dict) -> dict:
    """Align the two transcripts word by word."""
    wa, wb = words(ra), words(rb)
    ta, tb = ra.get("transcript"), rb.get("transcript")
    if not (wa and wb) or (ta and tb and ta["language"] != tb["language"]):
        return {"available": False, "similarity": None, "status": "unavailable", "pairs": [], "ops": []}
    sm = difflib.SequenceMatcher(a=[w["norm"] for w in wa], b=[w["norm"] for w in wb], autojunk=False)
    sim = sm.ratio()
    pairs, ops = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append({"op": tag, "a": " ".join(w["text"] for w in wa[i1:i2]), "b": " ".join(w["text"] for w in wb[j1:j2])})
        if tag == "equal":
            for x, y in zip(wa[i1:i2], wb[j1:j2]):
                pairs.append({
                    "word": y["text"],
                    "a_dur_ms": round((x["end"] - x["start"]) * 1000),
                    "b_dur_ms": round((y["end"] - y["start"]) * 1000),
                    "a_db": x.get("rel_db"), "b_db": y.get("rel_db"),
                    "a_st": x.get("rel_st"), "b_st": y.get("rel_st"),
                    "a_gap": x.get("gap_before", 0.0), "b_gap": y.get("gap_before", 0.0),
                    "a_contour": x.get("contour"), "b_contour": y.get("contour"),
                })
    status = "same" if sim >= SAME_TEXT else ("partial" if sim >= PARTIAL_TEXT else "different")
    return {"available": True, "similarity": round(sim, 3), "status": status, "pairs": pairs, "ops": ops}


def delivery_notes(al: dict) -> list[str]:
    """Plain-language notes on how the shared words were delivered differently."""
    pairs = al["pairs"]
    if len(pairs) < 3:
        return []
    notes = []
    da = np.array([p["a_dur_ms"] for p in pairs], float)
    db = np.array([p["b_dur_ms"] for p in pairs], float)
    ratio = float(np.median(db / np.maximum(da, 1)))
    if abs(ratio - 1) >= 0.1:
        notes.append(f"Shared words last **{abs(ratio - 1) * 100:.0f}% {'longer' if ratio > 1 else 'shorter'}** in B "
                     f"(median per-word duration ratio {ratio:.2f}) — B {'stretches' if ratio > 1 else 'clips'} its syllables.")
    else:
        notes.append("Per-word durations are similar in both recordings.")
    stretched = sorted(pairs, key=lambda p: (p["b_dur_ms"] - p["a_dur_ms"]) / max(p["a_dur_ms"], 1), reverse=True)
    top = [p for p in stretched[:3] if p["b_dur_ms"] > p["a_dur_ms"] * 1.35]
    if top:
        notes.append("Most lengthened in B: " + ", ".join(f"“{p['word']}” ({p['a_dur_ms']}→{p['b_dur_ms']} ms)" for p in top) + ".")
    emph_b = [p["word"] for p in pairs if (p["b_db"] or 0) >= 3 and (p["a_db"] or 0) < 3]
    emph_a = [p["word"] for p in pairs if (p["a_db"] or 0) >= 3 and (p["b_db"] or 0) < 3]
    if emph_b:
        notes.append("Stressed only in B (≥ 3 dB above its median word): " + ", ".join(f"“{w}”" for w in emph_b[:6]) + ".")
    if emph_a:
        notes.append("Stressed only in A: " + ", ".join(f"“{w}”" for w in emph_a[:6]) + ".")
    st_pairs = [(p["word"], p["b_st"] - p["a_st"]) for p in pairs if p["a_st"] is not None and p["b_st"] is not None]
    if st_pairs:
        spread_a = np.std([p["a_st"] for p in pairs if p["a_st"] is not None])
        spread_b = np.std([p["b_st"] for p in pairs if p["b_st"] is not None])
        if abs(spread_b - spread_a) >= 0.7:
            notes.append(f"Word pitches spread **{'wider' if spread_b > spread_a else 'narrower'}** in B "
                         f"({spread_a:.1f} → {spread_b:.1f} st SD, each relative to its own speaker median) — "
                         f"{'more' if spread_b > spread_a else 'less'} melodic phrasing.")
        big = sorted(st_pairs, key=lambda t: -abs(t[1]))[:3]
        big = [t for t in big if abs(t[1]) >= 3]
        if big:
            notes.append("Biggest pitch shifts: " + ", ".join(f"“{w}” ({d:+.1f} st)" for w, d in big) + ".")
    ga = sum(1 for p in pairs if p["a_gap"] >= I.PAUSE_MARK_S)
    gb = sum(1 for p in pairs if p["b_gap"] >= I.PAUSE_MARK_S)
    if ga != gb:
        notes.append(f"Pauses before shared words: {ga} in A vs {gb} in B — "
                     f"B is {'more broken-up' if gb > ga else 'more continuous'}.")
    return notes


# --------------------------------------------------------------------------- summary
def summary(ra: dict, rb: dict) -> list[str]:
    rows = {r["feature"]: r for r in metric_rows(ra, rb)}
    lines = []

    def cmp(feature, more, less, thresh):
        r = rows[feature]
        if r["delta"] is None:
            return
        if abs(r["delta"]) < thresh:
            lines.append(f"**{feature}** is about the same ({fmt(r['a'], r['nd'], r['unit'])} vs {fmt(r['b'], r['nd'], r['unit'])}).")
        else:
            lines.append(f"B is **{more if r['delta'] > 0 else less}** — {feature.lower()} "
                         f"{fmt(r['a'], r['nd'], r['unit'])} → {fmt(r['b'], r['nd'], r['unit'])}.")

    cmp("Median pitch", "higher-pitched", "lower-pitched", 8)
    cmp("Pitch variation", "more melodic", "flatter in intonation", 0.5)
    cmp("Speaking rate", "faster", "slower", 10)
    cmp("Energy (RMS)", "louder", "quieter", 2)
    cmp("Pause ratio", "more paused", "more continuous", 5)
    cmp("Spectral centroid", "brighter in timbre", "darker in timbre", 150)
    ea, eb = ra.get("emotion") or {}, rb.get("emotion") or {}
    if ea.get("ok") and eb.get("ok"):
        if ea["top"] == eb["top"]:
            lines.append(f"The emotion model gives both the same top class, **{ea['top']}** "
                         f"({ea['confidence'] * 100:.0f}% vs {eb['confidence'] * 100:.0f}%).")
        else:
            lines.append(f"The emotion model's top class shifts from **{ea['top']}** ({ea['confidence'] * 100:.0f}%) "
                         f"to **{eb['top']}** ({eb['confidence'] * 100:.0f}%).")
    return lines


def compare(ra: dict, rb: dict) -> dict:
    al = align(ra, rb)
    ea, eb = ra.get("emotion") or {}, rb.get("emotion") or {}
    return {
        "a": {"id": ra["id"], "name": ra["name"], "created_at": ra["created_at"]},
        "b": {"id": rb["id"], "name": rb["name"], "created_at": rb["created_at"]},
        "checks": checks(ra, rb),
        "metrics": metric_rows(ra, rb),
        "emotion": {
            "a": ea.get("probs") if ea.get("ok") else None,
            "b": eb.get("probs") if eb.get("ok") else None,
            "model": ea.get("model") or eb.get("model"),
        },
        "transcripts": {
            "a": (ra.get("transcript") or {}).get("text"),
            "b": (rb.get("transcript") or {}).get("text"),
        },
        "alignment": al,
        "delivery_notes": delivery_notes(al) if al["available"] and al["status"] != "different" else [],
        "summary": summary(ra, rb),
    }


# --------------------------------------------------------------------------- exports
def report_markdown(cmp: dict, ra: dict, rb: dict) -> str:
    L = ["# Velora comparison report",
         "",
         f"- **A:** {ra['name']} ({ra['created_at'].replace('T', ' ')}, {ra['stats']['duration']:.1f}s)",
         f"- **B:** {rb['name']} ({rb['created_at'].replace('T', ' ')}, {rb['stats']['duration']:.1f}s)",
         ""]
    if cmp["checks"]:
        L += ["## Compatibility notes", ""] + [f"- *{lvl}*: {msg}" for lvl, msg in cmp["checks"]] + [""]
    L += ["## Summary (rule-based interpretation)", ""] + [f"- {s.replace('**', '')}" for s in cmp["summary"]] + [""]
    L += ["## Transcripts (model output · Whisper)", "",
          f"**A:** {cmp['transcripts']['a'] or '_not available_'}", "",
          f"**B:** {cmp['transcripts']['b'] or '_not available_'}", ""]
    al = cmp["alignment"]
    if al["available"]:
        L += [f"Word-sequence similarity: {al['similarity'] * 100:.0f}% ({al['status']} wording)", ""]
        if cmp["delivery_notes"]:
            L += ["## Same words, different delivery (measured)", ""] + [f"- {n.replace('**', '')}" for n in cmp["delivery_notes"]] + [""]
        if al["pairs"]:
            L += ["| Word | A ms | B ms | A dB | B dB | A st | B st |", "|---|---:|---:|---:|---:|---:|---:|"]
            for p in al["pairs"]:
                L.append(f"| {p['word']} | {p['a_dur_ms']} | {p['b_dur_ms']} | {fmt(p['a_db'], 1)} | {fmt(p['b_db'], 1)} | "
                         f"{fmt(p['a_st'], 1)} | {fmt(p['b_st'], 1)} |")
            L += ["", "_dB and st are relative to each recording's own median word loudness / median pitch._", ""]
    L += ["## Measured acoustic features", "", "| Feature | A | B | Δ (B − A) |", "|---|---:|---:|---:|"]
    for r in cmp["metrics"]:
        pct = f" ({r['pct']:+.0f}%)" if r["pct"] is not None else ""
        L.append(f"| {r['feature']} | {fmt(r['a'], r['nd'], r['unit'])} | {fmt(r['b'], r['nd'], r['unit'])} | "
                 f"{fmt(r['delta'], r['nd'], r['unit'], sign=True)}{pct} |")
    L += ["", f"## Emotion model output ({cmp['emotion']['model'] or 'not available'})", "",
          "| Class | A | B |", "|---|---:|---:|"]
    for e in E.EMOTIONS:
        a = cmp["emotion"]["a"]
        b = cmp["emotion"]["b"]
        L.append(f"| {e} | {f'{a[e] * 100:.1f}%' if a else '—'} | {f'{b[e] * 100:.1f}%' if b else '—'} |")
    L += ["", "---", "_Generated locally by Velora. Emotion probabilities are raw outputs of a model trained on acted "
          "English speech; they are not a psychological assessment._"]
    return "\n".join(L)


def word_csv(cmp: dict) -> str:
    buf = io.StringIO()
    pairs = cmp["alignment"]["pairs"]
    if pairs:
        w = csv.DictWriter(buf, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    return buf.getvalue()


def report_bundle(cmp: dict, ra: dict, rb: dict, wav_a: bytes | None, wav_b: bytes | None) -> bytes:
    """ZIP with the Markdown report, full JSON for both analyses + comparison, word CSV and both WAVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("comparison_report.md", report_markdown(cmp, ra, rb))
        z.writestr("comparison.json", json.dumps({"comparison": cmp, "analysis_a": ra, "analysis_b": rb},
                                                 indent=2, ensure_ascii=False))
        if cmp["alignment"]["pairs"]:
            z.writestr("word_delivery.csv", word_csv(cmp))
        if wav_a:
            z.writestr("A_" + re.sub(r"[^\w.-]", "_", ra["name"]).removesuffix(".wav") + ".wav", wav_a)
        if wav_b:
            z.writestr("B_" + re.sub(r"[^\w.-]", "_", rb["name"]).removesuffix(".wav") + ".wav", wav_b)
    return buf.getvalue()
