"""Downloadable exports for one analysis: TXT transcript, JSON, CSV metrics/tracks, PDF report."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from pathlib import Path

from . import emotion as E
from . import interpret as I

FEATURES = [
    # key, label, unit, description
    ("duration", "Duration", "s", "Length of the analysed audio"),
    ("speech_duration", "Speech time", "s", "Frames above the adaptive speech-activity threshold"),
    ("speech_ratio", "Speech ratio", "", "Speech time ÷ duration"),
    ("voiced_ratio", "Voiced ratio", "", "Share of frames with a detected pitch"),
    ("pitch_median", "Pitch median", "Hz", "Median F0 over voiced frames (pYIN, 65–500 Hz)"),
    ("pitch_mean", "Pitch mean", "Hz", "Mean F0 over voiced frames"),
    ("pitch_std", "Pitch SD", "Hz", "Standard deviation of F0"),
    ("pitch_min", "Pitch 5th pct", "Hz", "Robust lower bound of the pitch range"),
    ("pitch_max", "Pitch 95th pct", "Hz", "Robust upper bound of the pitch range"),
    ("pitch_range", "Pitch range", "Hz", "95th − 5th percentile"),
    ("pitch_var_st", "Pitch variation", "st", "SD of pitch in semitones (intonation)"),
    ("energy_db", "Energy (RMS)", "dBFS", "Mean RMS level of speech frames"),
    ("energy_rms", "Energy (RMS, linear)", "", "Mean linear RMS amplitude of speech frames"),
    ("energy_std_db", "Energy variation", "dB", "SD of RMS level across speech frames"),
    ("centroid", "Spectral centroid", "Hz", "Brightness of timbre"),
    ("rolloff", "Spectral rolloff (85%)", "Hz", "Frequency below which 85% of energy lies"),
    ("bandwidth", "Spectral bandwidth", "Hz", "Spread of the spectrum around the centroid"),
    ("flatness", "Spectral flatness", "", "0 = tonal, 1 = noise-like"),
    ("zcr", "Zero-crossing rate", "", "Fraction of sign changes per frame (noisiness)"),
    ("syllables", "Syllable nuclei", "", "Energy peaks on voiced speech"),
    ("syllable_rate", "Articulation rate", "syll/s", "Syllables per second of speech time"),
    ("wpm", "Speaking rate (acoustic)", "wpm", "Syllables over speaking span ÷ 1.45 syllables/word"),
    ("pause_count", "Pauses", "", "Silent gaps ≥ 250 ms inside the speaking span"),
    ("pause_mean", "Mean pause", "s", "Average pause length"),
    ("pause_ratio", "Pause ratio", "", "Pause time ÷ speaking span"),
    ("jitter_pct", "Frame jitter", "%", "Frame-to-frame period change (approximation, not clinical jitter)"),
    ("shimmer_db", "Frame shimmer", "dB", "Frame-to-frame level change on voiced frames (approximation)"),
]
QUALITY = [
    ("peak_dbfs", "Peak level", "dBFS"),
    ("speech_level_db", "Speech level (p90)", "dBFS"),
    ("noise_floor_db", "Noise floor (p10)", "dBFS"),
    ("snr_db", "Estimated SNR (capped at 60)", "dB"),
    ("clipping_pct", "Clipped samples", "%"),
]


def base_name(result: dict) -> str:
    stem = re.sub(r"[^\w-]+", "_", os.path.splitext(result["name"])[0]).strip("_") or "recording"
    return f"velora_{stem}_{result['id']}"


# --------------------------------------------------------------------------- text / data
def analysis_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)


def transcript_txt(result: dict) -> str:
    tr = result.get("transcript")
    lines = [f"Velora transcript — {result['name']}", f"Analysed: {result['created_at'].replace('T', ' ')}", ""]
    if not tr:
        lines.append(f"No transcript available: {result.get('transcript_error') or 'speech-to-text did not run'}.")
        return "\n".join(lines) + "\n"
    lines += [f"Model: faster-whisper {tr['model']} · language {tr['language']} (p={tr['language_prob']:.2f})", "",
              "== Full text ==", tr["text"], "", "== Segments ==" ]
    for s in tr["segments"]:
        lines.append(f"[{_ts(s['start'])} → {_ts(s['end'])}] {s['text']}")
    expr = I.expressive_markdown(result)
    if expr:
        lines += ["", "== Expressive transcript (measured prosody markup) ==",
                  "**word** = louder than median word (≥ +3 dB) · _word_ = softer (≤ −5 dB) · ↗/↘ = pitch rise/fall · "
                  "… / [pause Xs] = measured pause", "", expr]
    return "\n".join(lines) + "\n"


def _ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def metrics_rows(result: dict) -> list[dict]:
    """All clip-level numbers with their source: measured, model output or derived."""
    s, rows = result["stats"], []
    for key, label, unit, desc in FEATURES:
        rows.append({"section": "acoustic", "metric": label, "value": s[key], "unit": unit,
                     "source": "measured (librosa/NumPy)", "description": desc})
    wpm, src = I.speaking_rate(s, result.get("transcript"))
    rows.append({"section": "acoustic", "metric": "Speaking rate (reported)", "value": wpm, "unit": "wpm",
                 "source": "measured", "description": src})
    for key, label, unit in QUALITY:
        rows.append({"section": "recording quality", "metric": label, "value": result["quality"][key], "unit": unit,
                     "source": "measured", "description": ""})
    for k, v in result["profile"].items():
        rows.append({"section": "voice profile (0–1)", "metric": k, "value": v, "unit": "", "source": "measured, normalised",
                     "description": "Scaled against fixed reference ranges"})
    emo = result.get("emotion") or {}
    if emo.get("ok"):
        for e, p in emo["probs"].items():
            rows.append({"section": "emotion model", "metric": f"P({e})", "value": p, "unit": "probability",
                         "source": f"model output ({emo['model']})", "description": emo["scope"]})
    else:
        rows.append({"section": "emotion model", "metric": "status", "value": "", "unit": "",
                     "source": "not available", "description": emo.get("error", "")})
    return rows


def metrics_csv(result: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["section", "metric", "value", "unit", "source", "description"])
    w.writeheader()
    w.writerows(metrics_rows(result))
    return buf.getvalue()


def tracks_csv(result: dict) -> str:
    """Frame-level measurements (every ~20 ms)."""
    t = result["tracks"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time_s", "f0_hz", "rms_dbfs", "spectral_centroid_hz", "speech_active"])
    for row in zip(t["t"], t["f0"], t["rms_db"], t["centroid"], t["active"]):
        w.writerow(["" if v is None else v for v in row])
    return buf.getvalue()


def timeline_csv(result: dict) -> str | None:
    emo = result.get("emotion") or {}
    if not emo.get("ok"):
        return None
    tl = emo["timeline"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start_s", "end_s", "contains_speech", *E.EMOTIONS])
    for i in range(len(tl["t"])):
        w.writerow([tl["start"][i], tl["end"][i], tl["speech"][i], *[tl[e][i] for e in E.EMOTIONS]])
    return buf.getvalue()


# --------------------------------------------------------------------------- PDF
_FONT_CANDIDATES = [
    # (regular, bold, italic) — first set that exists is used
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\segoeuii.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\ariali.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
]
_SYMBOL_FONTS = [r"C:\Windows\Fonts\seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_LAV, _INK, _MUTED, _LINE = (91, 71, 214), (31, 27, 58), (107, 103, 133), (225, 221, 236)


def _hex(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def pdf_available() -> bool:
    try:
        import fpdf  # noqa: F401
        return True
    except Exception:
        return False


def pdf_report(result: dict) -> bytes:
    """A printable A4 report. Requires fpdf2; uses a system Unicode font when one exists."""
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 16, 16)
    font = "Helvetica"
    unicode_ok = False
    for reg, bold, ital in _FONT_CANDIDATES:
        if Path(reg).exists() and Path(bold).exists() and Path(ital).exists():
            pdf.add_font("Body", "", reg)
            pdf.add_font("Body", "B", bold)
            pdf.add_font("Body", "I", ital)
            font, unicode_ok = "Body", True
            sym = next((f for f in _SYMBOL_FONTS if Path(f).exists() and f != reg), None)
            if sym:  # arrows such as ↘ are missing from some text fonts
                pdf.add_font("Sym", "", sym)
                pdf.set_fallback_fonts(["Sym"], exact_match=False)
            break

    def txt(s) -> str:
        s = str(s)
        if unicode_ok:
            return s
        return (s.replace("—", "-").replace("–", "-").replace("→", "->").replace("…", "...")
                .replace("“", '"').replace("”", '"').replace("’", "'").replace("≥", ">=").replace("≤", "<=")
                .replace("÷", "/").replace("↗", "^").replace("↘", "v").replace("−", "-")
                .encode("latin-1", "replace").decode("latin-1"))

    def h1(s):
        pdf.set_font(font, "B", 20); pdf.set_text_color(*_INK); pdf.cell(0, 10, txt(s), new_x="LMARGIN", new_y="NEXT")

    def h2(s, tag=None):
        pdf.ln(3)
        pdf.set_font(font, "B", 13); pdf.set_text_color(*_INK); pdf.cell(0, 8, txt(s), new_x="LMARGIN", new_y="NEXT")
        if tag:
            pdf.set_font(font, "B", 7.5); pdf.set_text_color(*_LAV)
            pdf.cell(0, 4.5, txt(tag.upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_LINE); pdf.line(16, pdf.get_y() + 1, 194, pdf.get_y() + 1); pdf.ln(3)

    def body(s, size=10, color=_INK):
        pdf.set_font(font, "", size); pdf.set_text_color(*color)
        pdf.multi_cell(0, 5.2, txt(s), new_x="LMARGIN", new_y="NEXT")

    s, emo = result["stats"], result.get("emotion") or {}
    tr = result.get("transcript")
    wpm, wpm_src = I.speaking_rate(s, tr)

    # header
    pdf.add_page()
    pdf.set_fill_color(*_LAV); pdf.rect(16, 14, 4, 12, "F")
    pdf.set_x(23); pdf.set_font(font, "B", 9); pdf.set_text_color(*_LAV)
    pdf.cell(0, 5, "VELORA · VOICE REPORT" if unicode_ok else "VELORA - VOICE REPORT", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(23); h1(result["name"])
    body(f"Analysed {result['created_at'].replace('T', ' ')} · source: {result['source']} · duration {s['duration']:.1f}s "
         f"(speech {s['speech_duration']:.1f}s)" + (" · first 3 min only" if result.get("truncated") else ""), 9, _MUTED)

    # model outputs
    h2("Emotion recognition", "Model output · " + E.MODEL_ID)
    if emo.get("ok"):
        body(f"Top class: {emo['top']} ({emo['confidence'] * 100:.0f}%) · scope: {emo['scope']}", 10.5)
        pdf.ln(1)
        for e in sorted(E.EMOTIONS, key=lambda k: -emo["probs"][k]):
            p = emo["probs"][e]
            y = pdf.get_y()
            pdf.set_font(font, "", 9.5); pdf.set_text_color(*_INK)
            pdf.cell(24, 6, txt(e))
            pdf.set_fill_color(239, 236, 244); pdf.rect(42, y + 1.6, 120, 2.8, "F")
            pdf.set_fill_color(*_hex(E.EMOTION_COLORS[e])); pdf.rect(42, y + 1.6, max(120 * p, 0.8), 2.8, "F")
            pdf.set_x(166); pdf.cell(0, 6, f"{p * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")
    else:
        body(f"No emotion prediction: {emo.get('error', 'model did not run')}.", 10, _MUTED)

    h2("Transcript", f"Model output · faster-whisper {tr['model']} · {tr['language']}" if tr else "Model output")
    body(tr["text"] if tr else f"No transcript: {result.get('transcript_error') or 'not available'}.",
         10.5 if tr else 10, _INK if tr else _MUTED)

    # measured
    h2("Key measurements", "Measured · librosa / NumPy")
    kv = [("Median pitch", f"{s['pitch_median']:.0f} Hz" if s["pitch_median"] else "not detectable"),
          ("Pitch variation", f"{s['pitch_var_st']:.2f} semitones SD"),
          ("Energy (RMS)", f"{s['energy_db']:.1f} dBFS (variation {s['energy_std_db']:.1f} dB)"),
          ("Spectral centroid", f"{s['centroid']:.0f} Hz"),
          ("Speaking rate", f"{wpm:.0f} wpm ({wpm_src})" if wpm else "not measurable"),
          ("Pauses", f"{s['pause_count']} (mean {s['pause_mean']:.2f}s, {s['pause_ratio'] * 100:.0f}% of speaking span)"),
          ("Recording quality", f"peak {result['quality']['peak_dbfs']} dBFS · SNR ≈ {result['quality']['snr_db']} dB · "
                                f"clipping {result['quality']['clipping_pct']}%")]
    for k, v in kv:
        pdf.set_font(font, "B", 9.5); pdf.set_text_color(*_INK); pdf.cell(44, 6, txt(k))
        pdf.set_font(font, "", 9.5); pdf.multi_cell(0, 6, txt(v), new_x="LMARGIN", new_y="NEXT")

    _pdf_contour(pdf, result, font, txt)

    # interpretation
    h2("Interpretation", "Rule-based text from the measurements and model output")
    body(result["interpretation"].replace("**", ""))
    if tr:
        pdf.ln(2)
        pdf.set_font(font, "B", 10); pdf.set_text_color(*_INK)
        pdf.cell(0, 6, "Expressive transcript", new_x="LMARGIN", new_y="NEXT")
        # fpdf2 markdown: **bold**, __italic__
        expr = re.sub(r"(?<![\w*])_([^_\s][^_]*?)_(?!\w)", r"__\1__", I.expressive_markdown(result))
        pdf.set_font(font, "", 9.5); pdf.set_text_color(*_INK)
        pdf.multi_cell(0, 5.2, txt(expr), new_x="LMARGIN", new_y="NEXT", markdown=True)
        body("bold = louder than your median word (≥ +3 dB) · italic = softer (≤ −5 dB) · ↗ ↘ = pitch rise / fall "
             "within the word · … / [pause] = measured pause", 8, _MUTED)

    # full table
    pdf.add_page()
    h2("All measured features", "Measured · librosa / NumPy")
    pdf.set_font(font, "B", 8.5); pdf.set_text_color(*_MUTED)
    pdf.cell(52, 6, "Feature"); pdf.cell(26, 6, "Value", align="R"); pdf.cell(14, 6, "  Unit"); pdf.cell(0, 6, "Meaning",
                                                                                                         new_x="LMARGIN", new_y="NEXT")
    for i, (key, label, unit, desc) in enumerate(FEATURES):
        if i % 2 == 0:
            pdf.set_fill_color(247, 245, 254); pdf.rect(16, pdf.get_y(), 178, 5.6, "F")
        pdf.set_font(font, "", 8.5); pdf.set_text_color(*_INK)
        pdf.cell(52, 5.6, txt(label)); pdf.cell(26, 5.6, txt(s[key]), align="R"); pdf.cell(14, 5.6, txt("  " + unit))
        pdf.set_text_color(*_MUTED); pdf.cell(0, 5.6, txt(desc)[:70], new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    body("The emotion model was trained on acted English speech (IEMOCAP) with four classes and can be wrong. "
         "Interpretation text is generated by fixed rules. This report is not a psychological or medical assessment. "
         "Generated locally by Velora.", 8, _MUTED)
    return bytes(pdf.output())


def _pdf_contour(pdf, result: dict, font: str, txt) -> None:
    """Small pitch (top) and energy (bottom) line plots drawn with vector primitives."""
    t = result["tracks"]
    times = t["t"]
    if not times:
        return
    if pdf.get_y() > 220:
        pdf.add_page()
    pdf.ln(3)
    x0, w, h = 30, 164, 22
    dur = times[-1] or 1.0
    for label, key, color, unit in (("Pitch", "f0", _LAV, "Hz"), ("Energy", "rms_db", (42, 157, 127), "dBFS")):
        vals = [v for v in t[key] if v is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if key == "rms_db":
            lo = max(lo, hi - 50)
        span = (hi - lo) or 1.0
        y0 = pdf.get_y()
        pdf.set_font(font, "", 7.5); pdf.set_text_color(*_MUTED)
        pdf.set_xy(16, y0); pdf.cell(14, 4, txt(f"{hi:.0f}"))
        pdf.set_xy(16, y0 + h - 4); pdf.cell(14, 4, txt(f"{lo:.0f}"))
        pdf.set_xy(16, y0 + h / 2 - 2); pdf.cell(14, 4, txt(unit))
        pdf.set_draw_color(*_LINE); pdf.set_line_width(0.2); pdf.rect(x0, y0, w, h)
        pdf.set_draw_color(*color); pdf.set_line_width(0.45)
        prev = None
        for ti, v in zip(times, t[key]):
            if v is None or (key == "rms_db" and v < lo):
                prev = None
                continue
            pt = (x0 + ti / dur * w, y0 + h - (v - lo) / span * h)
            if prev is not None:
                pdf.line(prev[0], prev[1], pt[0], pt[1])
            prev = pt
        pdf.set_xy(x0 + 2, y0 + 1); pdf.set_font(font, "B", 7.5); pdf.set_text_color(*_INK); pdf.cell(30, 4, txt(label))
        pdf.set_y(y0 + h + 2)
    pdf.set_font(font, "", 7.5); pdf.set_text_color(*_MUTED)
    pdf.set_x(x0); pdf.cell(w / 2, 4, "0 s"); pdf.cell(w / 2, 4, txt(f"{dur:.1f} s"), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.2)


if __name__ == "__main__":  # quick manual check: python -m velora.exports result.json
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(base_name(res) + ".pdf").write_bytes(pdf_report(res))
