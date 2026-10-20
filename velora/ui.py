"""Shared Streamlit UI: theme CSS, HTML components, settings and analysis runner."""
from __future__ import annotations

import html

import streamlit as st

from . import audio as A
from . import emotion as E
from . import pipeline as P
from . import storage as S
from . import transcribe as T

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Nunito+Sans:opsz,wght@6..12,400;6..12,600;6..12,700;6..12,800&display=swap');
:root{
  --ink:#1F1B3A; --muted:#6B6785; --faint:#9A96B0; --line:#ECE8F4;
  --lav:#5B47D6; --lav-deep:#4B3BC4; --lav-2:#8B7CF0; --lav-soft:#F2EFFD; --lav-wash:#F7F5FE;
  --ivory:#FBFAF6; --ivory-2:#F6F3EC; --card:#FFFFFF; --amber-soft:#FDF6EC; --amber:#B86E14;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, label {
  font-family:'Nunito Sans','Segoe UI',sans-serif !important;
}
.stApp { background: var(--ivory); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(251,250,246,.85); backdrop-filter: blur(6px); }
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1380px; }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; }
[data-testid="stSidebar"] .vl-pill { display:flex; width: fit-content; margin: 4px 0; }
h1, h2, h3, h4, .vl-serif { font-family:'Source Serif 4', Georgia, serif !important; color: var(--ink); letter-spacing:-.01em; }
h3 { font-weight:600 !important; }
p, li { color: var(--ink); }
[data-testid="stCaptionContainer"], .vl-muted { color: var(--muted) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--ivory-2); border-right:1px solid var(--line); }
[data-testid="stSidebarNav"] a { border-radius: 12px; padding: .45rem .75rem; margin: 2px 0; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--lav-soft); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--lav) !important; font-weight:700; }
[data-testid="stSidebarNav"] span { color: var(--ink); font-size: .95rem; }

/* Cards: any container keyed "card…" or "hero…" */
[class*="st-key-card"], [class*="st-key-hero"] {
  background: var(--card); border:1px solid var(--line); border-radius: 20px; padding: 24px 26px;
  box-shadow: 0 1px 2px rgba(31,27,58,.04), 0 10px 28px rgba(91,71,214,.05);
}
[class*="st-key-hero"] { background: linear-gradient(135deg,#F3F0FE 0%,#FFFFFF 55%,#FCF9F2 100%); }
[class*="st-key-soft"] { background: var(--lav-wash); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
  border-radius: 999px !important; font-weight: 700; padding: .5rem 1.2rem;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--lav-deep); border: 1px solid var(--lav-deep); box-shadow: 0 6px 16px rgba(75,59,196,.25);
}
.stButton > button[kind="primary"]:hover { background: var(--lav); border-color: var(--lav); }
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {
  border: 1.5px solid #CFC7F5; color: var(--lav-deep); background: white;
}
.stButton > button[kind="secondary"]:hover { border-color: var(--lav); color: var(--lav); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight:600; color: var(--muted); padding: .4rem .1rem; }
.stTabs [aria-selected="true"] { color: var(--lav) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--lav); height: 2.5px; }

/* Inputs */
[data-testid="stFileUploaderDropzone"] { background: var(--lav-wash); border: 1.5px dashed #CFC7F5; border-radius: 16px; }
[data-testid="stAudioInput"] > div { border-radius: 16px; border-color: #CFC7F5; background: var(--lav-wash); }
[data-testid="stMetric"] { background: var(--card); border:1px solid var(--line); border-radius: 16px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { font-family:'Source Serif 4', Georgia, serif; color: var(--ink); }
[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--line); background: var(--card); }

/* Custom components */
.vl-eyebrow { text-transform: uppercase; letter-spacing:.12em; font-size:.74rem; font-weight:800; color: var(--muted); margin-bottom:.3rem; }
.vl-hero-title { font-family:'Source Serif 4',Georgia,serif; font-size: 2.35rem; line-height:1.15; font-weight:600; margin:.1rem 0 .6rem; color: var(--ink); }
.vl-hero-sub { color: var(--muted); font-size: 1.03rem; max-width: 760px; line-height:1.6; }
.vl-h { font-family:'Source Serif 4',Georgia,serif; font-size:1.35rem; font-weight:600; margin:0; color:var(--ink); }
.vl-sub { color: var(--muted); font-size:.92rem; margin:.15rem 0 .9rem; }
.vl-emo { display:flex; gap:16px; align-items:center; padding:18px 20px; border-radius:16px;
  background: linear-gradient(120deg,#FDF6EC 0%,#F6F2FE 100%); border:1px solid var(--line); margin-bottom: 18px; }
.vl-emo .face { width:58px; height:58px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:30px; background:white; box-shadow: 0 4px 14px rgba(31,27,58,.08); }
.vl-emo .name { font-family:'Source Serif 4',Georgia,serif; font-size:1.7rem; font-weight:600; line-height:1.1; color: var(--ink); }
.vl-emo .conf { color: var(--muted); font-size:.92rem; }
.vl-bar { display:grid; grid-template-columns: 74px 1fr 46px; align-items:center; gap:12px; margin: 10px 0; font-size:.92rem; }
.vl-bar .lbl { color: var(--ink); display:flex; align-items:center; gap:7px; }
.vl-bar .lbl i { width:9px; height:9px; border-radius:50%; display:inline-block; }
.vl-bar .track { height: 9px; border-radius: 99px; background: #EFECF4; overflow:hidden; }
.vl-bar .fill { height:100%; border-radius: 99px; }
.vl-bar .pct { text-align:right; color: var(--ink); font-variant-numeric: tabular-nums; font-weight:600; }
.vl-insight { display:flex; gap:14px; align-items:flex-start; padding:14px 16px; border:1px solid var(--line);
  border-radius:16px; margin-bottom:10px; background:white; }
.vl-insight .ic { width:40px; height:40px; flex:0 0 40px; border-radius:50%; background: var(--lav-soft); color: var(--lav);
  display:flex; align-items:center; justify-content:center; font-size:18px; }
.vl-insight .body { flex:1; }
.vl-insight .top { display:flex; justify-content:space-between; font-weight:700; color: var(--ink); gap: 8px; }
.vl-insight .note { color: var(--muted); font-size:.85rem; margin-top:3px; }
.vl-quote { background: var(--lav-wash); border:1px solid var(--line); border-radius:16px; padding:16px 18px; line-height:1.65; }
.vl-note { background: var(--amber-soft); border:1px solid #F3E3CB; border-radius: 14px; padding: 12px 14px;
  font-size:.83rem; color:#6E5A3E; display:flex; gap:10px; }
.vl-pill { display:inline-flex; align-items:center; gap:6px; border-radius:99px; padding:4px 12px; font-size:.8rem; font-weight:700;
  border:1px solid var(--line); background:white; color: var(--ink); margin: 2px 6px 2px 0; }
.vl-pill i { width:8px; height:8px; border-radius:50%; display:inline-block; }
.vl-src { display:inline-block; font-size:.7rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
  padding:3px 9px; border-radius:99px; margin-bottom:8px; }
.vl-src.model { background:#EAE6FD; color:#4B3BC4; }
.vl-src.measured { background:#E3F3EE; color:#1F7A62; }
.vl-src.derived { background:#FBEFDF; color:#99600F; }
.vl-seg { margin: 0 0 .9rem; line-height: 2.0; }
.vl-ts { font-size:.75rem; color: var(--faint); font-variant-numeric: tabular-nums; margin-right:8px; }
.vl-chip { display:inline-flex; align-items:center; gap:5px; font-size:.72rem; font-weight:700; padding:2px 8px;
  border-radius:99px; background:white; border:1px solid var(--line); margin-right:6px; vertical-align: 1px; }
.vl-chip i { width:7px; height:7px; border-radius:50%; display:inline-block; }
.vl-loud { font-weight: 800; color: var(--ink); }
.vl-soft { color: var(--faint); font-style: italic; }
.vl-high { text-decoration: underline; text-decoration-color: #B8ACF6; text-decoration-thickness: 2px; text-underline-offset: 4px; }
.vl-pause { color: var(--lav-2); font-weight:700; font-size:.85rem; }
.vl-legend { font-size:.8rem; color: var(--muted); margin-top:.4rem; }
.vl-empty { text-align:center; padding: 44px 20px; color: var(--muted); }
.vl-empty .big { font-family:'Source Serif 4',Georgia,serif; font-size:1.3rem; color: var(--ink); margin-bottom:.3rem; }
.vl-steps { display:grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.vl-step { background:white; border:1px solid var(--line); border-radius:16px; padding:14px 16px; }
.vl-step b { display:block; margin-bottom:4px; }
.vl-step span { color: var(--muted); font-size:.86rem; }
.vl-brand { display:flex; gap:10px; align-items:center; margin: 4px 0 6px; }
.vl-brand .t { font-family:'Source Serif 4',Georgia,serif; font-size:1.45rem; font-weight:600; color: var(--ink); line-height:1; }
.vl-brand .s { font-size:.78rem; color: var(--muted); }
.vl-foot { color: var(--muted); font-size:.78rem; text-align:center; margin-top: 1rem; }
.vl-delta-up { color:#1F7A62; font-weight:700; } .vl-delta-down { color:#B4385C; font-weight:700; }

/* ---- polish ---- */
.vl-titlebar { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.vl-chip-top { display:inline-flex; align-items:center; gap:7px; border:1px solid var(--line); background:white;
  border-radius:99px; padding:6px 14px; font-size:.8rem; font-weight:700; color:var(--ink);
  box-shadow: 0 2px 10px rgba(91,71,214,.06); white-space:nowrap; }
.vl-chip-top i { width:8px; height:8px; border-radius:50%; display:inline-block; }
.vl-status { background:white; border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin-top:8px; }
.vl-status-h { font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); margin-bottom:6px; }
.vl-status-row { display:grid; grid-template-columns: 10px 1fr; column-gap:8px; align-items:center; margin:6px 0; font-size:.82rem; }
.vl-status-row i { width:8px; height:8px; border-radius:50%; display:inline-block; }
.vl-status-row b { color:var(--ink); font-weight:700; }
.vl-status-row span { grid-column:2; color:var(--muted); font-size:.76rem; }
.vl-foot svg { margin: 18px auto 6px; display:block; }
.vl-iconcircle { width:54px; height:54px; border-radius:50%; background:var(--lav-soft); display:flex; align-items:center;
  justify-content:center; flex:0 0 54px; }
.vl-inputhead { display:flex; gap:14px; align-items:center; margin-bottom:10px; }
.vl-inputhead b { display:block; font-size:1rem; color:var(--ink); }
.vl-inputhead span { color:var(--muted); font-size:.83rem; }
.vl-or { display:flex; flex-direction:column; align-items:center; gap:8px; color:var(--faint); font-weight:800; font-size:.8rem; }
.vl-or::before, .vl-or::after { content:""; width:1px; height:46px; background:var(--line); }
.vl-ready { display:flex; align-items:center; gap:12px; padding:12px 16px; border:1px solid var(--line); border-radius:16px;
  background:white; margin: 4px 0 10px; }
.vl-ready .ic { width:40px; height:40px; border-radius:12px; background:var(--lav-soft); display:flex; align-items:center;
  justify-content:center; color:var(--lav); font-size:18px; }
.vl-ready b { color:var(--ink); } .vl-ready span { color:var(--muted); font-size:.82rem; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow:hidden; border:1px solid var(--line); }
[data-testid="stAudio"] { border-radius: 999px; }
[data-testid="stSidebarHeader"] { padding-top: 1.4rem; padding-bottom: .4rem; }
[data-testid="stSidebarHeader"] img { height: 46px !important; }
[data-testid="stSidebarNavItems"] { padding-top: .6rem; }
[data-testid="stExpander"] summary p { font-weight:600; }
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stPills"] button[aria-checked="true"] {
  background: var(--lav-soft) !important; border-color:#CFC7F5 !important; color: var(--lav-deep) !important; }
h3 { margin-top: 1.4rem !important; }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] * { color: #FFFFFF !important; }
.stButton > button[kind="primary"]:disabled { background: #CFC7F5 !important; border-color:#CFC7F5 !important; box-shadow:none; }
.vl-stats { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:10px; }
.vl-stat { border:1px solid var(--line); border-radius:14px; padding:10px 14px; background:white; }
.vl-stat span { display:block; color:var(--muted); font-size:.78rem; font-weight:600; }
.vl-stat b { font-family:'Source Serif 4',Georgia,serif; font-size:1.3rem; font-weight:600; color:var(--ink); white-space:nowrap; }
</style>
"""

LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
<g stroke="#5B47D6" stroke-width="2.4" stroke-linecap="round">
<line x1="4" y1="14" x2="4" y2="20"/><line x1="9" y1="10" x2="9" y2="24"/><line x1="14" y1="5" x2="14" y2="29"/>
<line x1="19" y1="9" x2="19" y2="25"/><line x1="24" y1="12" x2="24" y2="22"/><line x1="29" y1="15" x2="29" y2="19"/>
</g></svg>"""

DEFAULTS = {
    "use_emotion": True,
    "use_transcript": True,
    "whisper_size": "base",
    "language": "auto",
    "save_history": True,
}


# --------------------------------------------------------------------------- setup
def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def settings() -> dict:
    return {k: st.session_state.get(k, v) for k, v in DEFAULTS.items()}


def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="vl-brand">{LOGO_SVG}<div><div class="t">Velora</div>'
            f'<div class="s">Your voice. Deeper insights.</div></div></div>',
            unsafe_allow_html=True,
        )


def sidebar_settings() -> None:
    with st.sidebar:
        with st.expander("⚙️ Engine settings"):
            st.toggle("Emotion model", key="use_emotion",
                      help=f"Pretrained classifier {E.MODEL_ID} (~380 MB, downloaded once).")
            st.toggle("Speech-to-text", key="use_transcript", help="faster-whisper on CPU (int8).")
            st.selectbox("Whisper model", T.WHISPER_SIZES, key="whisper_size",
                         help="base ≈ 145 MB (more accurate) · tiny ≈ 75 MB (faster)")
            st.selectbox("Language", ["auto", "en", "es", "fr", "de", "it", "pt", "hi", "ar", "zh", "ja"],
                         key="language")
            st.toggle("Save analyses to History", key="save_history")
        model_status_pills()
        st.markdown(
            '<div class="vl-foot"><svg width="150" height="34" viewBox="0 0 150 34" fill="none">'
            '<path d="M2 17 C 20 17, 24 6, 34 17 S 48 30, 58 17 S 70 2, 80 17 S 94 28, 104 17 S 120 10, 148 17" '
            'stroke="#B8ACF6" stroke-width="1.6" fill="none"/><path d="M2 19 C 26 19, 30 12, 42 19 S 62 26, 74 19 '
            'S 96 12, 110 19 S 134 22, 148 19" stroke="#DCD5FB" stroke-width="1.2" fill="none"/></svg><br>'
            'Analyze • Understand • Express<br><span style="opacity:.8">Velora v1.0 · runs 100% locally</span></div>',
            unsafe_allow_html=True,
        )


def model_status_pills() -> None:
    labels = {"loaded": ("#2A9D7F", "Ready"), "not loaded": ("#8B7CF0", "Loads on first use"),
              "error": ("#D6456E", "Failed — see About"), "missing": ("#9A96B0", "Not installed")}
    es, _ = E.model_status()
    ts, _ = T.transcriber_status()
    if ts == "available":
        ts = "not loaded"
    rows = []
    for name, s in (("Emotion model", es), ("Speech-to-text", ts)):
        c, txt = labels.get(s, ("#9A96B0", s))
        rows.append(f'<div class="vl-status-row"><i style="background:{c}"></i><b>{name}</b><span>{txt}</span></div>')
    st.markdown(f'<div class="vl-status"><div class="vl-status-h">AI engine</div>{"".join(rows)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- components
def header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<p class="vl-h">{html.escape(title)}</p>' +
                (f'<p class="vl-sub">{html.escape(subtitle)}</p>' if subtitle else ""), unsafe_allow_html=True)


def source_tag(kind: str, text: str | None = None) -> None:
    """kind: model | measured | derived — labels where the numbers come from."""
    default = {"model": "Model output", "measured": "Measured", "derived": "Interpretation"}[kind]
    st.markdown(f'<span class="vl-src {kind}">{html.escape(text or default)}</span>', unsafe_allow_html=True)


def _engine_chip() -> str:
    es, _ = E.model_status()
    ts, _ = T.transcriber_status()
    if es == "error" or ts == "error":
        dot, label = "#D6456E", "AI models · check About"
    elif es == "missing" and ts == "missing":
        dot, label = "#9A96B0", "Acoustic analysis only"
    else:
        dot, label = "#2A9D7F" if es == "loaded" else "#8B7CF0", "AI powered · runs locally"
    return f'<span class="vl-chip-top"><i style="background:{dot}"></i>{label}</span>'


def page_title(eyebrow: str, title: str, sub: str) -> None:
    st.markdown(
        f'<div class="vl-titlebar"><div class="vl-eyebrow">{html.escape(eyebrow)}</div>{_engine_chip()}</div>'
        f'<div class="vl-hero-title">{html.escape(title)}</div>'
        f'<div class="vl-hero-sub">{html.escape(sub)}</div>',
        unsafe_allow_html=True,
    )


def emotion_card(emo: dict, compact: bool = False) -> None:
    if not emo or not emo.get("ok"):
        reason = (emo or {}).get("error", "No emotion prediction available.")
        st.markdown(
            f'<div class="vl-note">⚠️<div><b>No emotion prediction.</b> {html.escape(reason)}<br>'
            f'Acoustic measurements and transcription are unaffected.</div></div>',
            unsafe_allow_html=True,
        )
        return
    top = emo["top"]
    st.markdown(
        f'<div class="vl-emo"><div class="face">{E.EMOTION_ICONS[top]}</div><div>'
        f'<div class="name">{top}</div><div class="conf">Model probability: {emo["confidence"] * 100:.0f}% · '
        f'{html.escape(emo["scope"])}</div></div></div>',
        unsafe_allow_html=True,
    )
    if not compact:
        prob_bars(emo["probs"])


def prob_bars(probs: dict) -> None:
    rows = []
    for e in sorted(E.EMOTIONS, key=lambda k: -probs[k]):
        p = probs[e]
        c = E.EMOTION_COLORS[e]
        rows.append(
            f'<div class="vl-bar"><div class="lbl"><i style="background:{c}"></i>{e}</div>'
            f'<div class="track"><div class="fill" style="width:{max(p * 100, 1.5):.1f}%;background:{c}"></div></div>'
            f'<div class="pct">{p * 100:.0f}%</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def stat_tiles(items: list[tuple[str, str]]) -> None:
    """Compact label/value tiles that wrap instead of truncating."""
    st.markdown('<div class="vl-stats">' + "".join(
        f'<div class="vl-stat"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>' for k, v in items
    ) + "</div>", unsafe_allow_html=True)


def insight_tiles(items: list[dict]) -> None:
    st.markdown("".join(
        f'<div class="vl-insight"><div class="ic">{it["icon"]}</div><div class="body">'
        f'<div class="top"><span>{html.escape(it["label"])}</span><span>{html.escape(it["value"])}</span></div>'
        f'<div class="note">{html.escape(it["note"])}</div></div></div>'
        for it in items
    ), unsafe_allow_html=True)


def md_bold(text: str) -> str:
    """Escape text and render **bold** spans."""
    parts = html.escape(text).split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def interpretation_block(result: dict) -> None:
    st.markdown(f'<div class="vl-quote">{md_bold(result["interpretation"])}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        '<div class="vl-note">⚠️<div>The interpretation is generated from rules applied to the measured features and '
        'the model output. The emotion model was trained on acted English speech (IEMOCAP) and can be wrong; this is '
        'not a psychological assessment.</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="vl-empty"><div class="big">{html.escape(title)}</div>{html.escape(text)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- analysis state
def run_analysis(data: bytes, name: str, source: str) -> dict | None:
    """Analyse audio with progress feedback; store as the current analysis."""
    cfg = settings()
    cache_key = (A.audio_hash(data), cfg["use_emotion"], cfg["use_transcript"], cfg["whisper_size"], cfg["language"])
    cache = st.session_state.setdefault("_analysis_cache", {})
    if cache_key in cache:
        cached = cache[cache_key]
        res = st.session_state.get("_results", {}).get(cached) or S.get(cached)
        if res:
            set_current(res)
            return res

    bar = st.progress(0.0, text="Starting analysis…")
    try:
        result, y, sr = P.analyze(
            data, name, source,
            use_emotion=cfg["use_emotion"],
            use_transcript=cfg["use_transcript"],
            whisper_size=cfg["whisper_size"],
            language=None if cfg["language"] == "auto" else cfg["language"],
            progress=lambda p, msg: bar.progress(min(p, 1.0), text=msg),
        )
    except A.AudioError as exc:
        bar.empty()
        st.error(f"**Couldn't analyze “{name}”.** {exc}", icon=":material/error:")
        return None
    except Exception as exc:  # keep the UI alive on unexpected errors
        bar.empty()
        st.error(f"**Analysis failed unexpectedly** ({type(exc).__name__}: {exc}). Your other analyses are "
                 "unaffected — try another file, or check the terminal for details.", icon=":material/error:")
        return None
    bar.empty()
    wav = A.to_wav_bytes(y, sr)
    st.session_state.setdefault("_audio", {})[result["id"]] = wav
    st.session_state.setdefault("_original", {})[result["id"]] = data
    st.session_state.setdefault("_results", {})[result["id"]] = result
    if cfg["save_history"]:
        try:
            S.save(result, y, sr, original=data)
        except Exception as exc:
            st.warning(f"Could not save to history: {exc}")
    cache[cache_key] = result["id"]
    set_current(result)
    return result


def set_current(result: dict) -> None:
    st.session_state["current_id"] = result["id"]
    st.session_state.setdefault("_results", {})[result["id"]] = result


def get_result(analysis_id: str | None) -> dict | None:
    if not analysis_id:
        return None
    res = st.session_state.get("_results", {}).get(analysis_id)
    if res is None:
        res = S.get(analysis_id)
        if res:
            st.session_state.setdefault("_results", {})[analysis_id] = res
    return res


def current() -> dict | None:
    return get_result(st.session_state.get("current_id"))


def audio_bytes(analysis_id: str) -> bytes | None:
    wav = st.session_state.get("_audio", {}).get(analysis_id)
    if wav is None:
        loaded = S.load_audio(analysis_id)
        if loaded:
            wav = A.to_wav_bytes(*loaded)
            st.session_state.setdefault("_audio", {})[analysis_id] = wav
    return wav


def original_bytes(result: dict) -> bytes | None:
    """The untouched uploaded/recorded file (session first, then history storage)."""
    data = st.session_state.get("_original", {}).get(result["id"])
    if data is None and result.get("original"):
        data = S.load_original(result)
    return data
