"""Dashboard sections shared by Home, Live Studio and Analysis."""
from __future__ import annotations

import html

import streamlit as st

from . import audio as A
from . import charts as C
from . import emotion as E
from . import exports as X
from . import features as F
from . import interpret as I
from . import ui

PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


@st.cache_data(show_spinner=False, max_entries=6)
def _spectral(wav: bytes):
    y, sr, _ = A.load_audio(wav, "x.wav")
    S_db, st_, sf_ = F.mel_spectrogram(y, sr)
    M, mt = F.mfcc(y, sr)
    # thin out time frames for plotting performance
    k = max(1, S_db.shape[1] // 1500)
    return S_db[:, ::k], st_[::k], sf_, M[:, ::k], mt[::k]


def player(result: dict) -> None:
    s = result["stats"]
    wav = ui.audio_bytes(result["id"])
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        st.markdown(f"**{html.escape(result['name'])}**")
        st.caption(f"{s['duration']:.1f}s · speech {s['speech_duration']:.1f}s · {result['source']} · "
                   f"{result['created_at'].replace('T', ' ')}" + (" · truncated to 3 min" if result.get("truncated") else ""))
    with c2:
        if wav:
            st.audio(wav, format="audio/wav")
        else:
            st.caption("Audio not stored for this analysis.")


def acoustic_tabs(result: dict, key: str, height: int = 280) -> None:
    ui.source_tag("measured", "Measured · librosa")
    tabs = st.tabs(["Waveform", "Spectrogram", "MFCC", "Pitch & Energy", "Spectral centroid"])
    with tabs[0]:
        st.plotly_chart(C.waveform(result["envelope"], height), config=PLOT_CFG, key=f"{key}_wave")
    wav = ui.audio_bytes(result["id"])
    spectral = _spectral(wav) if wav else None
    with tabs[1]:
        if spectral:
            st.plotly_chart(C.spectrogram(spectral[0], spectral[1], spectral[2], height + 20), config=PLOT_CFG, key=f"{key}_spec")
        else:
            st.caption("Spectrogram needs the stored audio.")
    with tabs[2]:
        if spectral:
            st.plotly_chart(C.mfcc(spectral[3], spectral[4], height + 20), config=PLOT_CFG, key=f"{key}_mfcc")
            st.caption("Mel-frequency cepstral coefficients 1–12 (c0, overall loudness, omitted). "
                       "They summarise the spectral envelope, roughly the shape of the vocal tract.")
        else:
            st.caption("MFCCs need the stored audio.")
    with tabs[3]:
        st.plotly_chart(C.pitch_energy(result["tracks"], result["pauses"], height + 100), config=PLOT_CFG, key=f"{key}_pe")
        st.caption("Pitch via probabilistic YIN (pYIN); gaps are unvoiced or silent frames.")
    with tabs[4]:
        st.plotly_chart(C.spectral_track(result["tracks"], height), config=PLOT_CFG, key=f"{key}_cent")
        st.caption("Spectral centroid = the 'centre of mass' of the spectrum; higher values sound brighter.")


def timeline(result: dict, key: str, height: int = 300) -> None:
    emo = result.get("emotion") or {}
    win = emo.get("window_s", 2.0)
    ui.source_tag("model", f"Model output · {win:g} s windows")
    if not emo.get("ok"):
        st.caption(f"Timeline unavailable: {emo.get('error', 'the emotion model did not run')}.")
        return
    tl = emo["timeline"]
    if len(tl["t"]) < 2:
        st.caption("The recording is shorter than two analysis windows, so the timeline has a single point. "
                   "See the clip-level prediction.")
    st.plotly_chart(C.emotion_timeline(tl, height, win), config=PLOT_CFG, key=f"{key}_tl")


def transcript_block(result: dict, show_raw: bool = True) -> None:
    tr = result.get("transcript")
    if tr:
        ui.source_tag("model", f"Model output · Whisper {tr['model']} · {tr['language']}")
        if show_raw:
            st.markdown(f'<div class="vl-quote">{html.escape(tr["text"])}</div>', unsafe_allow_html=True)
    else:
        st.caption(f"No transcript: {(result.get('transcript_error') or 'unavailable').rstrip('.')}. "
                   "Showing phrase-level prosody instead.")


def expressive_block(result: dict) -> None:
    ui.source_tag("derived", "Expressive markup · from measured prosody")
    st.markdown(f'<div class="vl-quote">{I.expressive_html(result)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vl-legend"><b>Bold</b>: at least 3 dB louder than your median word · '
        '<span class="vl-soft">Italic</span>: at least 5 dB softer · '
        '<span class="vl-high">Underline</span>: 3 or more semitones above your median pitch · '
        '↗ ↘: pitch rises or falls within the word · … and ‖: measured pause · '
        'Coloured label: the emotion model\'s top class for that line. Hover a word for its measurements.</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _pdf(result_json: str) -> bytes:
    import json

    return X.pdf_report(json.loads(result_json))


def downloads(result: dict, key: str) -> None:
    """Download buttons for every export of one analysis."""
    base = X.base_name(result)
    orig = ui.original_bytes(result)
    o = result.get("original") or {}
    c = st.columns(3)
    if orig:
        ext = o.get("ext", "wav")
        c[0].download_button(f"Original audio (.{ext})", orig, f"{base}_original.{ext}",
                             A.MIME_TYPES.get(ext, "application/octet-stream"),
                             icon=":material/audio_file:", width="stretch", key=f"{key}_dl_orig")
    else:
        c[0].button("Original audio unavailable", disabled=True, width="stretch", key=f"{key}_dl_orig_na",
                    help="This analysis was made before original files were stored, or history saving was off.")
    c[1].download_button("Transcript (.txt)", X.transcript_txt(result), f"{base}_transcript.txt", "text/plain",
                         icon=":material/subject:", width="stretch", key=f"{key}_dl_txt")
    c[2].download_button("Analysis (.json)", X.analysis_json(result), f"{base}.json", "application/json",
                         icon=":material/data_object:", width="stretch", key=f"{key}_dl_json")
    c = st.columns(3)
    c[0].download_button("Acoustic metrics (.csv)", X.metrics_csv(result), f"{base}_metrics.csv", "text/csv",
                         icon=":material/table:", width="stretch", key=f"{key}_dl_csv")
    c[1].download_button("Frame tracks (.csv)", X.tracks_csv(result), f"{base}_tracks.csv", "text/csv",
                         icon=":material/timeline:", width="stretch", key=f"{key}_dl_tracks",
                         help="Pitch, energy and spectral centroid every ~20 ms")
    if X.pdf_available():
        try:
            pdf = _pdf(X.analysis_json(result))
            c[2].download_button("PDF report", pdf, f"{base}_report.pdf", "application/pdf", type="primary",
                                 icon=":material/picture_as_pdf:", width="stretch", key=f"{key}_dl_pdf")
        except Exception as exc:
            c[2].button("PDF report failed", disabled=True, width="stretch", key=f"{key}_dl_pdf_err",
                        help=f"{type(exc).__name__}: {exc}")
    else:
        c[2].button("PDF needs fpdf2", disabled=True, width="stretch", key=f"{key}_dl_pdf_na",
                    help="pip install fpdf2")
    tl = X.timeline_csv(result)
    wav = ui.audio_bytes(result["id"])
    extra = st.columns(3)
    if tl:
        extra[0].download_button("Emotion timeline (.csv)", tl, f"{base}_emotion_timeline.csv", "text/csv",
                                 icon=":material/show_chart:", width="stretch", key=f"{key}_dl_tl")
    if wav:
        extra[1].download_button("Analysed audio (16 kHz .wav)", wav, f"{base}_16k.wav", "audio/wav",
                                 icon=":material/graphic_eq:", width="stretch", key=f"{key}_dl_wav")


def dashboard(result: dict, key: str = "home") -> None:
    """The main two-column results layout used on Home."""
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(key=f"card_{key}_player"):
            player(result)
        with st.container(key=f"card_{key}_acoustic"):
            ui.header("Acoustic analysis", "Waveform, spectrum, pitch and energy of the recording")
            acoustic_tabs(result, key)
        with st.container(key=f"card_{key}_timeline"):
            ui.header("Emotion over time", "Model probabilities for each analysis window")
            timeline(result, key)
    with right:
        with st.container(key=f"card_{key}_emotion"):
            ui.header("Emotion", "Predicted class and model probabilities")
            ui.source_tag("model", f"Model output · {E.short_name(result.get('emotion'))}")
            ui.emotion_card(result.get("emotion"))
        with st.container(key=f"card_{key}_insights"):
            ui.header("Key measurements", "Measured directly from the signal")
            ui.source_tag("measured")
            ui.insight_tiles(result["insights"])
        with st.container(key=f"card_{key}_text"):
            ui.header("Transcript", "What you said and how you said it")
            transcript_block(result, show_raw=False)
            expressive_block(result)
            st.markdown("")
            ui.source_tag("derived")
            ui.interpretation_block(result)
            st.markdown("")
            ui.disclaimer()
