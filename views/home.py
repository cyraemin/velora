import html

import streamlit as st

from velora import audio as A
from velora import dashboard as D
from velora import storage as S
from velora import ui

UPLOAD_SVG = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#5B47D6" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V4"/><path d="M7 9l5-5 5 5"/>'
              '<path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>')
MIC_SVG = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#5B47D6" stroke-width="2" '
           'stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/>'
           '<path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/></svg>')

nonce = st.session_state.setdefault("home_nonce", 0)


def _mark(kind: str):
    st.session_state["home_pending"] = kind


def _clear():
    # new widget keys reset both the uploader and the recorder
    st.session_state["home_nonce"] = st.session_state.get("home_nonce", 0) + 1
    st.session_state.pop("home_pending", None)


with st.container(key="hero_home"):
    ui.page_title(
        "Speech analysis",
        "Analyse a voice recording",
        "Upload a file or record yourself. Velora measures how you speak, estimates the emotion in your voice "
        "and transcribes what you said.",
    )
    st.markdown("")
    with st.container(key="card_home_inputs"):
        c1, mid, c2 = st.columns([1, 0.1, 1], vertical_alignment="center")
        with c1:
            st.markdown(
                f'<div class="vl-inputhead"><div class="vl-iconcircle">{UPLOAD_SVG}</div><div><b>Upload a file</b>'
                f'<span>WAV, MP3, FLAC, OGG, M4A or WebM, up to 25 MB. The first 3 minutes are analysed.</span></div></div>',
                unsafe_allow_html=True)
            upload = st.file_uploader("Upload audio", type=A.SUPPORTED_TYPES, label_visibility="collapsed",
                                      key=f"home_upload_{nonce}", on_change=_mark, args=("upload",))
        with mid:
            st.markdown('<div class="vl-or">OR</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="vl-inputhead"><div class="vl-iconcircle">{MIC_SVG}</div><div><b>Record with your microphone</b>'
                '<span>Click the microphone to start, and again to stop. You can replay the take before '
                'analysing it.</span></div></div>',
                unsafe_allow_html=True)
            rec = st.audio_input("Record", label_visibility="collapsed", key=f"home_rec_{nonce}",
                                 on_change=_mark, args=("record",))

    pending = st.session_state.get("home_pending")
    chosen = None
    if pending == "record" and rec is not None:
        chosen = (rec.getvalue(), f"recording_{st.session_state.get('rec_n', 1)}.wav", "microphone")
    elif upload is not None:
        chosen = (upload.getvalue(), upload.name, "upload")
    elif rec is not None:
        chosen = (rec.getvalue(), f"recording_{st.session_state.get('rec_n', 1)}.wav", "microphone")

    cur = ui.current()
    analysed = bool(chosen and cur and A.audio_hash(chosen[0]) == cur.get("audio_hash"))
    if chosen:
        kb = len(chosen[0]) / 1024
        what = "Microphone recording" if chosen[2] == "microphone" else html.escape(chosen[1])
        state = "Analysed, results below" if analysed else "Ready to analyse"
        st.markdown(f'<div class="vl-ready"><div class="ic">{ui.ICONS["file"]}</div><div><b>{what}</b><br>'
                    f'<span>{kb:,.0f} KB · {state}</span></div></div>', unsafe_allow_html=True)
        if chosen[2] == "upload" and not analysed:
            st.audio(chosen[0])

    b1, b2, _ = st.columns([1.1, 0.8, 2.3])
    with b1:
        go = st.button("Analyse voice", type="primary", disabled=chosen is None or analysed, width="stretch",
                       icon=":material/graphic_eq:")
    with b2:
        st.button("Clear", on_click=_clear, disabled=upload is None and rec is None, width="stretch",
                  icon=":material/close:", help="Remove the selected file and recording")
    if chosen is None:
        st.caption("Choose a file or record a clip to enable analysis.")

if go and chosen:
    if chosen[2] == "microphone":
        st.session_state["rec_n"] = st.session_state.get("rec_n", 1) + 1
    if ui.run_analysis(*chosen):
        st.rerun()   # refresh header/sidebar model status and the "analysed" state

result = ui.current()
st.markdown("")
if result is None:
    with st.container(key="card_home_empty"):
        ui.empty_state("Results appear here",
                       "Velora measures pitch, energy, timbre and pace, runs a pretrained emotion model "
                       "and transcribes what you said" + ("." if S.SESSION_ONLY else ", all on your own computer."))
        st.markdown(
            '<div class="vl-steps">'
            '<div class="vl-step"><b>1 · Measure</b><span>librosa extracts pitch (pYIN), RMS energy, '
            'spectral centroid, MFCCs, pauses and pace.</span></div>'
            '<div class="vl-step"><b>2 · Recognise</b><span>emotion2vec+ estimates one of seven emotions for '
            'the whole clip and for every 3-second window.</span></div>'
            '<div class="vl-step"><b>3 · Transcribe</b><span>faster-whisper turns speech into words with '
            'timestamps.</span></div>'
            '<div class="vl-step"><b>4 · Express</b><span>Words are marked up with measured emphasis, pitch '
            'movement and pauses.</span></div></div>',
            unsafe_allow_html=True,
        )
else:
    D.dashboard(result, "home")
    with st.container(key="soft_home_cta"):
        c1, c2, c3 = st.columns([3, 1.1, 1], vertical_alignment="center")
        c1.markdown("**Go further**  \n"
                    "<span class='vl-muted'>Open the full report with PDF, CSV and JSON downloads, or compare two "
                    "recordings side by side.</span>", unsafe_allow_html=True)
        c2.page_link("views/analysis.py", label="Full report", icon=":material/description:")
        c3.page_link("views/contrast_lab.py", label="Contrast Lab", icon=":material/compare_arrows:")
