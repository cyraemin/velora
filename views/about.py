import platform
import sys

import streamlit as st

import velora
from velora import emotion as E
from velora import storage as S
from velora import transcribe as T
from velora import ui

ui.page_title("About", "How Velora works",
              "Velora looks at how you speak as well as what you say, and labels every number with where it "
              "came from.")
st.markdown("")

with st.container(key="card_about_sources"):
    ui.header("Three kinds of output, always labelled")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        ui.source_tag("measured")
        st.markdown("Computed directly from the waveform with **librosa** and **NumPy**: pitch (pYIN), RMS energy, "
                    "spectral centroid / rolloff / flatness, MFCCs, pauses, syllable-based pace.")
    with c2:
        ui.source_tag("model")
        st.markdown("Raw outputs of pretrained neural networks: the **emotion classifier**'s probabilities and "
                    "**Whisper**'s transcript. Never edited, re-weighted or invented.")
    with c3:
        ui.source_tag("derived")
        st.markdown("Readable text produced by fixed rules from the two sources on the left: the summary "
                    "paragraph, insight notes and expressive-transcript markup.")

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(key="card_about_models"):
        ui.header("Models", "Downloaded once from Hugging Face, then cached and run offline on your CPU")
        es, eerr = E.model_status()
        ts, terr = T.transcriber_status()
        info = E.MODEL_INFO
        st.markdown(
            f"**Emotion: [{info['id']}]({info['url']})**  \n"
            f"{info['architecture']} · trained on {info['training_data']} · classes: {info['classes']} · "
            f"{info['size']} · status: `{es}`"
        )
        if eerr:
            st.error(eerr)
        st.markdown(
            "**Speech-to-text: faster-whisper (CTranslate2 Whisper)**  \n"
            f"Sizes offered: {', '.join(T.WHISPER_SIZES)} · int8 on CPU · word timestamps + VAD · status: `{ts}`"
        )
        if terr:
            st.error(terr)
        if st.button("Retry loading the emotion model", icon=":material/refresh:"):
            E.retry_model()
            st.rerun()
with c2:
    with st.container(key="card_about_limits"):
        ui.header("Limits worth knowing")
        st.markdown(
            "- The emotion model picks from **seven emotions** plus *other*. Real feelings are often mixed or subtle, "
            "so treat the result as an estimate. Accuracy drops for very short, quiet or noisy clips.\n"
            "- Absolute **loudness** depends on your microphone and gain; compare levels only between clips "
            "recorded the same way.\n"
            "- **Pitch** ranges differ between speakers; Velora describes your pitch, it doesn't grade it.\n"
            "- **Jitter/shimmer** here are frame-level approximations, not clinical voice measures.\n"
            "- Nothing here is a psychological or medical assessment."
        )

with st.container(key="card_about_privacy"):
    ui.header("Privacy")
    if S.SESSION_ONLY:
        st.markdown("This hosted copy analyses your audio on its server. Nothing is written to disk: History is kept "
                    "in memory for your browser session only, is never visible to other visitors, and is discarded "
                    "when you close the tab. For fully offline use, run Velora on your own computer.")
    else:
        st.markdown("Audio never leaves your computer. Analyses are stored in `data/velora.db` with a 16 kHz copy of "
                    "the audio in `data/audio/`. Delete them at any time from **History**, or turn off saving in "
                    "**Engine settings**. The only network access is the one-time model download.")

with st.expander("Environment"):
    import librosa
    import numpy

    st.code(
        f"Velora {velora.__version__}\nPython {sys.version.split()[0]} on {platform.system()} {platform.release()}\n"
        f"Streamlit {st.__version__} · librosa {librosa.__version__} · NumPy {numpy.__version__}",
        language="text",
    )
