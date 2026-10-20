import pandas as pd
import streamlit as st

from velora import charts as C
from velora import dashboard as D
from velora import emotion as E
from velora import exports as X
from velora import interpret as I
from velora import storage as S
from velora import ui

FEATURES = X.FEATURES

# ---- choose which analysis to show
options: dict[str, str] = {}
for rid, res in st.session_state.get("_results", {}).items():
    options[rid] = f"{res['name']} · {res['created_at'].replace('T', ' ')}"
for row in S.list_all():
    options.setdefault(row["id"], f"{row['name']} · {row['created_at'].replace('T', ' ')}")

ui.page_title("Analysis", "Full voice report",
              "Every measurement, every model output, and how they were turned into words — clearly separated.")
st.markdown("")

if not options:
    with st.container(key="card_analysis_empty"):
        ui.empty_state("Nothing to analyse yet", "Upload or record a clip on Home or in the Live Studio.")
        st.page_link("views/home.py", label="Go to Home", icon=":material/home:")
    st.stop()

ids = list(options)
cur = st.session_state.get("current_id")
sel = st.selectbox("Recording", ids, index=ids.index(cur) if cur in ids else 0, format_func=options.get)
if sel != cur:
    st.session_state["current_id"] = sel
result = ui.get_result(sel)
if result is None:
    st.error("This analysis could not be loaded.")
    st.stop()
s = result["stats"]
emo = result.get("emotion") or {}
wpm, wpm_src = I.speaking_rate(s, result.get("transcript"))

with st.container(key="card_an_player"):
    D.player(result)

st.markdown("")
k = st.columns(6)
k[0].metric("Duration", f"{s['duration']:.1f}s")
k[1].metric("Speech ratio", f"{s['speech_ratio'] * 100:.0f}%")
k[2].metric("Median pitch", f"{s['pitch_median']:.0f} Hz" if s["pitch_median"] else "—")
k[3].metric("Pitch variation", f"{s['pitch_var_st']:.1f} st")
k[4].metric("Speaking rate", f"{wpm:.0f} wpm" if wpm else "—", help=wpm_src)
k[5].metric("Pauses", s["pause_count"])

# ---- 1. Model outputs
st.markdown("### Model outputs")
st.caption("Raw predictions of the pretrained models — nothing here is computed by rules.")
c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(key="card_an_emotion"):
        ui.header("Speech emotion recognition", E.MODEL_ID)
        ui.source_tag("model")
        ui.emotion_card(emo)
        if emo.get("ok"):
            with st.expander("Model details & raw scores"):
                st.dataframe(pd.DataFrame({"class": list(emo["probs"]), "probability": list(emo["probs"].values())}),
                             hide_index=True, width="stretch")
                info = E.MODEL_INFO
                st.markdown(f"- **Model:** [{info['id']}]({info['url']})\n- **Architecture:** {info['architecture']}\n"
                            f"- **Training data:** {info['training_data']}\n- **Classes:** {info['classes']}\n"
                            f"- **Scope of this score:** {emo['scope']}")
with c2:
    with st.container(key="card_an_transcript"):
        ui.header("Speech-to-text", "faster-whisper")
        tr = result.get("transcript")
        D.transcript_block(result)
        if tr:
            st.caption(f"Detected language: {tr['language']} (p = {tr['language_prob']:.2f}) · "
                       f"{sum(len(x['words']) for x in tr['segments'])} words")
            with st.expander("Segments & word confidence"):
                rows = [{"start": w["start"], "end": w["end"], "word": w["word"].strip(), "confidence": w["prob"]}
                        for seg in tr["segments"] for w in seg["words"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=260,
                             column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

with st.container(key="card_an_timeline"):
    ui.header("Emotion timeline", "Model probabilities for overlapping 2-second windows")
    D.timeline(result, "an", 340)
    if emo.get("ok"):
        with st.expander("Window-level outputs"):
            tl = emo["timeline"]
            df = pd.DataFrame({"start (s)": tl["start"], "end (s)": tl["end"], "contains speech": tl["speech"],
                               **{e: tl[e] for e in E.EMOTIONS}})
            st.dataframe(df, hide_index=True, width="stretch", height=260)

# ---- 2. Measurements
st.markdown("### Measured acoustic features")
st.caption("Computed directly from the waveform with librosa and NumPy.")
with st.container(key="card_an_acoustic"):
    D.acoustic_tabs(result, "an", 300)

c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    with st.container(key="card_an_table"):
        ui.header("Feature table", "All clip-level statistics")
        ui.source_tag("measured")
        rows = [{"Feature": lbl, "Value": s[key], "Unit": unit, "What it means": desc} for key, lbl, unit, desc in FEATURES]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
with c2:
    with st.container(key="card_an_profile"):
        ui.header("Voice profile", "Measured features scaled to 0–1 against typical speech ranges")
        ui.source_tag("measured", "Measured · normalised")
        st.plotly_chart(C.radar([("This recording", result["profile"], C.SERIES_A)], 360),
                        config=D.PLOT_CFG, key="an_radar")
        st.caption("Scaling ranges are fixed reference values (e.g. pitch 90–280 Hz, rate 1.5–6.5 syll/s); "
                   "they describe, not judge.")

with st.container(key="card_an_quality"):
    ui.header("Recording quality")
    ui.source_tag("measured")
    q = result["quality"]
    m = st.columns(5)
    m[0].metric("Peak level", f"{q['peak_dbfs']} dBFS")
    m[1].metric("Speech level (p90)", f"{q['speech_level_db']} dB")
    m[2].metric("Noise floor (p10)", f"{q['noise_floor_db']} dB")
    m[3].metric("Est. SNR", f"{q['snr_db']} dB")
    m[4].metric("Clipped samples", f"{q['clipping_pct']}%")

# ---- 3. Interpretation
st.markdown("### Interpretation")
st.caption("Rule-based text built from the measurements and model outputs above.")
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    with st.container(key="card_an_expressive"):
        ui.header("Expressive transcript")
        D.expressive_block(result)
with c2:
    with st.container(key="card_an_interp"):
        ui.header("Summary")
        ui.source_tag("derived")
        ui.interpretation_block(result)
        st.markdown("")
        ui.insight_tiles(result["insights"])
        ui.disclaimer()

# ---- Export
st.markdown("### Downloads")
with st.container(key="card_an_downloads"):
    D.downloads(result, "an")
