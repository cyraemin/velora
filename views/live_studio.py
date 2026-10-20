import pandas as pd
import streamlit as st

from velora import audio as A
from velora import dashboard as D
from velora import emotion as E
from velora import ui

PROMPTS = {
    "Free speech": ("—", "Talk about your day, a recent idea, or anything on your mind for 10 to 20 seconds."),
    "Happy": ("Happy", "I can't believe it, we actually did it! This is the best news I've had all year."),
    "Neutral": ("Neutral", "The meeting is scheduled for Tuesday at ten. Please bring the quarterly figures."),
    "Sad": ("Sad", "I kept the letter for years. I suppose I always hoped it would end differently."),
    "Angry": ("Angry", "No. I told you three times already. This is not acceptable, and it stops today."),
    "Fearful": ("Fearful", "Did you hear that? There's someone downstairs. Please, call someone, quickly."),
    "Surprised": ("Surprised", "Wait, you're here? I had no idea you were coming. When did you get in?"),
    "Disgusted": ("Disgusted", "Ugh, what is that smell? That milk has been in there for weeks. It's revolting."),
}

ui.page_title("Live studio", "Record a take and get feedback",
              "Pick a prompt, record yourself reading it, and see how the measurements and the emotion model "
              "respond. Reading the same line in different moods is a good way to explore.")
st.markdown("")

left, right = st.columns([1, 1.15], gap="large")
with left:
    with st.container(key="card_studio_prompt"):
        ui.header("1 · Choose a prompt", "Optional. The target mood is only compared with the model's answer.")
        choice = st.pills("Prompt", list(PROMPTS), default="Free speech", label_visibility="collapsed",
                          key="studio_prompt") or "Free speech"
        target, text = PROMPTS[choice]
        st.markdown(f'<div class="vl-quote vl-serif" style="font-size:1.15rem">“{text}”</div>',
                    unsafe_allow_html=True)
    with st.container(key="card_studio_rec"):
        ui.header("2 · Record", "Click the microphone to start and again to stop")
        rec = st.audio_input("Studio recording", label_visibility="collapsed", key="studio_rec")
        auto = st.toggle("Analyse automatically after each take", value=True, key="studio_auto")
        go = st.button("Analyse this take", type="primary", disabled=rec is None, icon=":material/graphic_eq:")

result = None
if rec is not None:
    data = rec.getvalue()
    h = A.audio_hash(data)
    is_new = st.session_state.get("studio_last_hash") != h
    if go or (auto and is_new):
        n = len(st.session_state.get("takes", [])) + 1
        with right:
            result = ui.run_analysis(data, f"studio_take_{n}_{choice.lower().replace(' ', '_')}.wav", "live studio")
        if result:
            st.session_state["studio_last_hash"] = h
            st.session_state["studio_result_id"] = result["id"]
            if is_new:
                emo = result["emotion"]
                st.session_state.setdefault("takes", []).append({
                    "Take": n,
                    "Prompt": choice,
                    "Target": target,
                    "Model top class": emo["top"] if emo.get("ok") else "—",
                    "Probability": round(emo["confidence"], 3) if emo.get("ok") else None,
                    "Match": ("—" if target == "—" else ("Yes" if emo.get("ok") and emo["top"] == target else "No")),
                    "Pitch (Hz)": result["stats"]["pitch_median"],
                    "Pitch var (st)": result["stats"]["pitch_var_st"],
                    "Energy (dBFS)": result["stats"]["energy_db"],
                    "Rate (wpm)": result["insights"][2]["value"],
                    "_id": result["id"],
                })

result = result or ui.get_result(st.session_state.get("studio_result_id"))

with right:
    if result is None:
        with st.container(key="card_studio_empty"):
            ui.empty_state("No take yet", "Record a clip on the left. Results appear here a few seconds later.")
    else:
        q = result["quality"]
        with st.container(key="card_studio_quality"):
            ui.header("Mic check", "Recording quality of this take")
            ui.source_tag("measured")
            ui.stat_tiles([
                ("Peak", f"{q['peak_dbfs']:.1f} dBFS"),
                ("Est. SNR", f"{q['snr_db']:.0f} dB" + ("+" if q["snr_db"] >= 60 else "")),
                ("Clipping", f"{q['clipping_pct']:.2f}%"),
                ("Speech", f"{result['stats']['speech_duration']:.1f}s"),
            ])
            st.markdown("")
            tips = []
            if q["clipping_pct"] > 0.1:
                tips.append("Some samples clipped. Move back from the microphone or lower the input gain.")
            if q["peak_dbfs"] < -20:
                tips.append("The level is low. Speak closer to the microphone.")
            if q["snr_db"] < 15:
                tips.append("Background noise is high relative to your voice; measurements may be less reliable.")
            if result["stats"]["speech_duration"] < 2:
                tips.append("Very little speech detected. Takes of 5 seconds or more give steadier results.")
            st.caption(" ".join(tips) if tips else "Clean recording: good level and no clipping.")
        with st.container(key="card_studio_emotion"):
            ui.header("Emotion model", "Clip-level prediction")
            ui.source_tag("model", f"Model output · {E.short_name(result['emotion'])}")
            ui.emotion_card(result["emotion"])
        with st.container(key="card_studio_text"):
            ui.header("Expressive text")
            D.transcript_block(result, show_raw=False)
            D.expressive_block(result)
            st.markdown("")
            ui.source_tag("derived")
            ui.interpretation_block(result)
        st.page_link("views/analysis.py", label="Open full analysis", icon=":material/description:")

takes = st.session_state.get("takes", [])
if takes:
    st.markdown("")
    with st.container(key="card_studio_takes"):
        ui.header("Session takes", "Compare your takes in this session (target vs the model's top class)")
        df = pd.DataFrame(takes).drop(columns=["_id"])
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        if st.button("Clear session takes", icon=":material/delete_sweep:"):
            st.session_state["takes"] = []
            st.rerun()
