from collections import Counter

import pandas as pd
import streamlit as st

from velora import charts as C
from velora import dashboard as D
from velora import emotion as E
from velora import storage as S
from velora import ui

ui.page_title("History", "Your past analyses",
              ("Kept only for this browser session. Nobody else can see it, and it is cleared when you leave. "
               if S.SESSION_ONLY else "Everything is stored locally in the project's data folder. ")
              + "Open, compare, rename or delete recordings.")
st.markdown("")

rows = S.list_all()
if not rows:
    with st.container(key="card_hist_empty"):
        ui.empty_state("No saved analyses yet",
                       "Analyses are saved automatically (toggle in Engine settings). Record or upload a clip to start.")
        st.page_link("views/home.py", label="Go to Home", icon=":material/home:")
    st.stop()

# ---- overview
with_model = [r for r in rows if r["top_emotion"]]
counts = Counter(r["top_emotion"] for r in with_model)
m = st.columns(4)
m[0].metric("Recordings", len(rows))
m[1].metric("Total audio", f"{sum(r['duration'] or 0 for r in rows) / 60:.1f} min")
m[2].metric("Most frequent model class", counts.most_common(1)[0][0] if counts else "—")
m[3].metric("With transcript", sum(1 for r in rows if r["transcript"]))

if with_model:
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        with st.container(key="card_hist_counts"):
            ui.header("Top predicted class", "Across all saved recordings")
            ui.source_tag("model")
            st.plotly_chart(C.history_counts(counts), config=D.PLOT_CFG, key="hist_counts")
    with c2:
        with st.container(key="card_hist_time"):
            ui.header("Over time", "Each dot is one recording")
            ui.source_tag("model")
            st.plotly_chart(C.history_timeline(with_model), config=D.PLOT_CFG, key="hist_time")

# ---- table
with st.container(key="card_hist_table"):
    ui.header("Recordings", "Select a row to act on it")
    f1, f2, f3 = st.columns([2, 2, 1.2])
    q = f1.text_input("Search name or transcript", placeholder="Search…", label_visibility="collapsed")
    emos = f2.multiselect("Model class", E.EMOTIONS, placeholder="Filter by model class", label_visibility="collapsed")
    sources = sorted({r["source"] for r in rows})
    src = f3.multiselect("Source", sources, placeholder="Source", label_visibility="collapsed")

    filt = [
        r for r in rows
        if (not q or q.lower() in r["name"].lower() or q.lower() in (r["transcript"] or "").lower())
        and (not emos or r["top_emotion"] in emos)
        and (not src or r["source"] in src)
    ]
    df = pd.DataFrame([{
        "When": r["created_at"].replace("T", " "),
        "Name": r["name"],
        "Source": r["source"],
        "Duration (s)": r["duration"],
        "Model class": r["top_emotion"] or "—",
        "Probability": r["confidence"],
        "Transcript": (r["transcript"] or "")[:90],
    } for r in filt])
    if df.empty:
        st.caption("No recordings match the filters.")
        st.stop()
    event = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="hist_df",
        column_config={"Probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")},
    )
    st.download_button("Export list as CSV", df.to_csv(index=False), "velora_history.csv", "text/csv",
                       icon=":material/download:")

sel_rows = event.selection.rows if event and event.selection else []
if sel_rows:
    row = filt[sel_rows[0]]
    with st.container(key="card_hist_actions"):
        ui.header(row["name"], row["created_at"].replace("T", " "))
        wav = ui.audio_bytes(row["id"])
        if wav:
            st.audio(wav, format="audio/wav")
        a1, a2, a3, a4 = st.columns(4)
        if a1.button("Open in Analysis", type="primary", icon=":material/description:", width="stretch"):
            st.session_state["current_id"] = row["id"]
            st.switch_page("views/analysis.py")
        if a2.button("Use as A in Contrast Lab", icon=":material/looks_one:", width="stretch"):
            st.session_state["contrast_a"] = row["id"]
            st.switch_page("views/contrast_lab.py")
        if a3.button("Use as B in Contrast Lab", icon=":material/looks_two:", width="stretch"):
            st.session_state["contrast_b"] = row["id"]
            st.switch_page("views/contrast_lab.py")
        with a4.popover("Rename / delete", icon=":material/edit:", width="stretch"):
            new = st.text_input("New name", value=row["name"], key=f"rn_{row['id']}")
            if st.button("Save name", key=f"rns_{row['id']}"):
                S.rename(row["id"], new)
                st.session_state.get("_results", {}).pop(row["id"], None)
                st.rerun()
            st.divider()
            if st.button("Delete this recording", key=f"del_{row['id']}", icon=":material/delete:",
                         help="Removes the analysis, the 16 kHz copy and the original file."):
                S.delete(row["id"])
                st.session_state.get("_results", {}).pop(row["id"], None)
                if st.session_state.get("current_id") == row["id"]:
                    st.session_state.pop("current_id")
                st.rerun()

    full = ui.get_result(row["id"])
    if full:
        with st.expander("Downloads", icon=":material/download:"):
            D.downloads(full, f"hist_{row['id']}")

with st.expander("Danger zone"):
    confirm = st.checkbox("I understand this permanently deletes all saved analyses and audio.")
    if st.button("Delete all history", disabled=not confirm, icon=":material/delete_forever:"):
        S.clear()
        st.session_state.pop("current_id", None)
        st.session_state["_results"] = {}
        st.rerun()
