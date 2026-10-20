import html

import pandas as pd
import streamlit as st

from velora import audio as A
from velora import charts as C
from velora import contrast as X
from velora import dashboard as D
from velora import storage as S
from velora import ui

ui.page_title("Contrast lab", "Compare two recordings",
              "Same sentence, different mood? Before and after practice? Put two recordings side by side and see "
              "what changed — in the measurements, in the words, and in the emotion model's output.")
st.markdown("")


def _options() -> dict:
    opts = {}
    for rid, res in st.session_state.get("_results", {}).items():
        opts[rid] = f"{res['name']} · {res['created_at'][5:16].replace('T', ' ')}"
    for r in S.list_all():
        opts.setdefault(r["id"], f"{r['name']} · {r['created_at'][5:16].replace('T', ' ')}")
    return opts


def _swap():
    a, b = st.session_state.get("contrast_a"), st.session_state.get("contrast_b")
    st.session_state["contrast_a"], st.session_state["contrast_b"] = b, a
    for k in ("cl_sel_a", "cl_sel_b"):
        st.session_state.pop(k, None)


def slot(label: str, key: str, color: str) -> dict | None:
    with st.container(key=f"card_cl_{key}"):
        st.markdown(f'<p class="vl-h"><span style="color:{color}">●</span> Recording {label}</p>', unsafe_allow_html=True)
        mode = st.segmented_control("Source", ["Saved", "Upload", "Record"], default="Saved",
                                    key=f"cl_mode_{key}", label_visibility="collapsed") or "Saved"
        if mode == "Saved":
            opts = _options()
            if not opts:
                st.caption("No saved analyses yet — upload or record instead.")
                return None
            ids = list(opts)
            pre = st.session_state.get(f"contrast_{key}")
            default = ids.index(pre) if pre in ids else (0 if key == "a" else min(1, len(ids) - 1))
            rid = st.selectbox("Recording", ids, index=default, format_func=opts.get,
                               key=f"cl_sel_{key}", label_visibility="collapsed")
            st.session_state[f"contrast_{key}"] = rid
        else:
            if mode == "Upload":
                f = st.file_uploader("Upload", type=A.SUPPORTED_TYPES, key=f"cl_up_{key}", label_visibility="collapsed")
                data, name = (f.getvalue(), f.name) if f else (None, None)
            else:
                r = st.audio_input("Record", key=f"cl_rec_{key}", label_visibility="collapsed")
                data, name = (r.getvalue(), f"contrast_{key}_recording.wav") if r else (None, None)
            if data is None:
                st.caption("Add audio to analyse it here (it is also saved to History).")
                return None
            h = A.audio_hash(data)
            if st.session_state.get(f"cl_hash_{key}") != h:
                res = ui.run_analysis(data, name, "contrast lab")   # shows its own error for bad audio
                if not res:
                    return None
                st.session_state[f"cl_hash_{key}"] = h
                st.session_state[f"contrast_{key}"] = res["id"]
        res = ui.get_result(st.session_state.get(f"contrast_{key}"))
        if res is None:
            st.warning("This analysis could not be loaded — it may have been deleted.")
            return None
        wav = ui.audio_bytes(res["id"])
        if wav:
            st.audio(wav, format="audio/wav")
        else:
            st.caption("Audio file missing for this analysis; measurements are still available.")
        ui.emotion_card(res.get("emotion"), compact=True)
        return res


ca, cb = st.columns(2, gap="large")
with ca:
    ra = slot("A", "a", C.SERIES_A)
with cb:
    rb = slot("B", "b", C.SERIES_B)
st.button("⇄ Swap A and B", on_click=_swap, disabled=not (ra and rb))

if not (ra and rb):
    st.info("Choose two recordings to compare. Tip: in the Live Studio, read the same prompt in two moods, "
            "then compare the takes here.", icon=":material/compare_arrows:")
    st.stop()

cmp = X.compare(ra, rb)
errors = [m for lvl, m in cmp["checks"] if lvl == "error"]
if errors:
    for m in errors:
        st.error(m)
    st.stop()
notes = [(lvl, m) for lvl, m in cmp["checks"] if lvl != "error"]
if notes:
    with st.expander(f"Comparison notes ({len(notes)})", expanded=any(lvl == "warning" for lvl, _ in notes),
                     icon=":material/info:"):
        for lvl, m in notes:
            st.markdown(f"{'⚠️' if lvl == 'warning' else 'ℹ️'} {m}")

st.markdown("")
tab_over, tab_words, tab_emo, tab_pitch = st.tabs(
    ["Overview", "Same words, different delivery", "Emotion model", "Pitch & energy"])

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        with st.container(key="card_cl_summary"):
            ui.header("In plain words", "B compared with A")
            ui.source_tag("derived")
            st.markdown("\n".join(f"- {s}" for s in cmp["summary"]) or "_No notable differences._")
        with st.container(key="card_cl_table"):
            ui.header("Measured differences")
            ui.source_tag("measured")
            rows = [{
                "Feature": r["feature"],
                "A": X.fmt(r["a"], r["nd"], r["unit"]),
                "B": X.fmt(r["b"], r["nd"], r["unit"]),
                "Δ (B − A)": X.fmt(r["delta"], r["nd"], r["unit"], sign=True)
                             + (f"  ({r['pct']:+.0f}%)" if r["pct"] is not None else ""),
            } for r in cmp["metrics"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=460)
    with c2:
        with st.container(key="card_cl_radar"):
            ui.header("Voice profiles", "Measured features scaled 0–1")
            ui.source_tag("measured", "Measured · normalised")
            st.plotly_chart(C.radar([("A", ra["profile"], C.SERIES_A), ("B", rb["profile"], C.SERIES_B)], 420),
                            config=D.PLOT_CFG, key="cl_radar")
            st.caption("Level differences (dBFS) are only comparable when both clips were recorded with the "
                       "same microphone and gain.")

# ---------------------------------------------------------------- words
with tab_words:
    al = cmp["alignment"]
    with st.container(key="card_cl_transcripts"):
        ui.header("Transcripts", "Words only in one recording are highlighted")
        ui.source_tag("model", "Model output · Whisper")
        t1, t2 = st.columns(2, gap="large")
        if al["available"]:
            ha, hb = [], []
            for op in al["ops"]:
                a, b = html.escape(op["a"]), html.escape(op["b"])
                if op["op"] == "equal":
                    ha.append(a); hb.append(b)
                else:
                    if a:
                        ha.append(f'<mark style="background:#EDE8FD;border-radius:4px;padding:0 3px">{a}</mark>')
                    if b:
                        hb.append(f'<mark style="background:#FBEBD6;border-radius:4px;padding:0 3px">{b}</mark>')
            for col, lbl, color, toks in ((t1, "A", C.SERIES_A, ha), (t2, "B", C.SERIES_B, hb)):
                col.markdown(f'<div class="vl-quote"><b style="color:{color}">{lbl}</b> · {" ".join(toks)}</div>',
                             unsafe_allow_html=True)
            label = {"same": "the same words", "partial": "partly the same words", "different": "different words"}[al["status"]]
            st.caption(f"Word-sequence similarity: {al['similarity'] * 100:.0f}% — the recordings use {label}.")
        else:
            for col, r, lbl in ((t1, ra, "A"), (t2, rb, "B")):
                tr = r.get("transcript")
                body = (html.escape(tr["text"]) if tr else
                        f'<i>No transcript — {html.escape(r.get("transcript_error") or "unavailable")}</i>')
                col.markdown(f'<div class="vl-quote"><b>{lbl}</b> · {body}</div>', unsafe_allow_html=True)

    if not al["available"]:
        st.info("Word-by-word delivery needs a transcript of both recordings in the same language. "
                "Enable Speech-to-text in Engine settings and re-analyse.", icon=":material/subtitles_off:")
    elif al["status"] == "different" or len(al["pairs"]) < 3:
        st.info("These recordings say different things, so word-level delivery can't be compared. "
                "For this view, record the same sentence twice (e.g. a Live Studio prompt) with different moods.",
                icon=":material/compare_arrows:")
    else:
        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            with st.container(key="card_cl_delivery"):
                ui.header("How the delivery differs", f"{len(al['pairs'])} shared words compared")
                ui.source_tag("derived", "Interpretation of measured word prosody")
                st.markdown("\n".join(f"- {n}" for n in cmp["delivery_notes"]))
                st.markdown(
                    '<div class="vl-legend" style="margin-top:.8rem">The same words can carry very different '
                    'meaning: <b>timing</b> (how long each word lasts, where you pause), <b>stress</b> (which words '
                    'are louder than your typical word) and <b>melody</b> (how far each word sits above or below '
                    'your usual pitch). Loudness and pitch here are measured relative to each recording\'s own '
                    'median, so different voices and microphones remain comparable.</div>',
                    unsafe_allow_html=True)
        with c2:
            with st.container(key="card_cl_wordchart"):
                ui.header("Word by word", "Each pair of dots is one shared word")
                ui.source_tag("measured", "Measured · word timings from Whisper")
                st.plotly_chart(C.word_delivery(al["pairs"]), config=D.PLOT_CFG, key="cl_words")
                if len(al["pairs"]) > 40:
                    st.caption("Showing the first 40 shared words; the full table is in the downloadable report.")
        with st.expander("Word-level table"):
            st.dataframe(pd.DataFrame(al["pairs"]), hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- emotion
with tab_emo:
    ea, eb = ra.get("emotion") or {}, rb.get("emotion") or {}
    with st.container(key="card_cl_emo"):
        ui.header("Emotion model output", "Clip-level probabilities from the same pretrained classifier")
        ui.source_tag("model")
        if ea.get("ok") and eb.get("ok"):
            st.plotly_chart(C.emotion_compare(("A", ea["probs"]), ("B", eb["probs"])), config=D.PLOT_CFG, key="cl_emo")
        else:
            c1, c2 = st.columns(2)
            for col, e, lbl in ((c1, ea, "A"), (c2, eb, "B")):
                with col:
                    st.markdown(f"**Recording {lbl}**")
                    ui.emotion_card(e)
    c1, c2 = st.columns(2, gap="large")
    for col, r, lbl in ((c1, ra, "A"), (c2, rb, "B")):
        with col:
            with st.container(key=f"card_cl_tl_{lbl}"):
                ui.header(f"Timeline · {lbl}")
                D.timeline(r, f"cl_{lbl}", 260)

# ---------------------------------------------------------------- pitch
with tab_pitch:
    with st.container(key="card_cl_pitch"):
        ui.header("Pitch contours", "Aligned by relative position in each recording")
        ui.source_tag("measured")
        st.plotly_chart(C.pitch_overlay(("A", ra["tracks"]), ("B", rb["tracks"])), config=D.PLOT_CFG, key="cl_pitch")
    c1, c2 = st.columns(2, gap="large")
    for col, r, lbl in ((c1, ra, "A"), (c2, rb, "B")):
        with col:
            with st.container(key=f"card_cl_pe_{lbl}"):
                ui.header(f"Pitch & energy · {lbl}")
                st.plotly_chart(C.pitch_energy(r["tracks"], r["pauses"], 340), config=D.PLOT_CFG, key=f"cl_pe_{lbl}")

# ---------------------------------------------------------------- export
st.markdown("")
with st.container(key="soft_cl_export"):
    c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
    c1.markdown("**Download the comparison**  \n<span class='vl-muted'>Markdown report, or a ZIP bundle with the report, "
                "full JSON for both analyses, word-level CSV and both audio files.</span>", unsafe_allow_html=True)
    c2.download_button("Report (.md)", X.report_markdown(cmp, ra, rb), f"velora_compare_{ra['id']}_{rb['id']}.md",
                       "text/markdown", icon=":material/article:", width="stretch")
    c3.download_button("Full bundle (.zip)",
                       X.report_bundle(cmp, ra, rb, ui.audio_bytes(ra["id"]), ui.audio_bytes(rb["id"])),
                       f"velora_compare_{ra['id']}_{rb['id']}.zip", "application/zip", type="primary",
                       icon=":material/folder_zip:", width="stretch")
