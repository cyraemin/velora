"""Plotly figures in Velora's lavender-and-ivory style."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .emotion import EMOTION_COLORS, EMOTIONS
from .features import PROFILE_AXES

INK = "#1F1B3A"
MUTED = "#6B6785"
GRID = "#EFECF6"
LAV = "#7C6BE0"
LAV_DARK = "#4B3BC4"
FONT = "Nunito Sans, Segoe UI, sans-serif"
SERIES_A, SERIES_B = "#6E56CF", "#C9781F"   # Contrast Lab: recording A / B

SEQ_SCALE = [[0, "#FBFAF6"], [0.3, "#E2DBFB"], [0.65, "#8B7CF0"], [1, "#2E2170"]]
DIV_SCALE = [[0, "#3A86D1"], [0.5, "#F4F2EE"], [1, "#E08A2E"]]


def _base(fig: go.Figure, height: int = 300, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=16, t=16 if not legend else 40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", title_font=dict(size=12))
    return fig


def waveform(env: dict, height: int = 260) -> go.Figure:
    t, lo, hi = env["t"], env["min"], env["max"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0.6, color=LAV), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t, y=lo, mode="lines", line=dict(width=0.6, color=LAV), fill="tonexty",
        fillcolor="rgba(124,107,224,0.55)",
        customdata=np.stack([lo, hi], axis=1),
        hovertemplate="%{x:.2f}s<br>min %{customdata[0]:.2f} · max %{customdata[1]:.2f}<extra></extra>",
    ))
    fig.update_yaxes(range=[-1.05, 1.05], title="Amplitude")
    fig.update_xaxes(title="Time (s)", ticksuffix="s")
    fig.update_layout(hovermode="x")
    return _base(fig, height)


def spectrogram(S_db: np.ndarray, times: np.ndarray, freqs: np.ndarray, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=S_db, x=times, y=freqs, colorscale=SEQ_SCALE, zmin=-80, zmax=0,
        colorbar=dict(title=dict(text="dB", side="top"), thickness=10, outlinewidth=0, len=0.9),
        hovertemplate="%{x:.2f}s · %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
    ))
    fig.update_yaxes(title="Mel frequency (Hz)", type="log")
    fig.update_xaxes(title="Time (s)")
    return _base(fig, height)


def mfcc(M: np.ndarray, times: np.ndarray, height: int = 300) -> go.Figure:
    Z = M[1:]  # drop c0 (overall loudness) so the shape coefficients are visible
    lim = float(np.percentile(np.abs(Z), 98)) or 1.0
    fig = go.Figure(go.Heatmap(
        z=Z, x=times, y=list(range(1, Z.shape[0] + 1)), colorscale=DIV_SCALE, zmid=0, zmin=-lim, zmax=lim,
        colorbar=dict(thickness=10, outlinewidth=0, len=0.9),
        hovertemplate="%{x:.2f}s · MFCC %{y}<br>%{z:.1f}<extra></extra>",
    ))
    fig.update_yaxes(title="Coefficient", dtick=2)
    fig.update_xaxes(title="Time (s)")
    return _base(fig, height)


def _pause_shapes(fig, pauses, rows):
    for a, b in pauses:
        for r in rows:
            fig.add_vrect(x0=a, x1=b, fillcolor="#F1EEFC", line_width=0, layer="below", row=r, col=1)


def pitch_energy(tracks: dict, pauses: list, height: int = 380) -> go.Figure:
    """Pitch and energy as two stacked panels sharing the time axis (no dual y-axis)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Pitch (F0)", "Energy (RMS)"))
    t = tracks["t"]
    fig.add_trace(go.Scatter(x=t, y=tracks["f0"], mode="lines", line=dict(width=2, color=LAV_DARK),
                             connectgaps=False, name="Pitch",
                             hovertemplate="%{x:.2f}s · %{y:.0f} Hz<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=tracks["rms_db"], mode="lines", line=dict(width=1.5, color="#2A9D7F"),
                             fill="tozeroy", fillcolor="rgba(42,157,127,0.10)", name="Energy",
                             hovertemplate="%{x:.2f}s · %{y:.1f} dBFS<extra></extra>"), row=2, col=1)
    _pause_shapes(fig, pauses, [1, 2])
    fig.update_yaxes(title="Hz", row=1, col=1)
    fig.update_yaxes(title="dBFS", row=2, col=1)
    fig.update_xaxes(title="Time (s) · shaded = measured pauses", row=2, col=1)
    fig.update_annotations(font=dict(size=12, color=INK, family=FONT), x=0, xanchor="left")
    fig.update_layout(hovermode="x unified")
    return _base(fig, height)


def spectral_track(tracks: dict, height: int = 260) -> go.Figure:
    t = np.array(tracks["t"])
    c = np.array([np.nan if v is None else v for v in tracks["centroid"]], dtype=float)
    act = np.array(tracks["active"], dtype=bool)
    c[~act] = np.nan
    fig = go.Figure(go.Scatter(x=t, y=c, mode="lines", line=dict(width=2, color=LAV_DARK), connectgaps=False,
                               hovertemplate="%{x:.2f}s · %{y:.0f} Hz<extra></extra>"))
    fig.update_yaxes(title="Spectral centroid (Hz)")
    fig.update_xaxes(title="Time (s) · speech frames only")
    return _base(fig, height)


def emotion_timeline(tl: dict, height: int = 320) -> go.Figure:
    """Raw per-window model probabilities, with a band showing the top class per window."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.1, 0.9], vertical_spacing=0.04)
    st, en = np.array(tl["start"]), np.array(tl["end"])
    mids = np.array(tl["t"])
    # band: each window centre owns the span halfway to its neighbours
    edges = np.concatenate([[st[0]], (mids[1:] + mids[:-1]) / 2, [en[-1]]]) if len(mids) > 1 else np.array([st[0], en[0]])
    for i in range(len(mids)):
        top = max(EMOTIONS, key=lambda e: tl[e][i])
        color = EMOTION_COLORS[top] if tl["speech"][i] else "#DCD8E6"
        label = top if tl["speech"][i] else f"{top} (little speech)"
        fig.add_trace(go.Bar(
            x=[edges[i + 1] - edges[i]], base=[edges[i]], y=[0], orientation="h",
            marker=dict(color=color, line=dict(width=1, color="white")),
            showlegend=False, hovertemplate=f"{edges[i]:.1f}–{edges[i + 1]:.1f}s · top: {label}<extra></extra>",
        ), row=1, col=1)
    for e in EMOTIONS:
        fig.add_trace(go.Scatter(
            x=mids, y=tl[e], name=e, mode="lines+markers" if len(mids) < 4 else "lines",
            line=dict(width=2, color=EMOTION_COLORS[e], shape="spline", smoothing=0.6),
            marker=dict(size=8),
            hovertemplate=f"{e}: " + "%{y:.0%}<extra></extra>",
        ), row=2, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title="Model probability", range=[0, 1.02], tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title="Time (s) · 2 s analysis windows", row=2, col=1)
    fig.update_traces(width=0.9, selector=dict(type="bar"))
    fig.update_layout(barmode="overlay", bargap=0, hovermode="x unified")
    return _base(fig, height, legend=True)


def radar(series: list[tuple[str, dict, str]], height: int = 360) -> go.Figure:
    labels = [lbl for _, lbl in PROFILE_AXES]
    keys = [k for k, _ in PROFILE_AXES]
    fig = go.Figure()
    for name, prof, color in series:
        vals = [prof[k] for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name=name, fill="toself",
            line=dict(color=color, width=2), opacity=0.85,
            hovertemplate="%{theta}: %{r:.2f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=GRID, linecolor=GRID),
        angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=INK, size=12)),
    ))
    fig = _base(fig, height, legend=len(series) > 1)
    fig.update_layout(margin=dict(l=80, r=80, t=50, b=40))
    return fig


def pitch_overlay(a: tuple[str, dict], b: tuple[str, dict], height: int = 300) -> go.Figure:
    fig = go.Figure()
    for (name, tracks), color in ((a, SERIES_A), (b, SERIES_B)):
        t = np.array(tracks["t"])
        x = t / (t[-1] or 1) * 100
        fig.add_trace(go.Scatter(x=x, y=tracks["f0"], name=name, mode="lines", connectgaps=False,
                                 line=dict(width=2, color=color),
                                 hovertemplate="%{x:.0f}% · %{y:.0f} Hz<extra>" + name + "</extra>"))
    fig.update_xaxes(title="Position in recording (%)", ticksuffix="%")
    fig.update_yaxes(title="Pitch (Hz)")
    return _base(fig, height, legend=True)


def emotion_compare(a: tuple[str, dict], b: tuple[str, dict], height: int = 300) -> go.Figure:
    fig = go.Figure()
    for (name, probs), color in ((a, SERIES_A), (b, SERIES_B)):
        fig.add_trace(go.Bar(x=EMOTIONS, y=[probs[e] for e in EMOTIONS], name=name, marker_color=color,
                             marker_line=dict(width=2, color="white"),
                             texttemplate="%{y:.0%}", textposition="outside", textfont=dict(color=INK),
                             hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>"))
    fig.update_yaxes(title="Model probability", tickformat=".0%", range=[0, 1.12])
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_traces(marker_cornerradius=4)
    return _base(fig, height, legend=True)


def word_delivery(pairs: list[dict], max_words: int = 40, height: int = 520) -> go.Figure:
    """Shared words: duration, loudness and pitch (each relative to its own recording) for A vs B."""
    pairs = pairs[:max_words]
    x = list(range(len(pairs)))
    labels = [p["word"] for p in pairs]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Word duration (ms)", "Loudness vs own median word (dB)",
                                        "Pitch vs own median (semitones)"))
    panels = [("dur_ms", "{:.0f} ms"), ("db", "{:+.1f} dB"), ("st", "{:+.1f} st")]
    for row, (key, _) in enumerate(panels, start=1):
        ya = [p[f"a_{key}"] for p in pairs]
        yb = [p[f"b_{key}"] for p in pairs]
        # connector segments make each word's change visible
        cx, cy = [], []
        for i, (a, b) in enumerate(zip(ya, yb)):
            if a is not None and b is not None:
                cx += [i, i, None]
                cy += [a, b, None]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color="#D9D4E8", width=2),
                                 hoverinfo="skip", showlegend=False), row=row, col=1)
        for name, ys, color in (("A", ya, SERIES_A), ("B", yb, SERIES_B)):
            fig.add_trace(go.Scatter(
                x=x, y=ys, mode="markers", name=name, legendgroup=name, showlegend=row == 1,
                marker=dict(size=9, color=color, line=dict(width=2, color="white")),
                customdata=labels,
                hovertemplate="“%{customdata}” · " + name + ": %{y}<extra></extra>",
            ), row=row, col=1)
        if key != "dur_ms":
            fig.add_hline(y=0, line=dict(color="#CFC9DE", width=1, dash="dot"), row=row, col=1)
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=labels, tickangle=-45, row=3, col=1)
    fig.update_annotations(font=dict(size=12, color=INK, family=FONT), x=0, xanchor="left")
    fig.update_layout(hovermode="closest")
    return _base(fig, height, legend=True)


def history_counts(counts: dict, height: int = 260) -> go.Figure:
    emos = [e for e in EMOTIONS if counts.get(e)]
    fig = go.Figure(go.Bar(
        x=[counts[e] for e in emos], y=emos, orientation="h",
        marker=dict(color=[EMOTION_COLORS[e] for e in emos], cornerradius=4),
        text=[counts[e] for e in emos], textposition="outside", textfont=dict(color=INK),
        hovertemplate="%{y}: %{x} recordings<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Recordings (top predicted class)", dtick=1)
    return _base(fig, height)


def history_timeline(rows: list[dict], height: int = 260) -> go.Figure:
    fig = go.Figure()
    for e in EMOTIONS:
        pts = [r for r in rows if r["top_emotion"] == e]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[r["created_at"] for r in pts], y=[r["confidence"] for r in pts], mode="markers", name=e,
            marker=dict(size=11, color=EMOTION_COLORS[e], line=dict(width=2, color="white")),
            text=[r["name"] for r in pts],
            hovertemplate="%{text}<br>%{x}<br>" + e + " %{y:.0%}<extra></extra>",
        ))
    fig.update_yaxes(title="Top-class probability", tickformat=".0%", range=[0, 1.05])
    return _base(fig, height, legend=True)
