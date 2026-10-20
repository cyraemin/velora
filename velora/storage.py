"""Analysis history: SQLite metadata + JSON results + 16 kHz WAV copies.

When deployed (Streamlit Community Cloud, or VELORA_STORAGE=session) history lives only in the
visitor's Streamlit session, so nobody can see anyone else's recordings and nothing is written to disk.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"          # 16 kHz mono copies used for analysis views
ORIGINAL_DIR = DATA_DIR / "original"    # untouched uploaded / recorded files
DB_PATH = DATA_DIR / "velora.db"

# Community Cloud checks out apps under /mount/src
SESSION_ONLY = os.environ.get("VELORA_STORAGE") == "session" or str(DATA_DIR).startswith("/mount/src/")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    duration REAL,
    top_emotion TEXT,
    confidence REAL,
    transcript TEXT,
    result_json TEXT NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def _mem() -> dict:
    """Per-visitor history: {id: {"result", "audio": (y, sr), "original"}}."""
    import streamlit as st

    return st.session_state.setdefault("_velora_history", {})


def _original_path(result: dict) -> Path:
    return ORIGINAL_DIR / f"{result['id']}.{result.get('original', {}).get('ext', 'wav')}"


def save(result: dict, y: np.ndarray, sr: int, original: bytes | None = None) -> None:
    if SESSION_ONLY:
        _mem()[result["id"]] = {"result": json.loads(json.dumps(result)), "audio": (y, sr), "original": original}
        return
    emo = result.get("emotion") or {}
    tr = result.get("transcript") or {}
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(AUDIO_DIR / f"{result['id']}.wav", y, sr, subtype="PCM_16")
    if original is not None:
        ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
        _original_path(result).write_bytes(original)
    with closing(_conn()) as con, con:
        con.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?)",
            (
                result["id"], result["created_at"], result["name"], result["source"],
                result["stats"]["duration"],
                emo.get("top") if emo.get("ok") else None,
                emo.get("confidence") if emo.get("ok") else None,
                tr.get("text"),
                json.dumps(result),
            ),
        )


def list_all() -> list[dict]:
    if SESSION_ONLY:
        rows = []
        for e in _mem().values():
            r, emo, tr = e["result"], e["result"].get("emotion") or {}, e["result"].get("transcript") or {}
            rows.append({
                "id": r["id"], "created_at": r["created_at"], "name": r["name"], "source": r["source"],
                "duration": r["stats"]["duration"],
                "top_emotion": emo.get("top") if emo.get("ok") else None,
                "confidence": emo.get("confidence") if emo.get("ok") else None,
                "transcript": tr.get("text"),
            })
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)
    with closing(_conn()) as con:
        rows = con.execute(
            "SELECT id, created_at, name, source, duration, top_emotion, confidence, transcript "
            "FROM analyses ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get(analysis_id: str) -> dict | None:
    if SESSION_ONLY:
        e = _mem().get(analysis_id)
        return json.loads(json.dumps(e["result"])) if e else None
    with closing(_conn()) as con:
        row = con.execute("SELECT result_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def load_audio(analysis_id: str) -> tuple[np.ndarray, int] | None:
    if SESSION_ONLY:
        e = _mem().get(analysis_id)
        return e["audio"] if e else None
    p = AUDIO_DIR / f"{analysis_id}.wav"
    if not p.exists():
        return None
    y, sr = sf.read(p, dtype="float32")
    return y, sr


def load_original(result: dict) -> bytes | None:
    if SESSION_ONLY:
        e = _mem().get(result["id"])
        return e["original"] if e else None
    p = _original_path(result)
    return p.read_bytes() if p.exists() else None


def rename(analysis_id: str, name: str) -> None:
    if SESSION_ONLY:
        if e := _mem().get(analysis_id):
            e["result"]["name"] = name
        return
    res = get(analysis_id)
    if not res:
        return
    res["name"] = name
    with closing(_conn()) as con, con:
        con.execute("UPDATE analyses SET name=?, result_json=? WHERE id=?", (name, json.dumps(res), analysis_id))


def delete(analysis_id: str) -> None:
    if SESSION_ONLY:
        _mem().pop(analysis_id, None)
        return
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    (AUDIO_DIR / f"{analysis_id}.wav").unlink(missing_ok=True)
    for p in ORIGINAL_DIR.glob(f"{analysis_id}.*") if ORIGINAL_DIR.exists() else []:
        p.unlink(missing_ok=True)


def clear() -> None:
    for row in list_all():
        delete(row["id"])
