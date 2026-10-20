"""Local analysis history: SQLite metadata + JSON results + 16 kHz WAV copies."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"          # 16 kHz mono copies used for analysis views
ORIGINAL_DIR = DATA_DIR / "original"    # untouched uploaded / recorded files
DB_PATH = DATA_DIR / "velora.db"

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


def _original_path(result: dict) -> Path:
    return ORIGINAL_DIR / f"{result['id']}.{result.get('original', {}).get('ext', 'wav')}"


def save(result: dict, y: np.ndarray, sr: int, original: bytes | None = None) -> None:
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
    with closing(_conn()) as con:
        rows = con.execute(
            "SELECT id, created_at, name, source, duration, top_emotion, confidence, transcript "
            "FROM analyses ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get(analysis_id: str) -> dict | None:
    with closing(_conn()) as con:
        row = con.execute("SELECT result_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def load_audio(analysis_id: str) -> tuple[np.ndarray, int] | None:
    p = AUDIO_DIR / f"{analysis_id}.wav"
    if not p.exists():
        return None
    y, sr = sf.read(p, dtype="float32")
    return y, sr


def load_original(result: dict) -> bytes | None:
    p = _original_path(result)
    return p.read_bytes() if p.exists() else None


def rename(analysis_id: str, name: str) -> None:
    res = get(analysis_id)
    if not res:
        return
    res["name"] = name
    with closing(_conn()) as con, con:
        con.execute("UPDATE analyses SET name=?, result_json=? WHERE id=?", (name, json.dumps(res), analysis_id))


def delete(analysis_id: str) -> None:
    with closing(_conn()) as con, con:
        con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    (AUDIO_DIR / f"{analysis_id}.wav").unlink(missing_ok=True)
    for p in ORIGINAL_DIR.glob(f"{analysis_id}.*") if ORIGINAL_DIR.exists() else []:
        p.unlink(missing_ok=True)


def clear() -> None:
    for row in list_all():
        delete(row["id"])
