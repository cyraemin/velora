"""Exports, decoding of every advertised format, error handling and storage round-trips.

Uses VELORA_TEST_DIR/speech_test.wav (real speech) when available, else a synthetic signal.
"""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from velora import audio as A  # noqa: E402
from velora import exports as X  # noqa: E402
from velora import pipeline as P  # noqa: E402
from velora import storage as S  # noqa: E402

DIR = Path(os.environ.get("VELORA_TEST_DIR", ""))
SPEECH = DIR / "speech_test.wav" if DIR.name else None


@pytest.fixture(scope="module")
def fixtures_dir():
    import make_fixtures

    out = Path(tempfile.mkdtemp(prefix="velora_fx_"))
    src = SPEECH if SPEECH and SPEECH.exists() else out / "src.wav"
    if src == out / "src.wav":
        import soundfile as sf
        sf.write(src, A.synthetic_demo(), A.TARGET_SR)
    make_fixtures.main(str(src), str(out))
    return out


@pytest.fixture(scope="module")
def analysed():
    if SPEECH and SPEECH.exists():
        data, name, transcript = SPEECH.read_bytes(), SPEECH.name, True
    else:
        data, name, transcript = A.to_wav_bytes(A.synthetic_demo(), A.TARGET_SR), "synthetic.wav", False
    res, y, sr = P.analyze(data, name, "test", use_transcript=transcript)
    return res, y, sr, data


@pytest.mark.parametrize("fname", ["fx.wav", "fx.flac", "fx.ogg", "fx.mp3", "fx.m4a", "fx.webm", "fx_stereo_44k.wav"])
def test_decode_supported_formats(fixtures_dir, fname):
    y, sr, _ = A.load_audio((fixtures_dir / fname).read_bytes(), fname)
    assert sr == A.TARGET_SR and len(y) / sr > 5


@pytest.mark.parametrize("fname,msg", [
    ("fx_empty.wav", "empty"), ("fx_corrupt.wav", "damaged"),
    ("fx_silence.wav", "silent"), ("fx_short.wav", "too short"),
])
def test_bad_audio_rejected_cleanly(fixtures_dir, fname, msg):
    with pytest.raises(A.AudioError, match=msg):
        A.load_audio((fixtures_dir / fname).read_bytes(), fname)


def test_json_export(analysed):
    res = analysed[0]
    back = json.loads(X.analysis_json(res))
    assert back["id"] == res["id"] and back["original"]["filename"] == res["name"]


def test_transcript_txt(analysed):
    res = analysed[0]
    txt = X.transcript_txt(res)
    if res.get("transcript"):
        assert res["transcript"]["text"] in txt and "== Segments ==" in txt
    else:
        assert "No transcript available" in txt


def test_transcript_txt_without_transcript(analysed):
    res = dict(analysed[0], transcript=None, transcript_error="Transcription disabled in settings.")
    assert "No transcript available: Transcription disabled" in X.transcript_txt(res)


def test_metrics_csv(analysed):
    res = analysed[0]
    rows = list(csv.DictReader(io.StringIO(X.metrics_csv(res))))
    metrics = {r["metric"]: r for r in rows}
    assert float(metrics["Pitch median"]["value"]) == res["stats"]["pitch_median"]
    assert metrics["Energy (RMS)"]["source"].startswith("measured")
    if res["emotion"]["ok"]:
        top = res["emotion"]["top"]
        assert float(metrics[f"P({top})"]["value"]) == res["emotion"]["probs"][top]
        assert metrics[f"P({top})"]["source"].startswith("model output")


def test_tracks_and_timeline_csv(analysed):
    res = analysed[0]
    rows = list(csv.reader(io.StringIO(X.tracks_csv(res))))
    assert rows[0][0] == "time_s" and len(rows) == len(res["tracks"]["t"]) + 1
    tl = X.timeline_csv(res)
    assert (tl is None) == (not res["emotion"]["ok"])


def test_pdf_report(analysed):
    pypdf = pytest.importorskip("pypdf")
    res = analysed[0]
    pdf = X.pdf_report(res)
    assert pdf[:5] == b"%PDF-"
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert len(reader.pages) >= 2
    assert "Key measurements" in text and "All measured features" in text
    if res["emotion"]["ok"]:
        assert f"Top class: {res['emotion']['top']}" in text
    if res.get("transcript"):
        assert res["transcript"]["text"].split()[0] in text


def test_pdf_without_model_outputs(analysed):
    res = dict(analysed[0], emotion={"ok": False, "error": "disabled"}, transcript=None, transcript_error="off")
    assert X.pdf_report(res)[:5] == b"%PDF-"


def test_storage_roundtrip_with_original(analysed, monkeypatch, tmp_path):
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)
    monkeypatch.setattr(S, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(S, "ORIGINAL_DIR", tmp_path / "original")
    monkeypatch.setattr(S, "DB_PATH", tmp_path / "t.db")
    res, y, sr, data = analysed
    S.save(res, y, sr, original=data)
    assert S.load_original(res) == data
    assert S.get(res["id"])["id"] == res["id"]
    S.delete(res["id"])
    assert S.load_original(res) is None and S.list_all() == []


def test_session_storage_roundtrip(analysed, monkeypatch, tmp_path):
    import streamlit as st

    monkeypatch.setattr(S, "SESSION_ONLY", True)
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)  # nothing may be written here
    monkeypatch.setattr(st, "session_state", {})
    res, y, sr, data = analysed
    S.save(res, y, sr, original=data)
    assert [r["id"] for r in S.list_all()] == [res["id"]]
    assert S.load_original(res) == data and S.load_audio(res["id"])[1] == sr
    S.rename(res["id"], "renamed")
    assert S.get(res["id"])["name"] == "renamed" and S.list_all()[0]["name"] == "renamed"
    S.delete(res["id"])
    assert S.list_all() == [] and not any(tmp_path.iterdir())
