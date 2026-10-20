"""Contrast Lab tests: comparison logic, edge cases, exports and the page itself.

Uses WAV files from VELORA_TEST_DIR (speech_test.wav, speech_same_fast.wav, speech_other.wav)
when available; otherwise falls back to synthetic audio (no transcript).
"""
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

from velora import audio as A  # noqa: E402
from velora import contrast as X  # noqa: E402
from velora import pipeline as P  # noqa: E402

DIR = Path(os.environ.get("VELORA_TEST_DIR", ""))
_cache = {}


def _analyze(fname, **kw):
    key = (fname, tuple(sorted(kw.items())))
    if key not in _cache:
        p = DIR / fname
        if DIR.name and p.exists():
            data, name = p.read_bytes(), fname
        else:
            data, name = A.to_wav_bytes(A.synthetic_demo(), A.TARGET_SR), "synthetic_demo.wav"
            kw.setdefault("use_transcript", False)
        _cache[key] = P.analyze(data, name, "test", **kw)
    return _cache[key]


def test_same_words_alignment():
    ra, _, _ = _analyze("speech_test.wav")
    rb, _, _ = _analyze("speech_same_fast.wav")
    cmp = X.compare(ra, rb)
    json.dumps(cmp, allow_nan=False)
    if ra.get("transcript") and rb.get("transcript"):
        al = cmp["alignment"]
        assert al["available"] and al["status"] == "same", al["similarity"]
        assert len(al["pairs"]) >= 15
        assert cmp["delivery_notes"]
        print("  same-words notes:", *cmp["delivery_notes"], sep="\n    ")
    assert cmp["summary"]


def test_different_words():
    ra, _, _ = _analyze("speech_test.wav")
    rb, _, _ = _analyze("speech_other.wav")
    cmp = X.compare(ra, rb)
    if ra.get("transcript") and rb.get("transcript"):
        assert cmp["alignment"]["status"] == "different"
        assert cmp["delivery_notes"] == []


def test_missing_model_outputs_and_same_recording():
    ra, _, _ = _analyze("speech_test.wav")
    rb, _, _ = _analyze("speech_same_fast.wav", use_emotion=False, use_transcript=False)
    cmp = X.compare(ra, rb)
    msgs = " ".join(m for _, m in cmp["checks"])
    assert "No emotion-model output for recording B" in msgs
    # with synthetic fallback audio A has no transcript either ("... recording A and B")
    assert re.search(r"No transcript for recording (A and )?B", msgs)
    assert cmp["emotion"]["b"] is None and not cmp["alignment"]["available"]
    md = X.report_markdown(cmp, ra, rb)
    assert "| Happy |" in md and "—" in md
    same = X.compare(ra, ra)
    assert same["checks"][0][0] == "error"


def test_bundle():
    ra, ya, sr = _analyze("speech_test.wav")
    rb, yb, _ = _analyze("speech_same_fast.wav")
    cmp = X.compare(ra, rb)
    z = zipfile.ZipFile(io.BytesIO(X.report_bundle(cmp, ra, rb, A.to_wav_bytes(ya, sr), A.to_wav_bytes(yb, sr))))
    names = z.namelist()
    assert "comparison_report.md" in names and "comparison.json" in names
    assert sum(n.endswith(".wav") for n in names) == 2
    print("  bundle:", names)


def test_page_renders():
    ra, ya, sr = _analyze("speech_test.wav")
    rb, yb, _ = _analyze("speech_same_fast.wav")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    at.session_state["_results"] = {ra["id"]: ra, rb["id"]: rb}
    at.session_state["_audio"] = {ra["id"]: A.to_wav_bytes(ya, sr), rb["id"]: A.to_wav_bytes(yb, sr)}
    at.session_state["contrast_a"] = ra["id"]
    at.session_state["contrast_b"] = rb["id"]
    at.switch_page("views/contrast_lab.py")
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    # swap button works
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["contrast_a"] == rb["id"]


if __name__ == "__main__":
    for fn in [test_same_words_alignment, test_different_words, test_missing_model_outputs_and_same_recording,
               test_bundle, test_page_renders]:
        print(fn.__name__)
        fn()
    print("ALL PASSED")
