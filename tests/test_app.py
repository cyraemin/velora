"""Smoke tests: pipeline on synthetic/TTS audio and every page rendered headlessly.

Run:  python -m pytest -q tests   (or)   python tests/test_app.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

from velora import audio as A  # noqa: E402
from velora import pipeline as P  # noqa: E402

PAGES = ["views/home.py", "views/live_studio.py", "views/analysis.py", "views/history.py",
         "views/contrast_lab.py", "views/about.py"]
SAMPLE = os.environ.get("VELORA_TEST_WAV")


def _sample() -> tuple[bytes, str]:
    if SAMPLE and Path(SAMPLE).exists():
        return Path(SAMPLE).read_bytes(), Path(SAMPLE).name
    return A.to_wav_bytes(A.synthetic_demo(), A.TARGET_SR), "synthetic_demo.wav"


def test_pipeline_json_safe():
    data, name = _sample()
    res, y, sr = P.analyze(data, name, "test", use_transcript=bool(SAMPLE))
    json.dumps(res, allow_nan=False)
    assert res["stats"]["duration"] > 0
    assert res["interpretation"]


def test_pipeline_without_models():
    data, name = _sample()
    res, _, _ = P.analyze(data, name, "test", use_emotion=False, use_transcript=False)
    assert res["emotion"]["ok"] is False
    assert res["transcript"] is None
    assert res["insights"]


def _run_pages(preload: bool):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    if preload:
        data, name = _sample()
        res, y, sr = P.analyze(data, name, "test", use_transcript=bool(SAMPLE))
        at.session_state["_results"] = {res["id"]: res}
        at.session_state["_audio"] = {res["id"]: A.to_wav_bytes(y, sr)}
        at.session_state["current_id"] = res["id"]
        at.session_state["studio_result_id"] = res["id"]
        at.session_state["contrast_a"] = res["id"]
        at.session_state["contrast_b"] = res["id"]
    for p in PAGES:
        at.switch_page(p)
        at.run()
        assert not at.exception, f"{p}: {at.exception}"
        # with A = B preloaded, Contrast Lab must refuse the comparison (expected guard)
        errs = [e.value for e in at.error if "same analysis" not in e.value]
        assert not errs, f"{p}: {errs}"
        print(f"  ok  {p}")


def test_pages_empty():
    _run_pages(False)


def test_pages_with_result():
    _run_pages(True)


if __name__ == "__main__":
    for fn in [test_pipeline_json_safe, test_pipeline_without_models, test_pages_empty, test_pages_with_result]:
        print(fn.__name__)
        fn()
    print("ALL PASSED")

