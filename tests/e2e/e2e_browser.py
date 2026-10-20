"""End-to-end browser test against a running Velora server (default http://localhost:8501).

Requires: pip install playwright  (uses the installed Microsoft Edge; no browser download)
Usage:    python tests/e2e/e2e_browser.py <fixtures_dir> <speech.wav> [base_url]

The microphone is simulated with Chromium's fake-media flags, which feed <speech.wav>
into getUserMedia — so the real recorder widget is exercised.
"""
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

FX = Path(sys.argv[1])
SPEECH = Path(sys.argv[2])
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8501"
OUT = FX / "e2e_out"
OUT.mkdir(exist_ok=True)
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def no_exceptions(page, where):
    n = page.locator("[data-testid='stException']").count()
    check(f"{where}: no Python exceptions", n == 0,
          "" if n == 0 else page.locator("[data-testid='stException']").first.inner_text()[:300])


def idle(page, timeout=300):
    """Wait until Streamlit finished the current script run."""
    t0 = time.time()
    time.sleep(1.0)
    while time.time() - t0 < timeout:
        if page.locator("[data-testid='stStatusWidget']").count() == 0 and \
                page.locator("[data-testid='stProgress']").count() == 0:
            time.sleep(0.8)
            if page.locator("[data-testid='stStatusWidget']").count() == 0:
                return
        time.sleep(0.5)


def goto(page, path):
    page.goto(f"{BASE}/{path}")
    page.wait_for_selector(".vl-hero-title", timeout=120000)
    idle(page)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={SPEECH}",
    ])
    ctx = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True,
                              permissions=["microphone"])
    page = ctx.new_page()

    # ---------------------------------------------------------------- startup
    goto(page, "")
    check("startup: home renders", page.get_by_text("Analyse a voice recording").is_visible())
    check("startup: Analyse disabled before input", page.get_by_role("button", name="Analyse voice").is_disabled())
    no_exceptions(page, "home")

    # ---------------------------------------------------------------- error handling
    page.locator("[data-testid='stFileUploader'] input[type=file]").set_input_files(str(FX / "fx_corrupt.wav"))
    idle(page)
    page.get_by_role("button", name="Analyse voice").click()
    idle(page)
    err = page.locator("[data-testid='stAlert']").filter(has_text="Couldn't analyse")
    check("error: corrupt file shows a clear message", err.count() > 0,
          err.first.inner_text()[:160] if err.count() else "no alert")
    no_exceptions(page, "home after corrupt upload")
    page.get_by_role("button", name="Clear").click()
    idle(page)
    check("clear: selection removed", page.get_by_text("Ready to analyse").count() == 0
          and page.get_by_role("button", name="Analyse voice").is_disabled())

    # ---------------------------------------------------------------- upload + analysis (m4a)
    page.locator("[data-testid='stFileUploader'] input[type=file]").set_input_files(str(FX / "fx.m4a"))
    idle(page)
    check("upload: file shown as ready", page.get_by_text("Ready to analyse").count() > 0)
    t0 = time.time()
    page.get_by_role("button", name="Analyse voice").click()
    page.wait_for_selector(".vl-emo, .vl-note", timeout=600000)
    idle(page, 600)
    check("analysis: dashboard rendered for m4a upload", page.get_by_text("Key measurements").count() > 0,
          f"{time.time() - t0:.0f}s incl. model loading")
    prob = page.get_by_text("Model probability").count()
    check("analysis: emotion model output shown", prob > 0)
    check("analysis: transcript markup shown", page.locator(".vl-seg").count() > 0)
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / "home_result.png"), full_page=True)
    no_exceptions(page, "home result")

    # ---------------------------------------------------------------- downloads on Analysis
    goto(page, "analysis")
    page.get_by_text("Downloads").first.scroll_into_view_if_needed()
    expected = {
        "Original audio (.m4a)": lambda b: b == (FX / "fx.m4a").read_bytes(),
        "Transcript (.txt)": lambda b: b"== Full text ==" in b,
        "Analysis (.json)": lambda b: "stats" in json.loads(b),
        "Acoustic metrics (.csv)": lambda b: next(csv.reader(io.StringIO(b.decode())))[:3] == ["section", "metric", "value"],
        "Frame tracks (.csv)": lambda b: b.startswith(b"time_s,f0_hz"),
        "PDF report": lambda b: b[:5] == b"%PDF-" and len(b) > 20000,
        "Emotion timeline (.csv)": lambda b: b.startswith(b"start_s,end_s"),
        "Analysed audio (16 kHz .wav)": lambda b: b[:4] == b"RIFF",
    }
    for label, ok in expected.items():
        btn = page.get_by_role("button", name=label)
        if btn.count() == 0:
            check(f"download: {label}", False, "button not found")
            continue
        with page.expect_download(timeout=60000) as dl:
            btn.first.click()
        path = OUT / dl.value.suggested_filename
        dl.value.save_as(path)
        data = path.read_bytes()
        check(f"download: {label}", ok(data), f"{dl.value.suggested_filename} ({len(data):,} bytes)")
        idle(page)
    no_exceptions(page, "analysis")
    page.screenshot(path=str(OUT / "analysis.png"), full_page=True)

    # ---------------------------------------------------------------- microphone (Live Studio)
    goto(page, "live_studio")
    rec_btn = page.locator("[data-testid='stAudioInput'] button").first
    rec_btn.click()
    page.wait_for_timeout(7000)
    page.locator("[data-testid='stAudioInput'] button").first.click()   # stop
    try:
        page.wait_for_selector("text=Mic check", timeout=600000)
        idle(page, 600)
        check("mic: recording captured and analysed automatically", True)
        check("mic: take added to session table", page.get_by_text("Session takes").count() > 0)
        check("mic: replay player present", page.locator("[data-testid='stAudioInput'] audio, "
                                                         "[data-testid='stAudioInput'] [data-testid*='Waveform'], "
                                                         "[data-testid='stAudioInput'] button").count() > 0)
    except Exception as exc:
        check("mic: recording captured and analysed automatically", False, str(exc)[:200])
    page.screenshot(path=str(OUT / "live_studio.png"), full_page=True)
    no_exceptions(page, "live studio")

    # ---------------------------------------------------------------- history + contrast
    goto(page, "history")
    check("history: saved analyses listed", page.locator("[data-testid='stMetricValue']").first.inner_text().strip() == "2",
          page.locator("[data-testid='stMetricValue']").first.inner_text())
    no_exceptions(page, "history")

    goto(page, "contrast_lab")
    check("contrast: comparison rendered", page.get_by_text("Measured differences").count() > 0)
    page.get_by_role("tab", name="Same words, different delivery").click()
    idle(page)
    no_exceptions(page, "contrast lab")
    for label in ("Report (.md)", "Full bundle (.zip)"):
        with page.expect_download(timeout=60000) as dl:
            page.get_by_role("button", name=label).click()
        path = OUT / dl.value.suggested_filename
        dl.value.save_as(path)
        ok = path.read_bytes().startswith(b"# Velora comparison") if label.startswith("Report") else \
            "comparison_report.md" in zipfile.ZipFile(path).namelist()
        check(f"contrast download: {label}", ok, f"{path.name} ({path.stat().st_size:,} bytes)")
    page.screenshot(path=str(OUT / "contrast.png"), full_page=True)

    for path in ("about",):
        goto(page, path)
        no_exceptions(page, path)
    browser.close()

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
