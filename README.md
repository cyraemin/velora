<h1 align="center">
  <img src="assets/icon.svg" width="44" alt="" align="top"> Velora
</h1>

<p align="center">
  <b>Your voice. Deeper insights.</b><br>
  Speech emotion recognition, acoustic analysis and expressive transcription in the browser.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%20%7C%203.14-3776AB?logo=python&logoColor=white" alt="Python 3.13 | 3.14">
  <img src="https://img.shields.io/badge/streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit 1.66">
  <img src="https://img.shields.io/badge/runs%20on-CPU-5B47D6" alt="Runs on CPU">
</p>

![Velora dashboard](docs/screenshot-dashboard.webp)

## Features

- **Emotion recognition.** [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_base) estimates
  neutral, happy, sad, angry, fearful, disgusted or surprised for the whole clip and over time. Each result is
  marked *Clear*, *Likely* or *Uncertain*, depending on how far ahead the top emotion is.
- **Acoustic analysis.** Pitch, loudness, timbre, speaking rate and pauses, with waveform, spectrogram, MFCC and
  pitch views.
- **Expressive transcript.** Whisper transcribes the speech, and each word is marked by how it was said: louder,
  softer, higher, rising or falling, or after a pause.
- **Live Studio.** Prompts for recording the same kind of line in different moods, with a microphone check.
- **Contrast Lab.** Two recordings side by side, down to how individual shared words were delivered.
- **History and exports.** Saved analyses with search and charts. Every analysis exports as a PDF report, JSON,
  CSV, transcript or audio.

Every value is labelled by where it came from: **measured** from the signal, raw **model output**, or rule-based
**interpretation**. If a model is unavailable, Velora says so and never fills in a result.

## Getting started

Requires **Python 3.13 or 3.14**. No GPU or FFmpeg install is needed.

```bash
git clone https://github.com/cyraemin/velora.git
cd velora
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS and Linux), then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. The models (~0.5 GB) download during the first analysis, which takes a minute or
two. They are cached afterwards, so later runs work offline.

> **Windows:** run `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` first to get the
> smaller CPU-only PyTorch build. Afterwards you can start the app with `scripts\run_velora.bat`.

## Deployment

Velora runs on [Streamlit Community Cloud](https://share.streamlit.io) as is: choose **Create app**, select this
repository, branch `main` and main file `app.py`, and deploy.

When hosted, history is kept **in memory for each browser session only**. Visitors never see each other's
recordings, and nothing is written to disk. On other hosts, set `VELORA_STORAGE=session` to get the same behaviour.

The app needs no API keys. Any secrets added later belong in `.streamlit/secrets.toml` (git-ignored), not in code.

## Privacy

Run locally, audio never leaves your computer. History is saved in `data/`, which is git-ignored. You can delete it
from the **History** page or switch saving off in **Engine settings**. The only network access is the one-time model
download from Hugging Face.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The test suite covers the analysis pipeline, every supported audio format, invalid files, all exports, both storage
modes, Contrast Lab and every page. `tests/e2e/` holds a browser test that drives the running app in Microsoft Edge
with a simulated microphone.

## Project structure

```
├── app.py              Entry point and page navigation
├── velora/             Core package: audio, features, models, exports, storage, UI
├── views/              One module per page
├── tests/              pytest suite (tests/e2e/ holds the browser test)
├── assets/             Logo and icon
├── docs/               README images
└── scripts/            Windows launcher
```

## Models

| Task | Model |
|---|---|
| Emotion | [emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base), via [FunASR](https://github.com/modelscope/FunASR) |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `base` or `tiny` |

## Limitations

Emotion recognition is an estimate. Real feelings are often mixed or subtle, and results are less reliable for
very short, quiet or noisy recordings. Velora is not a psychological or medical assessment.
