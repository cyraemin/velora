# Velora

**Your voice. Deeper insights.**

Velora is a Streamlit app for voice emotion recognition, acoustic analysis and expressive transcription.
Upload or record speech and Velora measures pitch, energy, timbre and pace, runs a pretrained emotion model,
and transcribes what you said, marking up *how* each word was spoken.

Every number on screen is labelled with where it came from: **measured** from the waveform, **model output**,
or rule-based **interpretation**. Nothing is invented or filled in when a model is unavailable.

## Contents

- [Features](#features)
- [Getting started](#getting-started)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Privacy](#privacy)
- [Development](#development)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Features

| Page | What it does |
|---|---|
| **Home** | Upload a file (WAV, MP3, FLAC, OGG, M4A, AAC, WebM or MP4, up to 25 MB) or record with the microphone, then analyse it. The dashboard shows the waveform, spectrogram, MFCCs, pitch and energy, the emotion timeline, the emotion card, acoustic insights and the expressive transcript. |
| **Live Studio** | Guided prompts to read in different moods. Each take is analysed automatically, with a mic check (level, SNR, clipping) and a table comparing your target mood with the model's prediction. |
| **Analysis** | The full report for one recording, split into model outputs, measured features and interpretation, with every download. |
| **History** | All saved analyses with search, filters and charts, plus open, rename, delete and download actions. |
| **Contrast Lab** | Compare two recordings: measured differences, a voice-profile radar, transcripts side by side, word-by-word delivery for shared words, emotion outputs and pitch contours. |
| **About** | Methods, model status, limitations and privacy. |

### Downloads

Available from **Analysis** and from a selected row in **History**:

| Download | Contents |
|---|---|
| Original audio | The exact file you uploaded or recorded, byte for byte. |
| Transcript (.txt) | Whisper's text with timestamped segments and the expressive markup. |
| Analysis (.json) | The complete result: statistics, model outputs, word timings, tracks and interpretation. |
| Acoustic metrics (.csv) | Every clip-level number with its unit and source. |
| Frame tracks (.csv) | Pitch, energy and spectral centroid every ~20 ms. |
| PDF report | A 2–3 page A4 report with charts, transcript, key measurements and the full feature table. |
| Emotion timeline (.csv) | Per-window model probabilities (only when the emotion model ran). |
| Analysed audio (.wav) | The 16 kHz mono signal that was analysed. |

Contrast Lab adds a Markdown comparison report and a ZIP bundle containing the report, both analyses as JSON,
a word-level CSV and both audio files.

## Getting started

### Requirements

- **Python 3.13.** Use 3.13 rather than 3.14; librosa's numba dependency and the ML wheels are more reliable on it.
- About 1 GB of free disk space for PyTorch and the models.
- No GPU and no system FFmpeg are needed: PyAV bundles its own decoder.

### Install

```bash
git clone https://github.com/cyraemin/velora.git
cd velora
python -m venv .venv
```

Activate the environment, then install the dependencies:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, installing the CPU build of PyTorch first avoids the much larger CUDA download.
On Linux, `requirements.txt` already pins the CPU wheel.

### Run

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. On Windows you can also double-click `run_velora.bat`.

The emotion model (~380 MB) and Whisper (~75–145 MB) download once, on first use, into the Hugging Face cache
(`~/.cache/huggingface`). After that, Velora works offline. Either model can be switched off under
**Engine settings** in the sidebar.

## Deployment

Velora is set up for [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the repository to GitHub.
2. In Community Cloud choose **Create app → Deploy a public app from GitHub** and select the repository,
   branch `main` and main file `app.py`.
3. Under **Advanced settings**, set Python to **3.13**. The Linux PyTorch wheel in `requirements.txt` is built for it.
4. Deploy. The first build installs PyTorch, and the first analysis downloads the models, so allow a few minutes.

When hosted, analysis history is kept **per browser session, in memory only**. Visitors never see each other's
recordings, and nothing is written to the server's disk. Community Cloud is detected automatically. On any other
host, set `VELORA_STORAGE=session` to get the same behaviour.

> **Resources:** the two models plus PyTorch need roughly 1.5 GB of RAM. If the hosted app restarts during an
> analysis, it has hit the platform's memory limit. Turning off transcription in **Engine settings** reduces usage.

Velora needs no API keys or secrets. If you add any later, keep them in `.streamlit/secrets.toml` locally
(it is git-ignored) and in the app's **Settings → Secrets** panel on Community Cloud. Never commit them.

## Configuration

| Setting | Where | Default | Purpose |
|---|---|---|---|
| `VELORA_STORAGE` | Environment variable | unset (local disk) | `session` keeps history in memory per browser session. |
| `maxUploadSize` | `.streamlit/config.toml` | `25` (MB) | Largest accepted upload. |
| `fileWatcherType` | `.streamlit/config.toml` | `none` | Live reloading is off on purpose (see [Troubleshooting](#troubleshooting)). |
| `showErrorDetails` | `.streamlit/config.toml` | `type` | Browsers see only the error type; full tracebacks go to the server log. |
| Emotion model, Whisper size, language | Sidebar → **Engine settings** | on, `base`, auto | Per-session analysis options. |

Clips longer than 3 minutes are truncated, and clips shorter than 0.5 s are rejected.

## How it works

Velora labels each output with one of three tags.

**Measured**: computed from the waveform with librosa and NumPy:

- pitch with pYIN (65–500 Hz)
- RMS energy
- spectral centroid, rolloff, bandwidth and flatness
- zero-crossing rate and MFCCs
- pauses (silences of 250 ms or more inside the speaking span)
- syllable-nucleus speaking rate, from peaks in a 250–2500 Hz energy envelope

**Model output**: the raw output of a pretrained network, never edited or re-weighted:

- **Emotion:** [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er), a
  wav2vec 2.0 base model fine-tuned on IEMOCAP. Its classes are neutral, happy, angry and sad. It runs on the whole
  clip and on sliding 2-second windows for the timeline.
- **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base` or `tiny`, int8 on CPU)
  with word timestamps.

**Interpretation**: fixed-rule text built from the two sources above: the summary paragraph, insight notes,
expressive markup and the Contrast Lab comparison text.

If a model can't be loaded (offline on first run, a missing package, or an error), its section says so and
everything else keeps working.

### Expressive transcript markup

| Markup | Meaning |
|---|---|
| **bold** | The word is at least 3 dB louder than your median word. |
| *italic* | The word is at least 5 dB softer than your median word. |
| underline | The word is at least 3 semitones above your median pitch. |
| ↗ ↘ | Pitch rises or falls within the word. |
| … or ⏸ 1.2s | A measured pause before the word. |
| Chip before each line | The emotion model's top class for that span. |

## Privacy

- **Running locally:** audio never leaves your machine. Analyses are stored in `data/velora.db`, original files in
  `data/original/` and 16 kHz copies in `data/audio/`. Delete them from **History**, or turn off saving in
  **Engine settings**. The `data/` folder is git-ignored, so recordings are never committed.
- **Hosted:** audio is processed on the server, history lives only in your browser session, and nothing is stored
  after you leave.
- The only outbound network access is the one-time model download from Hugging Face.

## Development

Install the development tools:

```bash
python -m pip install -r requirements-dev.txt
```

Lint and test:

```bash
ruff check .
pytest -q
```

The test suite covers the analysis pipeline, decoding of every supported format, bad-file errors, every export
(including the PDF), both storage modes, Contrast Lab, and every page rendered headlessly. Set `VELORA_TEST_DIR`
to a folder containing `speech_test.wav`, `speech_same_fast.wav` and `speech_other.wav` to test with real speech.
Without it, the tests use a synthetic signal and skip the transcript checks.

End-to-end test in a real browser (Microsoft Edge), against a running server, with a simulated microphone:

```bash
python tests/make_fixtures.py <speech.wav> <fixtures_dir>
python tests/e2e_browser.py <fixtures_dir> <speech.wav> [base_url]
```

Linting rules live in `pyproject.toml`.

## Project structure

```
velora/
├── app.py                  Entry point: page navigation and theme
├── views/                  One file per page
│   ├── home.py
│   ├── live_studio.py
│   ├── analysis.py
│   ├── history.py
│   ├── contrast_lab.py
│   └── about.py
├── velora/                 Core package
│   ├── audio.py            Decoding, resampling, recording-quality metrics
│   ├── features.py         librosa feature extraction
│   ├── emotion.py          Pretrained emotion classifier
│   ├── transcribe.py       faster-whisper wrapper
│   ├── interpret.py        Rule-based text and expressive transcript
│   ├── contrast.py         Comparison logic and reports
│   ├── exports.py          TXT / JSON / CSV / PDF exports
│   ├── pipeline.py         End-to-end analysis
│   ├── storage.py          History: SQLite locally, per-session when hosted
│   ├── charts.py           Plotly figures
│   ├── dashboard.py        Shared result sections
│   └── ui.py               Theme, sidebar and shared widgets
├── tests/                  pytest suite, fixture generator, browser test
├── assets/                 Logo and icon
├── .streamlit/config.toml  Streamlit theme and server settings
├── requirements.txt        Runtime dependencies (used by Community Cloud)
├── requirements-dev.txt    Test and lint tools
├── pyproject.toml          Project metadata, ruff and pytest configuration
└── run_velora.bat          Windows launcher
```

## Troubleshooting

- **VS Code shows "Import could not be resolved".** `.vscode/settings.json` points VS Code to `.venv`. If it still
  uses another interpreter, run *Python: Select Interpreter* and choose the one in `.venv`.
- **Code edits don't appear while the app is running.** Live reloading is off on purpose
  (`fileWatcherType = "none"`). With it on, Streamlit's file watcher tried to import torchvision through
  transformers and flooded the terminal with tracebacks. Restart the app to pick up changes.
- **A model shows "Failed" in the sidebar.** Open **About**, read the error and click
  **Retry loading the emotion model**. Measurements and exports keep working without the models.
- **Models and the network.** Once downloaded, both models load from the local cache without network calls.
  The automatic background safetensors conversion in transformers is disabled, because it ran network calls
  alongside inference and crashed Python on Windows.

## Limitations

- The emotion model has four classes and was trained on acted English speech. It can be wrong, especially for
  other languages, accents or noisy audio.
- Absolute loudness depends on the microphone and its gain.
- Jitter and shimmer are frame-level approximations, not clinical voice measures.
- Velora is not a psychological or medical assessment.
