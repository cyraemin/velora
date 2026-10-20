# Velora

**Your voice. Deeper insights.** Velora is a local Streamlit app for voice emotion recognition,
acoustic pattern analysis and expressive transcription. Everything runs on your own computer.

## Quick start (Windows)

The project already contains a configured virtual environment in `.venv`.

```powershell
.\run_velora.bat
# or
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501.

### Fresh setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use Python 3.13 rather than 3.14. librosa's numba dependency and the ML wheels are more reliable on 3.13.

## Pages

| Page | What it does |
|---|---|
| **Home** | Upload a file (WAV, MP3, FLAC, OGG, M4A, AAC, WebM or MP4, up to 25 MB) or record with the mic (record, stop, replay), then analyze. **Clear** resets the selection. The dashboard shows the waveform, spectrogram, MFCC, pitch and energy, the emotion timeline, the emotion card, acoustic insights and expressive text. |
| **Live Studio** | Guided prompts to read in different moods. Each take is analyzed automatically, with a mic check (level, SNR, clipping) and a table of takes comparing your target mood with the model's prediction. |
| **Analysis** | Full report for one recording, split into model outputs, measured features and interpretation, with all downloads (see below). |
| **History** | All saved analyses with search and filters, charts of predicted classes over time, and open, rename, delete and download actions. |
| **Contrast Lab** | Compare two recordings: measured differences, voice-profile radar, transcripts side by side, a word-by-word delivery comparison for shared words, emotion-model outputs and pitch contours. Download a Markdown report or a ZIP bundle. |
| **About** | Methods, model status, limitations and privacy. |

## Downloads

These are available from **Analysis** and from a selected row in **History**:

| Download | Contents |
|---|---|
| Original audio | The exact file you uploaded or recorded, byte for byte. Kept for analyses saved to History. |
| Transcript (.txt) | Whisper's text with timestamped segments and the expressive markup. If no transcript exists, the file says why. |
| Analysis (.json) | The complete result: statistics, model outputs, word timings, tracks and interpretation. |
| Acoustic metrics (.csv) | Every clip-level number with its unit and source (measured, normalised or model output). |
| Frame tracks (.csv) | Pitch, energy and spectral centroid every ~20 ms. |
| PDF report | A 2–3 page A4 report with emotion bars, transcript, key measurements, pitch and energy plots, interpretation and the full feature table. |
| Emotion timeline (.csv) | Per-window model probabilities. Only offered if the emotion model ran. |
| Analysed audio (.wav) | The 16 kHz mono signal that was analysed. |

Contrast Lab adds a comparison report (.md) and a ZIP bundle containing the report, JSON for both analyses, a word-level CSV and both audio files.

## Where every number comes from

Velora labels each output with one of three tags:

- **Measured** values are computed from the waveform with librosa and NumPy:
  - pitch with pYIN (65–500 Hz)
  - RMS energy
  - spectral centroid, rolloff, bandwidth and flatness
  - zero-crossing rate and MFCCs
  - pauses (silences of 250 ms or more inside the speaking span)
  - syllable-nucleus speaking rate, from peaks in a 250–2500 Hz energy envelope
- **Model output** is the raw output of a pretrained network, never edited or re-weighted:
  - **Emotion:** [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er), a wav2vec 2.0 base model fine-tuned on IEMOCAP. Its classes are neutral, happy, angry and sad. It runs on the whole clip and on sliding 2-second windows for the timeline.
  - **Speech-to-text:** faster-whisper (`base` or `tiny`, int8 on CPU) with word timestamps.
- **Interpretation** is fixed-rule text built from the two sources above: the summary paragraph, the insight notes, the expressive-transcript markup and the Contrast Lab comparison text.

If a model can't be loaded (offline on first run, a missing package, or an error), its section says so. Everything else keeps working. Velora never fills in a missing prediction or transcript.

### Expressive transcript markup

| Markup | Meaning |
|---|---|
| **bold** | The word is at least 3 dB louder than your median word. |
| *italic* | The word is at least 5 dB softer than your median word. |
| underline | The word is at least 3 semitones above your median pitch. |
| ↗ ↘ | Pitch rises or falls within the word. |
| … or ⏸ 1.2s | A measured pause before the word. |
| Chip before each line | The emotion model's top class for that span. |

## Model downloads

Models download once, on first use, into `%USERPROFILE%\.cache\huggingface`:

| Model | Size |
|---|---|
| Emotion model | ~380 MB |
| Whisper base | ~145 MB |
| Whisper tiny | ~75 MB |

After that, Velora works offline. Turn either model off in the sidebar under **Engine settings**.

## Data and privacy

Analyses are stored in `data/velora.db`. Original files go in `data/original/` and 16 kHz WAV copies in `data/audio/`. Delete them from **History** (deleting removes all three), or turn off saving in **Engine settings**. Audio never leaves your machine.

## Tests

Install the test tools first with `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.

```powershell
# unit + page tests (pipeline, decoding of every format, bad-file errors, exports incl. PDF, storage, Contrast Lab)
.\.venv\Scripts\python.exe -m pytest -q tests

# end-to-end in a real browser (Edge) against a running server, with a simulated microphone:
.\.venv\Scripts\python.exe tests\make_fixtures.py <speech.wav> <fixtures_dir>
.\.venv\Scripts\python.exe tests\e2e_browser.py <fixtures_dir> <speech.wav>
```

Set `VELORA_TEST_DIR` to a folder containing `speech_test.wav`, `speech_same_fast.wav` and `speech_other.wav` to test with real speech. Without it, the tests use a synthetic signal and skip the transcript checks.

## Troubleshooting

- **VS Code shows "Import could not be resolved".** The project's `.vscode/settings.json` points VS Code to `.venv`. If VS Code still uses another Python, run *Python: Select Interpreter* and choose `.venv\Scripts\python.exe`.
- **Code edits don't appear while the app is running.** Live reloading is turned off on purpose (`fileWatcherType = "none"` in `.streamlit/config.toml`). With it on, Streamlit's file watcher tried to import torchvision through transformers and flooded the terminal with tracebacks. Restart the app to pick up code changes.
- **Models.** Once downloaded, both models load from the local cache with no network calls. The automatic background safetensors conversion in transformers is disabled; it ran network calls alongside inference and crashed Python on Windows.
- **A model shows "Failed" in the sidebar.** Open **About**, read the error, and click **Retry loading the emotion model**. Measurements and exports keep working without the models.

## Limitations

- The emotion model has four classes and was trained on acted English speech. It can be wrong, especially for other languages, accents or noisy audio.
- Absolute loudness depends on the microphone and its gain.
- Jitter and shimmer are frame-level approximations.
- Velora is not a psychological or medical assessment.

## Project layout

```
app.py                 navigation + theme
views/                 one file per page
velora/audio.py        decoding, resampling, quality metrics
velora/features.py     librosa feature extraction
velora/emotion.py      pretrained emotion classifier
velora/transcribe.py   faster-whisper wrapper
velora/interpret.py    rule-based text + expressive transcript
velora/contrast.py     comparison logic + reports
velora/exports.py      TXT / JSON / CSV / PDF exports
velora/pipeline.py     end-to-end analysis
velora/storage.py      SQLite history
velora/charts.py       Plotly figures
velora/ui.py, dashboard.py   shared UI
```
