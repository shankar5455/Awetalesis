# Real-Time Speech-to-Speech Translation System (S2ST)

An end-to-end pipeline that captures live microphone input, transcribes it,
translates it into a target language, and plays back synthesised speech – all
in near real time.

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Clone and enter the repo
git clone https://github.com/shankar5455/Awetalesis.git
cd Awetalesis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Launch the Streamlit web UI  ← easiest option
streamlit run app_ui.py

# 4b. OR run in the terminal
python main.py

# 4c. OR start the REST/WebSocket API server
python main.py --api --port 8000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    S2ST Pipeline                                     │
│                                                                      │
│  ┌──────────┐   ┌──────────────────┐   ┌────────────────────────┐   │
│  │Microphone│──▶│ Noise Suppression│──▶│   Voice Activity       │   │
│  │  Input   │   │   (noisereduce / │   │   Detection            │   │
│  │(sounddev)│   │  DeepFilterNet / │   │   (Silero VAD)         │   │
│  └──────────┘   │    RNNoise)      │   └──────────┬─────────────┘   │
│                 └──────────────────┘              │ speech segments  │
│                                                   ▼                  │
│  ┌──────────────────────┐   ┌────────────────────────────────────┐  │
│  │  Language Detection  │◀──│   ASR – Faster-Whisper             │  │
│  │  (Whisper built-in / │   │   (tiny / base / small / medium /  │  │
│  │   langdetect /       │   │    large-v2)                       │  │
│  │   fastText)          │   └────────────────────────────────────┘  │
│  └──────────┬───────────┘                                           │
│             │ (lang, text)                                           │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │  Translation         │                                           │
│  │  (MarianMT /         │                                           │
│  │   Google Translate / │                                           │
│  │   SeamlessM4T)       │                                           │
│  └──────────┬───────────┘                                           │
│             │ translated text                                        │
│             ▼                                                        │
│  ┌──────────────────────┐   ┌──────────────────────┐               │
│  │  TTS Synthesis       │──▶│  Audio Output        │               │
│  │  (gTTS / pyttsx3 /   │   │  (sounddevice)       │               │
│  │   Coqui TTS /        │   └──────────────────────┘               │
│  │   ElevenLabs)        │                                           │
│  └──────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**FastAPI layer** wraps the pipeline and provides REST + WebSocket endpoints
so a browser-based (or any HTTP) client can start/stop the pipeline and
receive live translation events.

---

## Project Structure

```
s2st-project/
│
├── main.py                  # CLI entry point
├── app_ui.py                # Streamlit web UI (run with: streamlit run app_ui.py)
├── config.py                # Centralised configuration (dataclasses)
├── requirements.txt
├── README.md
│
├── audio/
│   ├── __init__.py
│   ├── buffer.py            # Thread-safe ring buffer
│   └── stream.py            # Microphone capture (sounddevice)
│
├── processing/
│   ├── __init__.py
│   ├── noise_suppression.py # noisereduce / DeepFilterNet / RNNoise
│   ├── vad.py               # Silero VAD
│   ├── lid.py               # Language identification
│   ├── asr.py               # Faster-Whisper ASR
│   ├── translation.py       # MarianMT / Google / SeamlessM4T
│   └── tts.py               # gTTS / pyttsx3 / Coqui / ElevenLabs
│
├── pipeline/
│   ├── __init__.py
│   └── pipeline.py          # Wires all stages together
│
├── api/
│   ├── __init__.py
│   └── app.py               # FastAPI REST + WebSocket server
│
├── utils/
│   ├── __init__.py
│   └── logger.py            # Logging setup
│
└── tests/
    ├── __init__.py
    ├── test_audio_buffer.py
    ├── test_config.py
    ├── test_noise_suppression.py
    ├── test_vad.py
    ├── test_translation.py
    └── test_api.py
```

---

## Requirements

- Python 3.9+
- `ffmpeg` on the system PATH (used by pydub / gTTS audio decode).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/shankar5455/Awetalesis.git
cd Awetalesis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchaudio` are required for Silero VAD.  If you
> have a CUDA GPU, replace the generic `torch` wheel with the CUDA variant
> from https://pytorch.org.

### 4. Run the Streamlit UI (recommended)

```bash
streamlit run app_ui.py
```

A browser window will open automatically at **http://localhost:8501**.

**Step-by-step UI walkthrough:**

| Step | What to do | What you will see |
|------|-----------|-------------------|
| 1 | Open http://localhost:8501 in your browser | The app loads with a sidebar on the left |
| 2 | In the sidebar, choose a **Target language** (e.g. French) | Dropdown shows language names with BCP-47 codes |
| 3 | Pick an **ASR model** size and **TTS backend** | Settings locked once pipeline starts |
| 4 | Click **▶ Start** | A spinner appears; status badge changes to 🟢 **LIVE** |
| 5 | Speak into your microphone | Translation cards appear in the feed (newest first) |
| 6 | Click **⏹ Stop** | Badge changes to 🔴 **STOPPED**; microphone released |

**What the Translation Feed looks like:**

```
┌─────────────────────────────────────────────────────────────────┐
│  🗣 [EN] — heard              →    🔊 [FR] — translated         │
│  "Hello, how are you?"             "Bonjour, comment allez-vous?"│
│  Audio: 1200 ms  |  Processing: 840 ms                          │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  🗣 [EN] — heard              →    🔊 [FR] — translated         │
│  "The weather is nice today."      "Le temps est beau aujourd'hui."│
│  Audio: 980 ms  |  Processing: 620 ms                           │
└─────────────────────────────────────────────────────────────────┘
```

> The page auto-refreshes every second while the pipeline is running —
> you do **not** need to manually reload.

### 5. Run the pipeline (terminal mode)

```bash
python main.py
```

When the pipeline starts you will see a banner, then a line for every utterance:

```
============================================================
 Real-Time Speech-to-Speech Translation
 Target language : en
 ASR model       : base
 TTS backend     : gtts
 Noise suppression: on
 Press Ctrl-C to stop.
============================================================

[es → en]
  Heard   : Hola, ¿cómo estás?
  Translated: Hello, how are you?
  (1100 ms audio, 780 ms processing)

[es → en]
  Heard   : Hace buen tiempo hoy.
  Translated: The weather is nice today.
  (950 ms audio, 590 ms processing)
```

Press **Ctrl-C** to stop the pipeline gracefully.

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target fr` | `en` | Target language (BCP-47 code) |
| `--model small` | `base` | Whisper model size |
| `--device cuda` | `cpu` | Compute device |
| `--tts-backend pyttsx3` | `gtts` | TTS engine |
| `--no-noise-suppression` | off | Disable denoising |
| `--log-level DEBUG` | `INFO` | Logging verbosity |

**Examples:**

```bash
# Translate to French using a more accurate model
python main.py --target fr --model small

# Offline mode (no internet required for TTS or translation)
python main.py --tts-backend pyttsx3 --target de

# GPU-accelerated transcription
python main.py --device cuda --model medium

# Verbose debug output
python main.py --log-level DEBUG
```

### 6. Run the API server

```bash
python main.py --api --port 8000
```

You will see:

```
Starting API server on http://0.0.0.0:8000
  OpenAPI docs : http://0.0.0.0:8000/docs
  WebSocket    : ws://0.0.0.0:8000/ws/translate
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open these URLs in your browser:

| URL | What you'll find |
|-----|-----------------|
| http://localhost:8000/ | Minimal HTML live-translation page |
| http://localhost:8000/docs | Interactive Swagger / OpenAPI docs |
| http://localhost:8000/status | JSON pipeline status |
| `ws://localhost:8000/ws/translate` | WebSocket stream of translation events |

**Start the pipeline via the API and see live output:**

```bash
# Start the pipeline
curl -X POST http://localhost:8000/start

# Check status
curl http://localhost:8000/status

# Change the target language at runtime
curl -X POST http://localhost:8000/config/target \
     -H "Content-Type: application/json" \
     -d '{"language": "fr"}'

# Stop the pipeline
curl -X POST http://localhost:8000/stop
```

---

## Viewing Your Output

### Terminal output explained

Each translation event printed to the console has four parts:

```
[<source_lang> → <target_lang>]        ← detected and target language codes
  Heard   : <original transcription>   ← what Whisper heard you say
  Translated: <translated text>         ← MarianMT / Google / SeamlessM4T output
  (<N> ms audio, <M> ms processing)    ← playback length and total latency
```

Enable `--log-level DEBUG` to also see VAD events, noise-suppression statistics,
and model loading progress.

### Streamlit UI output explained

The web app shows **five live metrics** at the top of the page:

| Metric | Meaning |
|--------|---------|
| **Status** | 🟢 Running / 🔴 Stopped |
| **Target language** | The language currently being spoken aloud |
| **Segments translated** | Total utterances processed this session |
| **Uptime** | `MM:SS` elapsed since last **▶ Start** |
| **ASR model** | Whisper model size currently in use |

Below the metrics, every translated segment appears as a card with the original
text on the left and the translation on the right, plus timing information at
the bottom.  The feed is capped at the **50 most recent** cards and refreshes
automatically every second.

### WebSocket output (API mode)

Connect any WebSocket client (browser, `wscat`, Python `websockets`, etc.) to
`ws://localhost:8000/ws/translate` to receive a JSON event for every utterance:

```json
{
  "source_language": "en",
  "source_text": "Hello, how are you?",
  "target_language": "de",
  "translated_text": "Hallo, wie geht es dir?",
  "audio_duration_ms": 1200,
  "processing_time_ms": 840
}
```

Quick test with `wscat`:

```bash
npm install -g wscat
wscat -c ws://localhost:8000/ws/translate
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: sounddevice` | Missing audio library | `pip install sounddevice` |
| `OSError: PortAudio library not found` | PortAudio not installed | `sudo apt install portaudio19-dev` (Linux) or `brew install portaudio` (macOS) |
| No audio output / silent playback | `ffmpeg` not on PATH | Install ffmpeg: `sudo apt install ffmpeg` / `brew install ffmpeg` |
| `RuntimeError: CUDA out of memory` | GPU too small for model | Use a smaller model: `--model tiny` or `--device cpu` |
| Translation feed is empty | Pipeline not started | Click **▶ Start** in the UI or run `curl -X POST http://localhost:8000/start` |
| `uvicorn not installed` | Missing package | `pip install uvicorn[standard]` |
| Very slow first run | Models downloading | Wait for Whisper / MarianMT to download (one-time; cached afterwards) |
| `Pipeline initialization timed out` | Microphone not found | Check that your microphone is connected and not muted |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Minimal HTML UI |
| `GET` | `/status` | Pipeline status |
| `GET` | `/config` | Active configuration |
| `POST` | `/config/target` | Change target language |
| `POST` | `/start` | Start the pipeline |
| `POST` | `/stop` | Stop the pipeline |
| `WS` | `/ws/translate` | Stream translation events |

### WebSocket event format

```json
{
  "source_language": "en",
  "source_text": "Hello, how are you?",
  "target_language": "de",
  "translated_text": "Hallo, wie geht es dir?",
  "audio_duration_ms": 1200,
  "processing_time_ms": 840
}
```

### WebSocket command format

Send from client to change target language at runtime:

```json
{"action": "set_target", "language": "fr"}
```

---

## Configuration

All settings are dataclasses in `config.py`.  The easiest way to
override them is via environment variables or CLI flags.

| Setting | Env var | Default |
|---------|---------|---------|
| Google Translate API key | `GOOGLE_TRANSLATE_API_KEY` | — |
| ElevenLabs API key | `ELEVENLABS_API_KEY` | — |

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite is designed to work offline and without a microphone; heavy
model downloads (MarianMT, Faster-Whisper) are automatically skipped when
the relevant libraries are not installed.

---

## Supported Backends

### ASR
| Model | Speed | Quality |
|-------|-------|---------|
| `tiny` | Fastest | Low |
| `base` | Fast | Medium |
| `small` | Medium | Good |
| `medium` | Slow | Better |
| `large-v2` | Slowest | Best |

### Translation
| Backend | Requires | Quality |
|---------|----------|---------|
| `marian` | `transformers` | Good |
| `google` | API key + internet | Excellent |
| `seamless` | `transformers` ≥ 4.33 + GPU | Excellent |

### TTS
| Backend | Requires | Quality |
|---------|----------|---------|
| `gtts` | Internet | Natural |
| `pyttsx3` | OS engine | Robotic |
| `coqui` | `TTS` package | Neural |
| `elevenlabs` | API key + internet | Excellent |

---

## License

MIT