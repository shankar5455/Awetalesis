# Real-Time Speech-to-Speech Translation System (S2ST)

An end-to-end pipeline that captures live microphone input, transcribes it,
translates it into a target language, and plays back synthesised speech – all
in near real time.

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

A browser window will open automatically at http://localhost:8501.

**Using the UI:**

1. **Configure** – Use the sidebar to choose the target language, ASR model,
   TTS backend, and audio settings.
2. **Start** – Click **▶ Start** to begin listening.  Models load on first use.
3. **Speak** – Talk into your microphone.  Each utterance is transcribed,
   translated, and played back.  Results appear in the *Translation Feed*.
4. **Stop** – Click **⏹ Stop** to shut down the pipeline.

### 5. Run the pipeline (terminal mode)

```bash
python main.py
```

Speak into the microphone.  Each detected utterance is transcribed,
translated, and played back through the speakers.

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target fr` | `en` | Target language |
| `--model small` | `base` | Whisper model size |
| `--device cuda` | `cpu` | Compute device |
| `--tts-backend pyttsx3` | `gtts` | TTS engine |
| `--no-noise-suppression` | off | Disable denoising |
| `--log-level DEBUG` | `INFO` | Logging verbosity |

### 6. Run the API server

```bash
python main.py --api --port 8000
```

- OpenAPI docs: http://localhost:8000/docs
- Live translation feed (browser): http://localhost:8000/
- WebSocket: `ws://localhost:8000/ws/translate`

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