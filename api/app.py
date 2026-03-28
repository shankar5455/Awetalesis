"""
api/app.py – FastAPI backend for the S2ST system.

Exposes:
  GET  /                 → health check
  GET  /config           → current configuration (JSON)
  POST /config/target    → update target language
  GET  /status           → pipeline running state
  POST /start            → start the pipeline
  POST /stop             → stop the pipeline
  WS   /ws/translate     → WebSocket; streams TranslationEvent JSON objects

Run locally::

    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncGenerator, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import config as _cfg
from pipeline.pipeline import S2STPipeline, TranslationEvent
from utils.logger import get_logger, setup_logging

# Initialise logging as early as possible
setup_logging(_cfg.pipeline.log_level)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application state (shared between REST and WebSocket endpoints)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.pipeline: S2STPipeline | None = None
        self.is_running: bool = False
        self.ws_clients: Set[WebSocket] = set()
        self._lock = threading.Lock()

    def broadcast_event(self, event: TranslationEvent) -> None:
        """Called from the pipeline worker thread; schedules a WS broadcast."""
        payload = json.dumps(asdict(event))
        # We push into the event loop from a non-async context
        for ws in list(self.ws_clients):
            asyncio.run_coroutine_threadsafe(
                ws.send_text(payload),
                loop=_event_loop,
            )


_state = AppState()
_event_loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    logger.info("S2ST API starting up.")
    yield
    if _state.pipeline and _state.is_running:
        _state.pipeline.stop()
    logger.info("S2ST API shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real-Time Speech-to-Speech Translation API",
    description=(
        "End-to-end S2ST pipeline: mic → noise suppression → VAD → "
        "ASR (Faster-Whisper) → LID → translation → TTS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TargetLanguageRequest(BaseModel):
    target_language: str  # BCP-47 code, e.g. "en", "de", "fr"


class StatusResponse(BaseModel):
    is_running: bool
    target_language: str
    asr_model: str
    tts_backend: str
    noise_suppression: bool
    vad_threshold: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Serve a minimal status page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>S2ST API</title></head>
    <body>
      <h1>Real-Time Speech-to-Speech Translation</h1>
      <p>API is running. See <a href="/docs">/docs</a> for the OpenAPI spec.</p>
      <h2>Live Translation Feed</h2>
      <div id="feed" style="font-family:monospace;white-space:pre-wrap;background:#111;color:#0f0;padding:1em;height:300px;overflow-y:auto"></div>
      <script>
        const feed = document.getElementById('feed');
        const ws = new WebSocket(`ws://${location.host}/ws/translate`);
        ws.onmessage = (evt) => {
          const d = JSON.parse(evt.data);
          feed.textContent += `[${d.source_language}→${d.target_language}] ${d.source_text} ⟶ ${d.translated_text}\\n`;
          feed.scrollTop = feed.scrollHeight;
        };
        ws.onerror = () => feed.textContent += '\\n[WebSocket error]\\n';
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return the current pipeline status."""
    return StatusResponse(
        is_running=_state.is_running,
        target_language=_cfg.translation.target_language,
        asr_model=_cfg.asr.model_size,
        tts_backend=_cfg.tts.backend,
        noise_suppression=_cfg.noise_suppression.enabled,
        vad_threshold=_cfg.vad.threshold,
    )


@app.get("/config")
async def get_config() -> dict:
    """Return the active configuration as JSON (sensitive fields redacted)."""
    data = {
        "audio": {"sample_rate": _cfg.audio.sample_rate, "chunk_ms": _cfg.audio.chunk_ms},
        "vad": {"threshold": _cfg.vad.threshold, "min_speech_ms": _cfg.vad.min_speech_ms},
        "asr": {"model_size": _cfg.asr.model_size, "device": _cfg.asr.device},
        "lid": {"backend": _cfg.lid.backend},
        "translation": {
            "backend": _cfg.translation.backend,
            "target_language": _cfg.translation.target_language,
        },
        "tts": {"backend": _cfg.tts.backend, "language": _cfg.tts.language},
        "noise_suppression": {"enabled": _cfg.noise_suppression.enabled},
    }
    return data


@app.post("/config/target")
async def set_target_language(body: TargetLanguageRequest) -> dict:
    """Update the target language for translation and TTS output."""
    _cfg.translation.target_language = body.target_language
    _cfg.tts.language = body.target_language
    logger.info("Target language updated to: %s", body.target_language)
    return {"message": f"Target language set to '{body.target_language}'."}


@app.post("/start")
async def start_pipeline() -> dict:
    """Start the speech-to-speech translation pipeline."""
    if _state.is_running:
        raise HTTPException(status_code=409, detail="Pipeline is already running.")

    _state.pipeline = S2STPipeline(_cfg, on_result=_state.broadcast_event)
    _state.pipeline.start()
    _state.is_running = True
    return {"message": "Pipeline started."}


@app.post("/stop")
async def stop_pipeline() -> dict:
    """Stop the running pipeline."""
    if not _state.is_running or _state.pipeline is None:
        raise HTTPException(status_code=409, detail="Pipeline is not running.")

    _state.pipeline.stop()
    _state.is_running = False
    return {"message": "Pipeline stopped."}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket) -> None:
    """
    WebSocket endpoint.

    Clients connect here and receive ``TranslationEvent`` objects as JSON
    whenever the pipeline processes a speech segment.

    The client may also send a JSON message to update the target language::

        {"action": "set_target", "language": "de"}
    """
    await websocket.accept()
    _state.ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total).", len(_state.ws_clients))

    try:
        while True:
            # Receive commands from the client (non-blocking poll)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
                if msg.get("action") == "set_target":
                    lang = msg.get("language", "en")
                    _cfg.translation.target_language = lang
                    _cfg.tts.language = lang
                    await websocket.send_text(
                        json.dumps({"message": f"Target language set to '{lang}'"})
                    )
            except asyncio.TimeoutError:
                pass  # no message from client – normal
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        _state.ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d total).", len(_state.ws_clients))
