"""
pipeline/pipeline.py – End-to-end Speech-to-Speech Translation Pipeline.

This module wires together all processing stages into a continuous loop:

    MicrophoneStream
        → NoiseSuppressionProcessor
        → SileroVAD
        → WhisperASR
        → LanguageIdentifier
        → Translator
        → TTSEngine
        → Audio playback

Each stage runs synchronously within a single worker thread; the microphone
capture runs in a separate I/O thread provided by sounddevice, and results
are communicated to subscribers via a callback.

For higher throughput the translation and TTS stages can be moved to a thread
pool (see ``PipelineConfig.worker_threads``).

Usage::

    pipeline = S2STPipeline(config)
    pipeline.start()        # non-blocking; starts background threads
    ...
    pipeline.stop()         # graceful shutdown

Or as a context manager::

    with S2STPipeline(config) as p:
        time.sleep(30)      # listen for 30 seconds
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio.buffer import AudioBuffer
from audio.stream import MicrophoneStream
from config import Config
from processing.asr import ASRResult, WhisperASR
from processing.lid import LanguageIdentifier
from processing.noise_suppression import NoiseSuppressionProcessor
from processing.tts import TTSEngine
from processing.translation import Translator
from processing.vad import SileroVAD, SpeechSegment
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event / result dataclass shared with the API / UI layer
# ---------------------------------------------------------------------------

@dataclass
class TranslationEvent:
    """Represents one completed translation cycle."""

    source_language: str
    source_text: str
    target_language: str
    translated_text: str
    audio_duration_ms: float
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class S2STPipeline:
    """
    Full speech-to-speech translation pipeline.

    Args:
        cfg: :class:`~config.Config` root configuration object.
        on_result: Optional callback invoked with a
            :class:`TranslationEvent` after each successful cycle.
    """

    def __init__(
        self,
        cfg: Config,
        on_result: Optional[Callable[[TranslationEvent], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._on_result = on_result

        # ---- Instantiate processing components ----
        logger.info("Initialising pipeline components…")

        self._noise_suppressor = NoiseSuppressionProcessor(
            cfg.noise_suppression, sample_rate=cfg.audio.sample_rate
        )
        self._vad = SileroVAD(cfg.vad, sample_rate=cfg.audio.sample_rate)
        self._asr = WhisperASR(cfg.asr)
        # LID shares the Whisper model to avoid loading it twice
        self._lid = LanguageIdentifier(cfg.lid, whisper_model=self._asr.model)
        self._translator = Translator(cfg.translation)
        self._tts = TTSEngine(cfg.tts)

        # ---- Audio buffer & stream ----
        self._audio_buffer = AudioBuffer(
            max_seconds=cfg.pipeline.buffer_max_seconds,
            sample_rate=cfg.audio.sample_rate,
        )
        self._mic_stream = MicrophoneStream(cfg.audio)

        # ---- Control ----
        self._running = threading.Event()
        self._segment_queue: queue.Queue[SpeechSegment] = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        logger.info("Pipeline initialised.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start microphone capture and processing worker threads."""
        if self._running.is_set():
            logger.warning("Pipeline already running.")
            return

        self._running.set()
        self._vad.reset()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="s2st-capture",
            daemon=True,
        )
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            name="s2st-worker",
            daemon=True,
        )

        self._capture_thread.start()
        self._worker_thread.start()
        logger.info("Pipeline started.")

    def stop(self) -> None:
        """Gracefully stop all threads."""
        logger.info("Pipeline stopping…")
        self._running.clear()
        self._mic_stream.stop()
        # Unblock the worker queue
        self._segment_queue.put(None)  # type: ignore[arg-type]
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        logger.info("Pipeline stopped.")

    def __enter__(self) -> "S2STPipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """I/O thread: reads mic chunks, denoises, and runs VAD."""
        self._mic_stream.start()
        try:
            for chunk in self._mic_stream:
                if not self._running.is_set():
                    break

                # 1. Noise suppression
                clean = self._noise_suppressor.process(chunk)

                # 2. VAD – may return zero or more complete speech segments
                segments = self._vad.process_chunk(clean)
                for seg in segments:
                    self._segment_queue.put(seg)

        except Exception as exc:
            logger.error("Capture loop error: %s", exc, exc_info=True)
        finally:
            # Flush any remaining speech
            for seg in self._vad.flush():
                self._segment_queue.put(seg)
            self._segment_queue.put(None)  # type: ignore[arg-type]

    def _process_loop(self) -> None:
        """Worker thread: ASR → LID → Translation → TTS."""
        while self._running.is_set():
            try:
                item = self._segment_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break  # shutdown signal

            self._handle_segment(item)

    def _handle_segment(self, segment: SpeechSegment) -> None:
        """Process one VAD-detected speech segment end-to-end."""
        t0 = time.perf_counter()

        # 3. ASR
        asr_result: ASRResult = self._asr.transcribe(segment.audio)
        if not asr_result.text.strip():
            logger.debug("ASR returned empty text; skipping.")
            return

        # 4. LID (use ASR's detected language + refine with text if needed)
        src_lang = asr_result.language
        if src_lang in ("unknown", None) or asr_result.language_probability < self._cfg.lid.confidence_threshold:
            src_lang, _ = self._lid.detect(audio=segment.audio)
        if src_lang in ("unknown", None):
            src_lang, _ = self._lid.detect(text=asr_result.text)

        # 5. Translation
        tgt_lang = self._cfg.translation.target_language
        translated = self._translator.translate(
            asr_result.text, source_lang=src_lang, target_lang=tgt_lang
        )

        # 6. TTS + playback
        self._tts.synthesize(translated, language=tgt_lang, play=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        event = TranslationEvent(
            source_language=src_lang,
            source_text=asr_result.text,
            target_language=tgt_lang,
            translated_text=translated,
            audio_duration_ms=segment.duration_ms,
            processing_time_ms=elapsed_ms,
        )
        logger.info(
            "[%s → %s] %r → %r  (%.0f ms audio, %.0f ms processing)",
            src_lang,
            tgt_lang,
            asr_result.text[:60],
            translated[:60],
            segment.duration_ms,
            elapsed_ms,
        )

        if self._on_result:
            try:
                self._on_result(event)
            except Exception as exc:
                logger.warning("on_result callback raised: %s", exc)
