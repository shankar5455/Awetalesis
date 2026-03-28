"""
processing/asr.py – Automatic Speech Recognition using Faster-Whisper.

Faster-Whisper is a reimplementation of OpenAI Whisper using CTranslate2 for
faster CPU/GPU inference and lower memory usage.

Reference:
    https://github.com/SYSTRAN/faster-whisper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import ASRConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ASRResult:
    """Structured result from one ASR inference pass."""

    text: str
    language: str = "unknown"
    language_probability: float = 0.0
    segments: List[dict] = field(default_factory=list)
    duration_seconds: float = 0.0


class WhisperASR:
    """
    Wraps Faster-Whisper ``WhisperModel`` for synchronous transcription.

    The model is loaded lazily on the first call to :meth:`transcribe`.

    Args:
        cfg: :class:`~config.ASRConfig` instance.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Explicitly load (or reload) the Whisper model."""
        try:
            from faster_whisper import WhisperModel  # type: ignore

            logger.info(
                "Loading Faster-Whisper model  size=%s  device=%s  compute=%s",
                self._cfg.model_size,
                self._cfg.device,
                self._cfg.compute_type,
            )
            self._model = WhisperModel(
                self._cfg.model_size,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )
            self._loaded = True
            logger.info("Faster-Whisper model loaded.")
        except ImportError:
            logger.error(
                "faster-whisper is not installed.  "
                "Install with: pip install faster-whisper"
            )
            self._loaded = False
        except Exception as exc:
            logger.error("Failed to load Faster-Whisper model: %s", exc)
            self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> ASRResult:
        """
        Transcribe a speech segment.

        Args:
            audio: 1-D float32 PCM array at 16 kHz.
            language: BCP-47 language hint (e.g. ``"en"``).  Pass *None* to
                      use auto-detection.

        Returns:
            :class:`ASRResult` with the transcription and metadata.
        """
        self._ensure_loaded()

        if not self._loaded or self._model is None:
            logger.warning("ASR model not available; returning empty result.")
            return ASRResult(text="", language="unknown")

        # Whisper expects float32 audio
        audio = audio.astype(np.float32)

        # Clamp audio to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        lang = language if language and language != "unknown" else (
            self._cfg.language
        )

        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language=lang,
                task=self._cfg.task,
                beam_size=self._cfg.beam_size,
                vad_filter=self._cfg.vad_filter,
                word_timestamps=self._cfg.word_timestamps,
            )

            # Materialise the generator (CTranslate2 is lazy)
            segments_list = []
            full_text_parts = []
            for seg in segments_iter:
                segments_list.append(
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "words": [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in (seg.words or [])
                        ],
                    }
                )
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)
            detected_lang = info.language or "unknown"
            lang_prob = float(info.language_probability or 0.0)
            duration = float(info.duration) if hasattr(info, "duration") else 0.0

            result = ASRResult(
                text=full_text,
                language=detected_lang,
                language_probability=lang_prob,
                segments=segments_list,
                duration_seconds=duration,
            )
            logger.debug(
                "ASR: lang=%s prob=%.2f text=%r",
                detected_lang,
                lang_prob,
                full_text[:80],
            )
            return result

        except Exception as exc:
            logger.error("ASR transcription error: %s", exc, exc_info=True)
            return ASRResult(text="", language="unknown")

    # ------------------------------------------------------------------
    # Model access (shared with LID)
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The raw ``WhisperModel`` instance (loads on first access)."""
        self._ensure_loaded()
        return self._model
