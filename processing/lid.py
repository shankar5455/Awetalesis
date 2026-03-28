"""
processing/lid.py – Language Identification (LID).

Three backends are supported:

1. **whisper** – Uses the built-in language-detection pass of Faster-Whisper.
   No extra model is required; the same Whisper model that runs ASR can detect
   the language from the first ~30 s of audio.

2. **langdetect** – ``langdetect`` library (pure-Python, text-based). Falls
   back to this when ASR has already produced a transcript.

3. **fasttext** – Facebook's fastText ``lid.176.bin`` compact model.  Fast and
   accurate, but requires downloading the model file separately.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from config import LIDConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class LanguageIdentifier:
    """
    Detect the language of a spoken utterance.

    Args:
        cfg: :class:`~config.LIDConfig` instance.
        whisper_model: A pre-loaded Faster-Whisper
            ``WhisperModel`` instance (optional).  Shared with ASR to avoid
            loading the model twice.
    """

    def __init__(
        self,
        cfg: LIDConfig,
        whisper_model=None,
    ) -> None:
        self._cfg = cfg
        self._whisper_model = whisper_model
        self._fasttext_model = None
        self._backend: str = "passthrough"

        self._init_backend(cfg.backend)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backend(self, requested: str) -> None:
        if requested == "whisper":
            if self._whisper_model is not None:
                self._backend = "whisper"
                logger.info("LID backend: whisper (shared model)")
                return
            logger.warning(
                "LID backend 'whisper' requested but no WhisperModel supplied. "
                "Falling back to 'langdetect'."
            )
            requested = "langdetect"

        if requested == "langdetect":
            try:
                import langdetect  # type: ignore  # noqa: F401

                self._backend = "langdetect"
                logger.info("LID backend: langdetect")
                return
            except ImportError:
                logger.warning("langdetect not installed; trying fasttext.")

        if requested == "fasttext":
            path = self._cfg.fasttext_model_path
            if path:
                try:
                    import fasttext  # type: ignore

                    self._fasttext_model = fasttext.load_model(path)
                    self._backend = "fasttext"
                    logger.info("LID backend: fasttext (model=%s)", path)
                    return
                except Exception as exc:
                    logger.warning("fasttext init failed: %s", exc)

        logger.warning("No LID backend available; language will be 'unknown'.")
        self._backend = "passthrough"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        audio: Optional[np.ndarray] = None,
        text: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Identify the language of the provided audio or text.

        Exactly one of *audio* or *text* should be provided.  When *audio*
        is given, the Whisper / fastText audio backend is preferred; when only
        *text* is available, langdetect is used.

        Args:
            audio: 1-D float32 PCM array (16 kHz).
            text: Already-transcribed text (used with langdetect).

        Returns:
            ``(language_code, confidence)`` where *language_code* is a BCP-47
            string (e.g. ``"en"``) and *confidence* is in 0 … 1.
        """
        try:
            return self._dispatch(audio, text)
        except Exception as exc:
            logger.warning("LID failed: %s", exc)
            return "unknown", 0.0

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        audio: Optional[np.ndarray],
        text: Optional[str],
    ) -> Tuple[str, float]:
        if self._backend == "whisper" and audio is not None:
            return self._detect_whisper(audio)
        if self._backend == "langdetect" and text is not None:
            return self._detect_langdetect(text)
        if self._backend == "langdetect" and audio is not None:
            # We have audio but only langdetect; return unknown until text is ready
            return "unknown", 0.0
        if self._backend == "fasttext" and audio is not None:
            # fastText is text-based; we cannot detect from raw audio here
            return "unknown", 0.0
        if self._backend == "fasttext" and text is not None:
            return self._detect_fasttext(text)
        return "unknown", 0.0

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_whisper(self, audio: np.ndarray) -> Tuple[str, float]:
        """Use Faster-Whisper's built-in language detection pass."""
        # Whisper processes at most 30 s; take the first 30 s
        max_samples = 30 * 16_000
        segment = audio[:max_samples]

        _, info = self._whisper_model.transcribe(
            segment,
            task="transcribe",
            beam_size=1,
            language=None,          # force auto-detect
            vad_filter=False,
        )
        lang = info.language or "unknown"
        prob = float(info.language_probability or 0.0)
        logger.debug("LID (whisper): lang=%s prob=%.2f", lang, prob)
        return lang, prob

    def _detect_langdetect(self, text: str) -> Tuple[str, float]:
        """langdetect text-based detection."""
        from langdetect import detect_langs  # type: ignore

        results = detect_langs(text)
        if not results:
            return "unknown", 0.0
        top = results[0]
        lang = top.lang
        prob = float(top.prob)
        logger.debug("LID (langdetect): lang=%s prob=%.2f", lang, prob)
        return lang, prob

    def _detect_fasttext(self, text: str) -> Tuple[str, float]:
        """fastText LID model."""
        clean = text.replace("\n", " ")
        labels, probs = self._fasttext_model.predict(clean, k=1)
        lang = labels[0].replace("__label__", "")
        prob = float(probs[0])
        logger.debug("LID (fasttext): lang=%s prob=%.2f", lang, prob)
        return lang, prob
