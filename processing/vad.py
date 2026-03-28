"""
processing/vad.py – Voice Activity Detection using Silero VAD.

Silero VAD is a lightweight LSTM-based model (~1 MB) that runs efficiently on
CPU and provides per-chunk speech probability.  This wrapper buffers incoming
audio frames, runs the Silero model on each 512-sample window, and emits
complete speech *segments* when a silence boundary is detected.

Silero VAD model download happens automatically via ``torch.hub``.

Reference:
    https://github.com/snakers4/silero-vad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import VADConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment."""

    audio: np.ndarray          # float32 PCM, 16 kHz
    start_sample: int          # offset from the beginning of the current session
    end_sample: int
    duration_ms: float

    @classmethod
    def from_samples(
        cls, audio: np.ndarray, start: int, end: int, sample_rate: int
    ) -> "SpeechSegment":
        duration_ms = (end - start) / sample_rate * 1000
        return cls(
            audio=audio.astype(np.float32),
            start_sample=start,
            end_sample=end,
            duration_ms=duration_ms,
        )


class SileroVAD:
    """
    Silero-VAD wrapper that accumulates audio frames and yields speech segments.

    Args:
        cfg: :class:`~config.VADConfig` instance.
        sample_rate: Input audio sample rate (must be 8 000 or 16 000 Hz).
    """

    _MODEL_REPO = "snakers4/silero-vad"
    _MODEL_NAME = "silero_vad"

    def __init__(self, cfg: VADConfig, sample_rate: int = 16_000) -> None:
        self._cfg = cfg
        self._sr = sample_rate
        self._model = None
        self._utils = None
        self._loaded = False

        # State machine for segment detection
        self._speech_buffer: List[np.ndarray] = []
        self._is_speaking: bool = False
        self._silence_samples: int = 0
        self._total_samples: int = 0

        # Partial samples that don't fill a full window yet
        self._pending: np.ndarray = np.array([], dtype=np.float32)

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download (or load from cache) the Silero VAD model."""
        try:
            import torch  # type: ignore

            model, utils = torch.hub.load(
                repo_or_dir=self._MODEL_REPO,
                model=self._MODEL_NAME,
                force_reload=False,
                onnx=False,
                verbose=False,
                trust_repo=True,
            )
            model.eval()
            self._model = model
            self._utils = utils
            self._loaded = True
            logger.info("Silero VAD model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Could not load Silero VAD model (%s).  "
                "Falling back to energy-based VAD.",
                exc,
            )
            self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> List[SpeechSegment]:
        """
        Feed a raw PCM chunk and return any completed speech segments.

        Args:
            chunk: 1-D float32 array at ``self._sr`` Hz.

        Returns:
            A (possibly empty) list of :class:`SpeechSegment` objects.
        """
        # Accumulate pending samples + new chunk
        combined = np.concatenate([self._pending, chunk])
        window = self._cfg.window_size_samples
        n_complete = len(combined) // window
        self._pending = combined[n_complete * window:]

        segments: List[SpeechSegment] = []

        for i in range(n_complete):
            frame = combined[i * window: (i + 1) * window]
            prob = self._get_speech_probability(frame)
            segments.extend(self._update_state(frame, prob))
            self._total_samples += window

        return segments

    def flush(self) -> List[SpeechSegment]:
        """
        Force-emit any buffered speech (call at end of session).

        Returns:
            Remaining speech segments.
        """
        if not self._speech_buffer:
            return []

        audio = np.concatenate(self._speech_buffer)
        start = self._total_samples - len(audio)
        seg = SpeechSegment.from_samples(audio, start, self._total_samples, self._sr)
        self._reset_speech_state()

        if seg.duration_ms >= self._cfg.min_speech_ms:
            logger.debug("VAD flush: segment %.0f ms", seg.duration_ms)
            return [seg]
        return []

    def reset(self) -> None:
        """Reset all internal state (start a new session)."""
        self._speech_buffer.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._total_samples = 0
        self._pending = np.array([], dtype=np.float32)
        if self._loaded and self._model is not None:
            self._model.reset_states()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_speech_probability(self, frame: np.ndarray) -> float:
        """Return speech probability (0–1) for a single window."""
        if self._loaded and self._model is not None:
            try:
                import torch  # type: ignore

                tensor = torch.from_numpy(frame).unsqueeze(0)  # (1, T)
                with torch.no_grad():
                    prob = self._model(tensor, self._sr).item()
                return float(prob)
            except Exception as exc:
                logger.debug("Silero inference error: %s", exc)

        # Fallback: simple energy-based detection
        return self._energy_vad(frame)

    @staticmethod
    def _energy_vad(frame: np.ndarray, threshold: float = 0.01) -> float:
        """Naïve RMS energy gate as a fallback VAD."""
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return 1.0 if rms > threshold else 0.0

    def _update_state(
        self, frame: np.ndarray, prob: float
    ) -> List[SpeechSegment]:
        """Apply the state machine and return finished segments."""
        cfg = self._cfg
        segments: List[SpeechSegment] = []
        is_speech = prob >= cfg.threshold

        if is_speech:
            self._speech_buffer.append(frame)
            self._is_speaking = True
            self._silence_samples = 0

            # Hard cap: flush if the segment is already very long
            buffered_samples = sum(len(f) for f in self._speech_buffer)
            if buffered_samples >= int(cfg.max_segment_ms / 1000 * self._sr):
                seg = self._emit_segment()
                if seg is not None:
                    segments.append(seg)
        else:
            if self._is_speaking:
                self._speech_buffer.append(frame)  # include brief silences
                self._silence_samples += len(frame)
                silence_ms = self._silence_samples / self._sr * 1000

                if silence_ms >= cfg.min_silence_ms:
                    seg = self._emit_segment()
                    if seg is not None:
                        segments.append(seg)

        return segments

    def _emit_segment(self) -> Optional[SpeechSegment]:
        """Concatenate the speech buffer into a segment."""
        if not self._speech_buffer:
            return None

        audio = np.concatenate(self._speech_buffer)
        end = self._total_samples
        start = end - len(audio)
        seg = SpeechSegment.from_samples(audio, start, end, self._sr)
        self._reset_speech_state()

        if seg.duration_ms < self._cfg.min_speech_ms:
            logger.debug("VAD: segment too short (%.0f ms), discarding.", seg.duration_ms)
            return None

        logger.debug("VAD segment: %.0f ms", seg.duration_ms)
        return seg

    def _reset_speech_state(self) -> None:
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_samples = 0
