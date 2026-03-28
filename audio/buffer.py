"""
audio/buffer.py – Thread-safe circular audio buffer.

Producers (the microphone stream) write raw PCM frames; consumers (the VAD /
pipeline) read contiguous segments.  The buffer is implemented on top of
``collections.deque`` so it never blocks the capture thread.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class AudioBuffer:
    """
    Thread-safe, capacity-limited ring buffer for raw PCM audio.

    Args:
        max_seconds: Maximum audio duration stored at any time (seconds).
        sample_rate: Expected sample rate of incoming audio (Hz).
    """

    def __init__(self, max_seconds: float = 5.0, sample_rate: int = 16_000) -> None:
        self._max_samples: int = int(max_seconds * sample_rate)
        self._sample_rate: int = sample_rate
        self._buf: Deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """
        Append a PCM chunk to the buffer.

        Old samples are silently discarded when the buffer is full.

        Args:
            chunk: 1-D float32 numpy array of PCM samples.
        """
        with self._lock:
            self._buf.append(chunk.copy())
            self._total_samples += len(chunk)

            # Trim oldest chunks if capacity exceeded
            while self._total_samples > self._max_samples and self._buf:
                oldest = self._buf.popleft()
                self._total_samples -= len(oldest)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_all(self) -> np.ndarray:
        """
        Return all buffered audio as a single contiguous array **without**
        removing it from the buffer.

        Returns:
            float32 numpy array, possibly empty.
        """
        with self._lock:
            if not self._buf:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buf)).astype(np.float32)

    def drain(self) -> np.ndarray:
        """
        Return **and remove** all buffered audio.

        Returns:
            float32 numpy array, possibly empty.
        """
        with self._lock:
            if not self._buf:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(list(self._buf)).astype(np.float32)
            self._buf.clear()
            self._total_samples = 0
            return audio

    def drain_seconds(self, seconds: float) -> Optional[np.ndarray]:
        """
        Drain exactly *seconds* worth of audio from the front of the buffer.

        Returns *None* if the buffer does not hold enough audio yet.

        Args:
            seconds: Desired duration.

        Returns:
            float32 numpy array or *None*.
        """
        n = int(seconds * self._sample_rate)
        with self._lock:
            if self._total_samples < n:
                return None

            collected: list[np.ndarray] = []
            remaining = n
            while remaining > 0 and self._buf:
                chunk = self._buf.popleft()
                self._total_samples -= len(chunk)
                if len(chunk) <= remaining:
                    collected.append(chunk)
                    remaining -= len(chunk)
                else:
                    # Split the chunk; put the tail back
                    collected.append(chunk[:remaining])
                    tail = chunk[remaining:]
                    self._buf.appendleft(tail)
                    self._total_samples += len(tail)
                    remaining = 0

            return np.concatenate(collected).astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        """Current buffered duration in seconds."""
        with self._lock:
            return self._total_samples / self._sample_rate

    @property
    def num_samples(self) -> int:
        """Number of samples currently buffered."""
        with self._lock:
            return self._total_samples

    def clear(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._buf.clear()
            self._total_samples = 0

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        return (
            f"AudioBuffer(duration={self.duration_seconds:.3f}s, "
            f"samples={self.num_samples}, "
            f"max_samples={self._max_samples})"
        )
