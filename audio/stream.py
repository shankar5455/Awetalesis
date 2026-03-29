"""
audio/stream.py – Real-time microphone capture with sounddevice.

The :class:`MicrophoneStream` starts a background thread that continuously
reads raw PCM frames from the default input device and places them on a
``queue.Queue``.  Consumers dequeue frames at their own pace.

Usage::

    stream = MicrophoneStream(config.audio)
    with stream:
        for chunk in stream:          # blocks until a frame is available
            process(chunk)
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np

from config import AudioConfig
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except (ImportError, OSError):
    _SD_AVAILABLE = False
    logger.warning(
        "sounddevice is not available (library not installed or PortAudio not found).  "
        "MicrophoneStream will operate in SILENT / TEST mode only."
    )


class MicrophoneStream:
    """
    Continuously capture microphone audio and expose it as an iterator.

    Args:
        cfg: :class:`~config.AudioConfig` instance.
        maxsize: Maximum number of chunks that may be queued before the
                 oldest are dropped (prevents unbounded memory growth).
    """

    def __init__(self, cfg: AudioConfig, maxsize: int = 100) -> None:
        self._cfg = cfg
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=maxsize)
        self._stream: Optional[object] = None  # sounddevice.InputStream
        self._running = threading.Event()

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin capturing."""
        if not _SD_AVAILABLE:
            logger.warning("sounddevice unavailable – stream will produce silence.")
            self._running.set()
            return

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("sounddevice status: %s", status)
            chunk = indata[:, 0].copy()  # take first channel → 1-D
            try:
                self._queue.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                # Drop the oldest frame to keep latency low
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(chunk.astype(np.float32))

        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            blocksize=self._cfg.chunk_samples,
            device=self._cfg.device_index,
            channels=self._cfg.channels,
            dtype=self._cfg.dtype,
            callback=_callback,
        )
        self._stream.start()  # type: ignore[union-attr]
        self._running.set()
        logger.info(
            "MicrophoneStream started  sample_rate=%d  chunk_ms=%d  device=%s",
            self._cfg.sample_rate,
            self._cfg.chunk_ms,
            self._cfg.device_index,
        )

    def stop(self) -> None:
        """Stop the input stream and signal the iterator to exit."""
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()   # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
        # Poison-pill so that __iter__ exits cleanly
        self._queue.put(None)
        logger.info("MicrophoneStream stopped.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yield PCM chunks as float32 numpy arrays until the stream stops."""
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            yield chunk

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Block until a chunk is available and return it (or *None* on timeout).

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            float32 1-D numpy array or *None*.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """*True* while the stream is active."""
        return self._running.is_set()


@contextmanager
def open_microphone(cfg: AudioConfig) -> Generator[MicrophoneStream, None, None]:
    """
    Convenience context manager::

        with open_microphone(config.audio) as mic:
            for chunk in mic:
                ...
    """
    stream = MicrophoneStream(cfg)
    try:
        stream.start()
        yield stream
    finally:
        stream.stop()
