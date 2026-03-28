"""
processing/noise_suppression.py – Audio denoising before VAD / ASR.

The module tries the following backends in preference order and falls back
gracefully when a library is not installed:

1. ``noisereduce`` (spectral subtraction, pure-Python, always available as a
   dependency)
2. ``deepfilter`` (DeepFilterNet neural denoiser)
3. ``rnnoise`` (Xiph.org RNNoise – requires the ``rnnoise`` Python wrapper)
4. **Passthrough** – no-op; the raw audio is returned unchanged.
"""

from __future__ import annotations

import numpy as np

from config import NoiseSuppressionConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_import_noisereduce():
    try:
        import noisereduce as nr  # type: ignore
        return nr
    except ImportError:
        return None


def _try_import_deepfilter():
    try:
        from df import enhance, init_df  # type: ignore
        return enhance, init_df
    except ImportError:
        return None


def _try_import_rnnoise():
    try:
        import rnnoise  # type: ignore
        return rnnoise
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# NoiseSuppressionProcessor
# ---------------------------------------------------------------------------

class NoiseSuppressionProcessor:
    """
    Wraps a chosen denoising backend behind a single :meth:`process` method.

    Args:
        cfg: :class:`~config.NoiseSuppressionConfig` instance.
        sample_rate: Input audio sample rate (Hz).
    """

    def __init__(self, cfg: NoiseSuppressionConfig, sample_rate: int = 16_000) -> None:
        self._cfg = cfg
        self._sample_rate = sample_rate
        self._backend_name: str = "passthrough"
        self._nr_module = None
        self._df_model = None
        self._df_state = None

        if not cfg.enabled:
            logger.info("Noise suppression disabled.")
            return

        self._init_backend(cfg.backend)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_backend(self, requested: str) -> None:
        if requested == "noisereduce" or requested == "auto":
            nr = _try_import_noisereduce()
            if nr is not None:
                self._nr_module = nr
                self._backend_name = "noisereduce"
                logger.info("Noise suppression backend: noisereduce")
                return

        if requested == "deepfilter" or requested == "auto":
            result = _try_import_deepfilter()
            if result is not None:
                enhance, init_df = result
                try:
                    df_state, _ = init_df()
                    self._df_model = enhance
                    self._df_state = df_state
                    self._backend_name = "deepfilter"
                    logger.info("Noise suppression backend: DeepFilterNet")
                    return
                except Exception as exc:
                    logger.warning("DeepFilterNet init failed: %s", exc)

        if requested == "rnnoise" or requested == "auto":
            rnn = _try_import_rnnoise()
            if rnn is not None:
                self._backend_name = "rnnoise"
                self._rnnoise_module = rnn
                logger.info("Noise suppression backend: RNNoise")
                return

        logger.warning(
            "Requested noise suppression backend '%s' not available; "
            "falling back to passthrough.",
            requested,
        )
        self._backend_name = "passthrough"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise a segment of audio.

        Args:
            audio: 1-D float32 PCM array (values expected in –1 … +1).

        Returns:
            Denoised float32 1-D array of the same length.
        """
        if not self._cfg.enabled or self._backend_name == "passthrough":
            return audio

        try:
            return self._dispatch(audio)
        except Exception as exc:
            logger.warning("Noise suppression failed (%s); returning raw audio: %s", self._backend_name, exc)
            return audio

    def _dispatch(self, audio: np.ndarray) -> np.ndarray:
        if self._backend_name == "noisereduce":
            return self._run_noisereduce(audio)
        if self._backend_name == "deepfilter":
            return self._run_deepfilter(audio)
        if self._backend_name == "rnnoise":
            return self._run_rnnoise(audio)
        return audio

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _run_noisereduce(self, audio: np.ndarray) -> np.ndarray:
        """Spectral subtraction via the *noisereduce* library."""
        reduced = self._nr_module.reduce_noise(
            y=audio,
            sr=self._sample_rate,
            prop_decrease=self._cfg.prop_decrease,
            stationary=False,
        )
        return reduced.astype(np.float32)

    def _run_deepfilter(self, audio: np.ndarray) -> np.ndarray:
        """DeepFilterNet neural denoiser."""
        import torch  # type: ignore

        # DeepFilterNet expects float32 tensor of shape (1, samples)
        tensor = torch.from_numpy(audio).unsqueeze(0)
        enhanced = self._df_model(self._df_state, tensor)
        return enhanced.squeeze(0).numpy().astype(np.float32)

    def _run_rnnoise(self, audio: np.ndarray) -> np.ndarray:
        """RNNoise expects 16-bit PCM at 48 kHz; we resample if needed."""
        import io
        import wave

        # Normalise to int16
        pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        # Write to an in-memory WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_int16.tobytes())
        buf.seek(0)

        # Feed through rnnoise
        denoised_bytes = self._rnnoise_module.process_wav(buf.read())

        # Parse the output WAV
        with io.BytesIO(denoised_bytes) as out_buf:
            with wave.open(out_buf, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
        result = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return result

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend_name
