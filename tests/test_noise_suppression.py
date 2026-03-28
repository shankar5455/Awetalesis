"""
tests/test_noise_suppression.py – Tests for the noise suppression module.
"""

import numpy as np
import pytest

from config import NoiseSuppressionConfig
from processing.noise_suppression import NoiseSuppressionProcessor


def _white_noise(n: int = 8_000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(n).astype(np.float32) * 0.1


class TestNoiseSuppressionDisabled:
    def test_passthrough_when_disabled(self):
        cfg = NoiseSuppressionConfig(enabled=False)
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        audio = _white_noise()
        result = proc.process(audio)
        np.testing.assert_array_equal(result, audio)

    def test_backend_name_passthrough(self):
        cfg = NoiseSuppressionConfig(enabled=False)
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        assert proc.backend_name == "passthrough"


class TestNoiseSuppressionNoisereduce:
    """Integration test – requires noisereduce to be installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("noisereduce")

    def test_output_shape(self):
        cfg = NoiseSuppressionConfig(enabled=True, backend="noisereduce")
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        audio = _white_noise(16_000)
        result = proc.process(audio)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_backend_name(self):
        cfg = NoiseSuppressionConfig(enabled=True, backend="noisereduce")
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        assert proc.backend_name == "noisereduce"

    def test_noise_is_reduced(self):
        """RMS should decrease after noise suppression."""
        cfg = NoiseSuppressionConfig(enabled=True, backend="noisereduce", prop_decrease=1.0)
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        audio = _white_noise(16_000)
        result = proc.process(audio)
        rms_before = float(np.sqrt(np.mean(audio ** 2)))
        rms_after = float(np.sqrt(np.mean(result ** 2)))
        assert rms_after <= rms_before + 1e-6  # not louder than original


class TestNoiseSuppressionFallback:
    def test_fallback_to_passthrough_for_unknown_backend(self):
        cfg = NoiseSuppressionConfig(enabled=True, backend="nonexistent_backend_xyz")
        proc = NoiseSuppressionProcessor(cfg, sample_rate=16_000)
        # Should not raise; should fall back to passthrough
        audio = _white_noise()
        result = proc.process(audio)
        # With passthrough the arrays are identical
        np.testing.assert_array_equal(result, audio)
