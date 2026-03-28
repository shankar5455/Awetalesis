"""
tests/test_audio_buffer.py – Unit tests for audio/buffer.py.
"""

import numpy as np
import pytest

from audio.buffer import AudioBuffer


def _ramp(n: int, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    """Helper: linearly spaced float32 array of length n."""
    return np.linspace(start, end, n, dtype=np.float32)


class TestAudioBufferPush:
    def test_push_increases_num_samples(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        chunk = _ramp(512)
        buf.push(chunk)
        assert buf.num_samples == 512

    def test_push_multiple_chunks(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        buf.push(_ramp(256))
        buf.push(_ramp(512))
        assert buf.num_samples == 768

    def test_push_does_not_modify_input(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        original = _ramp(128)
        copy = original.copy()
        buf.push(original)
        original[0] = 999.0  # mutate after push
        # Buffer should hold the pre-mutation copy
        audio = buf.read_all()
        assert audio[0] == pytest.approx(copy[0])

    def test_overflow_trims_oldest(self):
        sr = 16_000
        buf = AudioBuffer(max_seconds=1.0, sample_rate=sr)  # cap at 16000 samples
        # Push 2 seconds worth
        buf.push(np.zeros(sr, dtype=np.float32))
        buf.push(np.ones(sr, dtype=np.float32))
        # Should not exceed 16000 samples
        assert buf.num_samples <= sr


class TestAudioBufferRead:
    def test_read_all_does_not_drain(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        buf.push(_ramp(100))
        _ = buf.read_all()
        assert buf.num_samples == 100  # still there

    def test_read_all_empty(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        result = buf.read_all()
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_drain_returns_and_removes(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        chunk = _ramp(256)
        buf.push(chunk)
        result = buf.drain()
        assert buf.num_samples == 0
        assert len(result) == 256

    def test_drain_empty(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        result = buf.drain()
        assert len(result) == 0

    def test_drain_seconds_returns_none_when_insufficient(self):
        sr = 16_000
        buf = AudioBuffer(max_seconds=5.0, sample_rate=sr)
        buf.push(np.zeros(sr // 2, dtype=np.float32))  # 0.5 s
        result = buf.drain_seconds(1.0)  # request 1 s
        assert result is None

    def test_drain_seconds_returns_exactly_n_samples(self):
        sr = 16_000
        buf = AudioBuffer(max_seconds=5.0, sample_rate=sr)
        buf.push(np.ones(sr * 2, dtype=np.float32))  # 2 s
        result = buf.drain_seconds(1.0)  # request 1 s
        assert result is not None
        assert len(result) == sr
        assert buf.num_samples == sr  # remaining: 1 s


class TestAudioBufferProperties:
    def test_duration_seconds(self):
        sr = 16_000
        buf = AudioBuffer(max_seconds=5.0, sample_rate=sr)
        buf.push(np.zeros(sr, dtype=np.float32))
        assert buf.duration_seconds == pytest.approx(1.0)

    def test_len(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        buf.push(np.zeros(100, dtype=np.float32))
        assert len(buf) == 100

    def test_clear(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        buf.push(np.zeros(256, dtype=np.float32))
        buf.clear()
        assert buf.num_samples == 0
        assert len(buf.read_all()) == 0

    def test_repr(self):
        buf = AudioBuffer(max_seconds=5.0, sample_rate=16_000)
        r = repr(buf)
        assert "AudioBuffer" in r


class TestAudioBufferThreadSafety:
    """Smoke-test concurrent push/drain from multiple threads."""

    def test_concurrent_push_drain(self):
        import threading

        sr = 16_000
        buf = AudioBuffer(max_seconds=2.0, sample_rate=sr)
        errors = []

        def producer():
            try:
                for _ in range(50):
                    buf.push(np.random.rand(256).astype(np.float32))
            except Exception as exc:
                errors.append(exc)

        def consumer():
            try:
                for _ in range(50):
                    buf.drain_seconds(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
