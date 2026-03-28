"""
tests/test_vad.py – Tests for the Silero VAD wrapper.
"""

import numpy as np
import pytest

from config import VADConfig
from processing.vad import SileroVAD, SpeechSegment


SR = 16_000
WINDOW = 512


def _silent(n_windows: int) -> np.ndarray:
    """Generate silent audio (zeros) of n_windows × WINDOW samples."""
    return np.zeros(n_windows * WINDOW, dtype=np.float32)


def _speech(n_windows: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine-wave burst to simulate speech."""
    t = np.arange(n_windows * WINDOW, dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestSpeechSegment:
    def test_from_samples(self):
        audio = np.ones(SR, dtype=np.float32)
        seg = SpeechSegment.from_samples(audio, start=0, end=SR, sample_rate=SR)
        assert seg.duration_ms == pytest.approx(1000.0)
        assert len(seg.audio) == SR
        assert seg.start_sample == 0
        assert seg.end_sample == SR


class TestSileroVADInit:
    def test_init_does_not_raise(self):
        """VAD should load (or fall back to energy-based) without error."""
        cfg = VADConfig()
        vad = SileroVAD(cfg, sample_rate=SR)
        assert vad is not None

    def test_reset_does_not_raise(self):
        cfg = VADConfig()
        vad = SileroVAD(cfg, sample_rate=SR)
        vad.reset()


class TestSileroVADProcessing:
    """
    These tests patch the Silero model with the energy-based fallback to avoid
    requiring a real model download in CI.
    """

    def _make_vad(self, threshold: float = 0.5) -> SileroVAD:
        cfg = VADConfig(
            threshold=threshold,
            min_speech_ms=100,
            min_silence_ms=200,
        )
        vad = SileroVAD(cfg, sample_rate=SR)
        # Force energy-based fallback (no network in CI)
        vad._loaded = False
        vad._model = None
        return vad

    def test_silence_produces_no_segments(self):
        vad = self._make_vad()
        audio = _silent(200)  # 200 × 512 = ~6.4 s of silence
        for i in range(0, len(audio), WINDOW):
            chunk = audio[i: i + WINDOW]
            if len(chunk) == WINDOW:
                segs = vad.process_chunk(chunk)
                assert segs == []

    def test_speech_then_silence_produces_segment(self):
        vad = self._make_vad(threshold=0.01)  # low threshold so energy gate triggers
        # 1 second speech (500 ms above threshold)
        speech_frames = int(SR * 0.5) // WINDOW
        silence_frames = int(SR * 0.5) // WINDOW

        segments = []
        for _ in range(speech_frames):
            chunk = _speech(1, amplitude=0.5)
            segs = vad.process_chunk(chunk)
            segments.extend(segs)

        for _ in range(silence_frames):
            chunk = _silent(1)
            segs = vad.process_chunk(chunk)
            segments.extend(segs)

        # Should have produced at least one segment
        assert len(segments) >= 1
        for seg in segments:
            assert isinstance(seg, SpeechSegment)
            assert seg.duration_ms > 0
            assert seg.audio.dtype == np.float32

    def test_flush_returns_pending_speech(self):
        vad = self._make_vad(threshold=0.01)
        # Push speech without trailing silence
        for _ in range(50):
            chunk = _speech(1, amplitude=0.5)
            vad.process_chunk(chunk)

        flushed = vad.flush()
        # flush may return segments if speech was buffered
        for seg in flushed:
            assert isinstance(seg, SpeechSegment)

    def test_reset_clears_state(self):
        vad = self._make_vad(threshold=0.01)
        for _ in range(20):
            vad.process_chunk(_speech(1, amplitude=0.5))
        vad.reset()
        # After reset, fresh silence should produce no segments
        segs = vad.process_chunk(_silent(1))
        assert segs == []
