"""
tests/test_config.py – Smoke tests for config.py.
"""

from config import (
    AudioConfig,
    ASRConfig,
    Config,
    LIDConfig,
    NoiseSuppressionConfig,
    PipelineConfig,
    TranslationConfig,
    TTSConfig,
    VADConfig,
    config,
)


class TestAudioConfig:
    def test_chunk_samples(self):
        cfg = AudioConfig(sample_rate=16_000, chunk_ms=30)
        assert cfg.chunk_samples == 480  # 16000 * 30 / 1000

    def test_chunk_samples_40ms(self):
        cfg = AudioConfig(sample_rate=16_000, chunk_ms=40)
        assert cfg.chunk_samples == 640

    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16_000
        assert cfg.channels == 1
        assert cfg.chunk_ms == 30


class TestVADConfig:
    def test_defaults(self):
        cfg = VADConfig()
        assert 0.0 < cfg.threshold < 1.0
        assert cfg.min_speech_ms > 0
        assert cfg.window_size_samples == 512


class TestASRConfig:
    def test_defaults(self):
        cfg = ASRConfig()
        assert cfg.model_size in {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
        assert cfg.device in {"cpu", "cuda"}


class TestConfig:
    def test_singleton_is_config(self):
        assert isinstance(config, Config)

    def test_all_sub_configs_present(self):
        cfg = Config()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.vad, VADConfig)
        assert isinstance(cfg.asr, ASRConfig)
        assert isinstance(cfg.lid, LIDConfig)
        assert isinstance(cfg.translation, TranslationConfig)
        assert isinstance(cfg.tts, TTSConfig)
        assert isinstance(cfg.noise_suppression, NoiseSuppressionConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)

    def test_default_target_language(self):
        cfg = Config()
        assert isinstance(cfg.translation.target_language, str)
        assert len(cfg.translation.target_language) >= 2

    def test_independent_instances(self):
        """Two Config instances should be independent."""
        a = Config()
        b = Config()
        a.translation.target_language = "de"
        b.translation.target_language = "fr"
        assert a.translation.target_language != b.translation.target_language
