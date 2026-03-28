"""
tests/test_translation.py – Tests for the translation module.
"""

import pytest

from config import TranslationConfig
from processing.translation import Translator


class TestTranslatorPassthrough:
    def test_same_language_returns_original(self):
        cfg = TranslationConfig(backend="passthrough", source_language="en", target_language="en")
        t = Translator(cfg)
        result = t.translate("Hello world", source_lang="en", target_lang="en")
        assert result == "Hello world"

    def test_unknown_source_returns_original(self):
        cfg = TranslationConfig(backend="passthrough", source_language="auto", target_language="de")
        t = Translator(cfg)
        result = t.translate("Hello world", source_lang="unknown", target_lang="de")
        assert result == "Hello world"

    def test_empty_string_returns_empty(self):
        cfg = TranslationConfig(backend="passthrough")
        t = Translator(cfg)
        assert t.translate("") == ""
        assert t.translate("   ") == "   "

    def test_passthrough_backend(self):
        cfg = TranslationConfig(backend="passthrough", target_language="fr")
        t = Translator(cfg)
        text = "some text"
        result = t.translate(text, source_lang="en", target_lang="fr")
        # passthrough always returns original
        assert result == text


class TestTranslatorConfig:
    def test_target_language_from_config(self):
        cfg = TranslationConfig(backend="passthrough", target_language="es")
        t = Translator(cfg)
        # target_lang not provided → uses config default
        result = t.translate("Hi", source_lang="en")
        # passthrough returns original
        assert result == "Hi"

    def test_override_target_language(self):
        cfg = TranslationConfig(backend="passthrough", target_language="de")
        t = Translator(cfg)
        # override at call site
        result = t.translate("Hi", source_lang="en", target_lang="fr")
        assert result == "Hi"  # passthrough


class TestTranslatorMarian:
    """MarianMT integration tests – skipped if transformers not installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("transformers")
        pytest.importorskip("sentencepiece")

    def test_en_to_de_short_sentence(self):
        cfg = TranslationConfig(
            backend="marian",
            source_language="en",
            target_language="de",
        )
        t = Translator(cfg)
        result = t.translate("Hello, how are you?", source_lang="en", target_lang="de")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be the same as English (unless model is unavailable)

    def test_fallback_on_missing_model(self, monkeypatch):
        """If a MarianMT model doesn't exist, translate() should return source text."""
        cfg = TranslationConfig(backend="marian", target_language="xx")
        t = Translator(cfg)
        text = "Test fallback"
        result = t.translate(text, source_lang="en", target_lang="xx")
        # Fallback: original text is returned
        assert result == text
