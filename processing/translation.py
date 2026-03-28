"""
processing/translation.py – Text translation between languages.

Supported backends:

1. **marian** – Helsinki-NLP MarianMT models (local, free, offline).
2. **google** – Google Cloud Translation REST API (requires API key).
3. **seamless** – Meta SeamlessM4T model (requires GPU for best performance).
4. **passthrough** – Returns the source text unchanged (development / testing).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import TranslationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Translator:
    """
    Language-agnostic translation interface.

    Args:
        cfg: :class:`~config.TranslationConfig` instance.
    """

    def __init__(self, cfg: TranslationConfig) -> None:
        self._cfg = cfg
        self._backend: str = cfg.backend
        # MarianMT: cache loaded models keyed by (src, tgt)
        self._marian_cache: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Args:
            text: Input text to translate.
            source_lang: BCP-47 source language code.  Overrides config.
            target_lang: BCP-47 target language code.  Overrides config.

        Returns:
            Translated text string.
        """
        if not text or not text.strip():
            return text

        src = source_lang or self._cfg.source_language
        tgt = target_lang or self._cfg.target_language

        # If source and target are the same, skip translation
        if src == tgt:
            return text
        # If source is "auto" / "unknown", skip
        if src in ("auto", "unknown"):
            logger.debug("Translation skipped: source language is '%s'.", src)
            return text

        try:
            return self._dispatch(text, src, tgt)
        except Exception as exc:
            logger.warning("Translation failed (%s): %s", self._backend, exc)
            return text   # graceful fallback: return source text

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, text: str, src: str, tgt: str) -> str:
        if self._backend == "marian":
            return self._translate_marian(text, src, tgt)
        if self._backend == "google":
            return self._translate_google(text, src, tgt)
        if self._backend == "seamless":
            return self._translate_seamless(text, src, tgt)
        # passthrough
        return text

    # ------------------------------------------------------------------
    # MarianMT
    # ------------------------------------------------------------------

    def _translate_marian(self, text: str, src: str, tgt: str) -> str:
        """Translate using HuggingFace MarianMT (downloaded on demand)."""
        try:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers not installed") from exc

        model_name = self._cfg.marian_model_template.format(src=src, tgt=tgt)
        cache_key = (src, tgt)

        if cache_key not in self._marian_cache:
            logger.info("Loading MarianMT model: %s", model_name)
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.eval()
                self._marian_cache[cache_key] = (tokenizer, model)
            except Exception as exc:
                raise RuntimeError(
                    f"MarianMT model '{model_name}' not available: {exc}"
                ) from exc

        tokenizer, model = self._marian_cache[cache_key]

        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)

        import torch  # type: ignore

        with torch.no_grad():
            translated_tokens = model.generate(**inputs)

        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        logger.debug("MarianMT %s→%s: %r → %r", src, tgt, text[:40], translated[:40])
        return translated

    # ------------------------------------------------------------------
    # Google Translate REST API
    # ------------------------------------------------------------------

    def _translate_google(self, text: str, src: str, tgt: str) -> str:
        """Translate via Google Cloud Translation API v2."""
        import urllib.parse
        import urllib.request
        import json

        api_key = self._cfg.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_TRANSLATE_API_KEY not configured.")

        url = "https://translation.googleapis.com/language/translate/v2"
        params = urllib.parse.urlencode(
            {
                "q": text,
                "source": src,
                "target": tgt,
                "format": "text",
                "key": api_key,
            }
        )
        req = urllib.request.Request(
            f"{url}?{params}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        translated = data["data"]["translations"][0]["translatedText"]
        logger.debug("Google %s→%s: %r → %r", src, tgt, text[:40], translated[:40])
        return translated

    # ------------------------------------------------------------------
    # SeamlessM4T
    # ------------------------------------------------------------------

    def _translate_seamless(self, text: str, src: str, tgt: str) -> str:
        """Translate using Meta SeamlessM4T (text-to-text mode)."""
        try:
            from transformers import SeamlessM4TForTextToText, AutoProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers ≥ 4.33 required for SeamlessM4T") from exc

        import torch  # type: ignore

        model_name = "facebook/hf-seamless-m4t-medium"
        # Cache model lazily
        if not hasattr(self, "_seamless_model"):
            logger.info("Loading SeamlessM4T model (first use)…")
            self._seamless_processor = AutoProcessor.from_pretrained(model_name)
            self._seamless_model = SeamlessM4TForTextToText.from_pretrained(model_name)
            self._seamless_model.eval()

        inputs = self._seamless_processor(text=text, src_lang=src, return_tensors="pt")
        with torch.no_grad():
            output_tokens = self._seamless_model.generate(
                **inputs,
                tgt_lang=tgt,
                generate_speech=False,
            )
        translated = self._seamless_processor.decode(
            output_tokens[0].tolist(), skip_special_tokens=True
        )
        logger.debug("SeamlessM4T %s→%s: %r → %r", src, tgt, text[:40], translated[:40])
        return translated
