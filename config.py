"""
config.py – Central configuration for the S2ST pipeline.

All tuneable parameters are gathered here so that each module
imports only what it needs without holding its own magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Microphone / playback settings."""

    sample_rate: int = 16_000          # Hz – Silero VAD & Whisper prefer 16 kHz
    channels: int = 1                  # Mono
    chunk_ms: int = 30                 # Frame duration in milliseconds
    device_index: Optional[int] = None # None → system default input device
    dtype: str = "float32"             # PyAudio / sounddevice dtype

    @property
    def chunk_samples(self) -> int:
        """Number of samples per audio frame."""
        return int(self.sample_rate * self.chunk_ms / 1000)


# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

@dataclass
class VADConfig:
    """Silero VAD parameters."""

    threshold: float = 0.5           # Speech probability threshold (0–1)
    min_speech_ms: int = 250         # Minimum speech segment length (ms)
    min_silence_ms: int = 300        # Silence needed to end a segment (ms)
    max_segment_ms: int = 10_000     # Hard cap on a single segment (ms)
    window_size_samples: int = 512   # Processing window (must be 512 for Silero)


# ---------------------------------------------------------------------------
# Automatic Speech Recognition
# ---------------------------------------------------------------------------

@dataclass
class ASRConfig:
    """Faster-Whisper settings."""

    model_size: str = "base"          # tiny / base / small / medium / large-v2
    device: str = "cpu"               # cpu | cuda
    compute_type: str = "int8"        # float16 | int8 | float32
    beam_size: int = 5
    language: Optional[str] = None    # None = auto-detect per segment
    task: str = "transcribe"          # transcribe | translate (built-in Whisper)
    vad_filter: bool = False          # Let our own VAD handle this
    word_timestamps: bool = True


# ---------------------------------------------------------------------------
# Language Identification
# ---------------------------------------------------------------------------

@dataclass
class LIDConfig:
    """Language identification settings."""

    backend: str = "whisper"          # "whisper" | "langdetect" | "fasttext"
    confidence_threshold: float = 0.5
    fasttext_model_path: Optional[str] = None  # path to lid.176.bin


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

@dataclass
class TranslationConfig:
    """Translation settings."""

    backend: str = "marian"           # "marian" | "google" | "seamless"
    source_language: str = "auto"     # "auto" = use LID output
    target_language: str = "en"       # BCP-47 language tag
    google_api_key: Optional[str] = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    marian_model_template: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

@dataclass
class TTSConfig:
    """TTS engine settings."""

    backend: str = "gtts"             # "coqui" | "gtts" | "elevenlabs" | "pyttsx3"
    language: str = "en"              # Spoken output language
    coqui_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    elevenlabs_api_key: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # default voice
    output_sample_rate: int = 22_050  # Hz


# ---------------------------------------------------------------------------
# Noise Suppression
# ---------------------------------------------------------------------------

@dataclass
class NoiseSuppressionConfig:
    """Noise suppression settings."""

    enabled: bool = True
    backend: str = "noisereduce"      # "noisereduce" | "deepfilter" | "rnnoise"
    prop_decrease: float = 1.0        # noisereduce: proportion of noise to remove


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Top-level pipeline knobs."""

    buffer_max_seconds: float = 5.0   # Maximum audio buffer before forced flush
    worker_threads: int = 2
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    websocket_path: str = "/ws/translate"


# ---------------------------------------------------------------------------
# Root config – the single object imported everywhere
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Aggregates all sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    lid: LIDConfig = field(default_factory=LIDConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    noise_suppression: NoiseSuppressionConfig = field(default_factory=NoiseSuppressionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Module-level singleton
config = Config()
