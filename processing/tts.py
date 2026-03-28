"""
processing/tts.py – Text-to-Speech synthesis.

Supported backends (tried in preference order or selected via config):

1. **gtts** – Google Text-to-Speech (requires internet, no API key).
2. **pyttsx3** – Offline TTS using OS speech engines (no install needed on
   most platforms).
3. **coqui** – Coqui TTS deep-learning models (local, high quality).
4. **elevenlabs** – ElevenLabs cloud API (requires API key and internet).

Audio output is played back via ``sounddevice`` or ``pygame``; falls back to
writing a WAV file if neither is available.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
from typing import Optional

import numpy as np

from config import TTSConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------

def _play_audio_bytes(data: bytes, mime_type: str = "mp3") -> None:
    """Play audio bytes (MP3 or WAV) using available system libraries."""

    # Try pygame first
    try:
        import pygame  # type: ignore
        import pygame.mixer as mixer

        if not mixer.get_init():
            mixer.init()
        sound = pygame.mixer.Sound(io.BytesIO(data))
        sound.play()
        # Block until playback finishes
        while pygame.mixer.get_busy():
            pygame.time.wait(50)
        return
    except Exception:
        pass

    # Try sounddevice + soundfile
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(io.BytesIO(data))
        sd.play(audio, sr)
        sd.wait()
        return
    except Exception:
        pass

    # Last resort: write to a temp file and use playsound / ffplay
    try:
        import subprocess

        suffix = f".{mime_type}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        os.unlink(tmp_path)
        return
    except Exception:
        pass

    logger.warning(
        "No audio playback library available.  Install pygame, sounddevice, "
        "or ffplay to hear TTS output."
    )


def _play_wav_array(audio: np.ndarray, sample_rate: int) -> None:
    """Play a float32 numpy array directly via sounddevice."""
    try:
        import sounddevice as sd  # type: ignore

        sd.play(audio.astype(np.float32), sample_rate)
        sd.wait()
    except Exception as exc:
        logger.warning("Could not play audio array: %s", exc)


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Synthesise speech from text and optionally play it back.

    Args:
        cfg: :class:`~config.TTSConfig` instance.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._coqui_tts = None

        if cfg.backend == "coqui":
            self._init_coqui()

    # ------------------------------------------------------------------
    # Coqui TTS init
    # ------------------------------------------------------------------

    def _init_coqui(self) -> None:
        try:
            from TTS.api import TTS  # type: ignore

            logger.info("Loading Coqui TTS model: %s", self._cfg.coqui_model_name)
            self._coqui_tts = TTS(self._cfg.coqui_model_name, progress_bar=False)
            logger.info("Coqui TTS ready.")
        except ImportError:
            logger.warning(
                "TTS (Coqui) not installed; falling back to gTTS.  "
                "Install with: pip install TTS"
            )
            self._cfg = TTSConfig(
                backend="gtts",
                language=self._cfg.language,
            )
        except Exception as exc:
            logger.warning("Coqui TTS init failed (%s); falling back to gTTS.", exc)
            self._cfg = TTSConfig(
                backend="gtts",
                language=self._cfg.language,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        play: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Convert *text* to speech.

        Args:
            text: Text to synthesise.
            language: BCP-47 language code.  Overrides the config value.
            play: If *True*, the audio is played through the system speakers.

        Returns:
            float32 numpy audio array, or *None* on failure.
        """
        if not text or not text.strip():
            return None

        lang = language or self._cfg.language

        try:
            audio, sr = self._dispatch(text, lang)
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc, exc_info=True)
            return None

        if play:
            _play_wav_array(audio, sr)

        return audio

    def synthesize_to_bytes(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Return synthesised audio as raw WAV bytes.

        Args:
            text: Text to synthesise.
            language: BCP-47 language code.

        Returns:
            WAV bytes or *None* on failure.
        """
        import wave

        audio = self.synthesize(text, language=language, play=False)
        if audio is None:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._cfg.output_sample_rate)
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, text: str, lang: str):
        if self._cfg.backend == "gtts":
            return self._synth_gtts(text, lang)
        if self._cfg.backend == "pyttsx3":
            return self._synth_pyttsx3(text, lang)
        if self._cfg.backend == "coqui" and self._coqui_tts is not None:
            return self._synth_coqui(text, lang)
        if self._cfg.backend == "elevenlabs":
            return self._synth_elevenlabs(text, lang)
        # fallback
        return self._synth_gtts(text, lang)

    # ------------------------------------------------------------------
    # gTTS
    # ------------------------------------------------------------------

    def _synth_gtts(self, text: str, lang: str):
        """Google TTS (MP3 → float32 via pydub)."""
        from gtts import gTTS  # type: ignore

        mp3_buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return self._mp3_to_array(mp3_buf.read())

    def _mp3_to_array(self, mp3_bytes: bytes):
        """Decode MP3 bytes to float32 numpy array using pydub or soundfile."""
        try:
            from pydub import AudioSegment  # type: ignore

            seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            seg = seg.set_channels(1).set_frame_rate(self._cfg.output_sample_rate)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            return samples, self._cfg.output_sample_rate
        except ImportError:
            pass

        try:
            import soundfile as sf  # type: ignore

            audio, sr = sf.read(io.BytesIO(mp3_bytes))
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio.astype(np.float32), sr
        except Exception:
            pass

        # Last resort: save to disk and read back
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            tmp_mp3 = f.name
        wav_path = tmp_mp3.replace(".mp3", ".wav")
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sr
        finally:
            for p in (tmp_mp3, wav_path):
                if os.path.exists(p):
                    os.unlink(p)

    # ------------------------------------------------------------------
    # pyttsx3
    # ------------------------------------------------------------------

    def _synth_pyttsx3(self, text: str, lang: str):
        """Offline TTS via pyttsx3 (uses OS speech engine)."""
        import pyttsx3  # type: ignore
        import wave

        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sr
        finally:
            engine.stop()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Coqui TTS
    # ------------------------------------------------------------------

    def _synth_coqui(self, text: str, lang: str):
        """Coqui TTS – deep learning neural TTS."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self._coqui_tts.tts_to_file(text=text, file_path=tmp_path)
            import wave
            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # ElevenLabs
    # ------------------------------------------------------------------

    def _synth_elevenlabs(self, text: str, lang: str):
        """ElevenLabs cloud API."""
        import urllib.request
        import json

        api_key = self._cfg.elevenlabs_api_key
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not set.")

        voice_id = self._cfg.elevenlabs_voice_id
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = json.dumps(
            {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            mp3_bytes = resp.read()

        return self._mp3_to_array(mp3_bytes)
