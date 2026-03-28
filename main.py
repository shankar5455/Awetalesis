"""
main.py – Entry point for the Real-Time Speech-to-Speech Translation System.

Usage:
    # Run the full pipeline (microphone → speakers):
    python main.py

    # Start the FastAPI server (includes WebSocket live feed):
    python main.py --api

    # Override the target language:
    python main.py --target fr

    # Use a different Whisper model size:
    python main.py --model small

    # Show all available options:
    python main.py --help
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from config import config
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and update the global config."""
    parser = argparse.ArgumentParser(
        description="Real-Time Speech-to-Speech Translation System (S2ST)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the FastAPI server instead of running the pipeline directly.",
    )
    parser.add_argument(
        "--target",
        default=config.translation.target_language,
        help="Target language code (BCP-47), e.g. 'en', 'de', 'fr'.",
    )
    parser.add_argument(
        "--model",
        default=config.asr.model_size,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Faster-Whisper model size.",
    )
    parser.add_argument(
        "--device",
        default=config.asr.device,
        choices=["cpu", "cuda"],
        help="Compute device for Whisper.",
    )
    parser.add_argument(
        "--tts-backend",
        default=config.tts.backend,
        choices=["gtts", "pyttsx3", "coqui", "elevenlabs"],
        help="TTS engine to use.",
    )
    parser.add_argument(
        "--no-noise-suppression",
        action="store_true",
        help="Disable noise suppression.",
    )
    parser.add_argument(
        "--log-level",
        default=config.pipeline.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--host",
        default=config.api.host,
        help="API server host (only used with --api).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.api.port,
        help="API server port (only used with --api).",
    )
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """Apply parsed CLI arguments to the global config singleton."""
    config.translation.target_language = args.target
    config.tts.language = args.target
    config.asr.model_size = args.model
    config.asr.device = args.device
    config.tts.backend = args.tts_backend
    config.pipeline.log_level = args.log_level
    config.api.host = args.host
    config.api.port = args.port
    if args.no_noise_suppression:
        config.noise_suppression.enabled = False


def run_pipeline() -> None:
    """Run the pipeline directly in the terminal (no API server)."""
    from pipeline.pipeline import S2STPipeline, TranslationEvent

    logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)

    def on_result(event: TranslationEvent) -> None:
        print(
            f"\n[{event.source_language} → {event.target_language}]\n"
            f"  Heard   : {event.source_text}\n"
            f"  Translated: {event.translated_text}\n"
            f"  ({event.audio_duration_ms:.0f} ms audio, "
            f"{event.processing_time_ms:.0f} ms processing)\n"
        )

    pipeline = S2STPipeline(config, on_result=on_result)

    # Graceful shutdown on Ctrl-C / SIGTERM
    stop_event = __import__("threading").Event()

    def _signal_handler(sig, frame):
        print("\n\nStopping pipeline…")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 60)
    print(" Real-Time Speech-to-Speech Translation")
    print(f" Target language : {config.translation.target_language}")
    print(f" ASR model       : {config.asr.model_size}")
    print(f" TTS backend     : {config.tts.backend}")
    print(f" Noise suppression: {'on' if config.noise_suppression.enabled else 'off'}")
    print(" Press Ctrl-C to stop.")
    print("=" * 60)

    pipeline.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        pipeline.stop()


def run_api(host: str, port: int) -> None:
    """Start the FastAPI / Uvicorn server."""
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("uvicorn not installed.  Run: pip install uvicorn")
        sys.exit(1)

    print(f"Starting API server on http://{host}:{port}")
    print(f"  OpenAPI docs : http://{host}:{port}/docs")
    print(f"  WebSocket    : ws://{host}:{port}/ws/translate")

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=config.api.reload,
        log_level=config.pipeline.log_level.lower(),
    )


def main() -> None:
    args = parse_args()
    apply_args(args)
    setup_logging(config.pipeline.log_level)

    if args.api:
        run_api(args.host, args.port)
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
