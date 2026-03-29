"""
app_ui.py – Streamlit UI for the Real-Time Speech-to-Speech Translation System.

Launch with:
    streamlit run app_ui.py

The app runs the S2ST pipeline directly inside the Streamlit process –
no separate FastAPI server is required.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path regardless of where Streamlit is invoked
sys.path.insert(0, str(Path(__file__).parent))

from config import Config  # noqa: E402
from utils.logger import setup_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="S2ST – Real-Time Translation",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Supported languages (BCP-47 code → display name)
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
}

LANG_DISPLAY = [f"{name} ({code})" for code, name in LANGUAGES.items()]
LANG_CODES = list(LANGUAGES.keys())


def _display_to_code(display: str) -> str:
    """Extract the BCP-47 code from a display string like 'English (en)'."""
    return display.split("(")[-1].rstrip(")")


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "pipeline": None,
        "is_running": False,
        "events": [],          # list of TranslationEvent dicts
        "event_queue": queue.Queue(),
        "error": None,
        "total_segments": 0,
        "start_time": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _on_result(event) -> None:
    """Callback invoked by the pipeline worker thread for each translation."""
    from dataclasses import asdict

    st.session_state.event_queue.put(asdict(event))


def _drain_queue() -> None:
    """Move all pending events from the thread-safe queue into session state."""
    q: queue.Queue = st.session_state.event_queue
    while not q.empty():
        try:
            event = q.get_nowait()
            st.session_state.events.insert(0, event)  # newest first
            st.session_state.total_segments += 1
        except queue.Empty:
            break


def _start_pipeline(cfg: Config) -> None:
    """Instantiate and start the S2ST pipeline in a background thread."""
    from pipeline.pipeline import S2STPipeline

    setup_logging(cfg.pipeline.log_level)

    try:
        pipeline = S2STPipeline(cfg, on_result=_on_result)
        pipeline.start()
        st.session_state.pipeline = pipeline
        st.session_state.is_running = True
        st.session_state.error = None
        st.session_state.start_time = time.time()
    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.is_running = False


def _stop_pipeline() -> None:
    """Stop the running pipeline."""
    import logging

    pipeline = st.session_state.pipeline
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception as exc:
            logging.getLogger(__name__).warning("Error stopping pipeline: %s", exc)
    st.session_state.pipeline = None
    st.session_state.is_running = False


# ---------------------------------------------------------------------------
# Sidebar – configuration panel
# ---------------------------------------------------------------------------

def render_sidebar() -> Config:
    """Render the sidebar and return a configured Config object."""
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # --- Target language ---
    st.sidebar.subheader("🌍 Translation")
    default_lang_idx = LANG_CODES.index("en") if "en" in LANG_CODES else 0
    target_display = st.sidebar.selectbox(
        "Target language",
        options=LANG_DISPLAY,
        index=default_lang_idx,
        help="The language you want the speech translated into.",
        disabled=st.session_state.is_running,
    )
    target_lang = _display_to_code(target_display)

    translation_backend = st.sidebar.selectbox(
        "Translation backend",
        options=["marian", "google", "seamless"],
        index=0,
        help=(
            "marian: offline MarianMT model (default)\n"
            "google: requires GOOGLE_TRANSLATE_API_KEY\n"
            "seamless: SeamlessM4T (requires GPU for best results)"
        ),
        disabled=st.session_state.is_running,
    )

    # --- ASR ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎤 Speech Recognition")
    model_size = st.sidebar.selectbox(
        "Whisper model size",
        options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1,
        help="Larger models are more accurate but slower and use more memory.",
        disabled=st.session_state.is_running,
    )
    device = st.sidebar.radio(
        "Compute device",
        options=["cpu", "cuda"],
        index=0,
        help="Use 'cuda' if you have an NVIDIA GPU with CUDA installed.",
        disabled=st.session_state.is_running,
    )

    # --- TTS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔊 Text-to-Speech")
    tts_backend = st.sidebar.selectbox(
        "TTS backend",
        options=["gtts", "pyttsx3", "coqui", "elevenlabs"],
        index=0,
        help=(
            "gtts: Google TTS (requires internet, natural voice)\n"
            "pyttsx3: offline system TTS\n"
            "coqui: neural TTS (requires TTS package)\n"
            "elevenlabs: ElevenLabs API (requires API key)"
        ),
        disabled=st.session_state.is_running,
    )

    # --- Audio & VAD ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔉 Audio & VAD")
    noise_suppression = st.sidebar.toggle(
        "Noise suppression",
        value=True,
        help="Apply spectral noise reduction before speech detection.",
        disabled=st.session_state.is_running,
    )
    vad_threshold = st.sidebar.slider(
        "VAD sensitivity threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values require the model to be more confident before declaring speech.",
        disabled=st.session_state.is_running,
    )

    # --- Logging ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🪵 Logging")
    log_level = st.sidebar.selectbox(
        "Log level",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        index=1,
        disabled=st.session_state.is_running,
    )

    # Build Config from selections
    cfg = Config()
    cfg.translation.target_language = target_lang
    cfg.translation.backend = translation_backend
    cfg.tts.language = target_lang
    cfg.tts.backend = tts_backend
    cfg.asr.model_size = model_size
    cfg.asr.device = device
    cfg.noise_suppression.enabled = noise_suppression
    cfg.vad.threshold = vad_threshold
    cfg.pipeline.log_level = log_level

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "ℹ️ Settings are locked while the pipeline is running. "
        "Stop the pipeline to change them."
    )

    return cfg


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render_header() -> None:
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.title("🎙️ Real-Time Speech-to-Speech Translation")
        st.caption(
            "Speak into your microphone – the system will transcribe, translate, "
            "and play back the result in the target language automatically."
        )
    with col_badge:
        if st.session_state.is_running:
            st.markdown(
                "<div style='text-align:right; padding-top:1.5rem;'>"
                "<span style='background:#28a745;color:white;padding:6px 14px;"
                "border-radius:20px;font-weight:600;'>● LIVE</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:right; padding-top:1.5rem;'>"
                "<span style='background:#6c757d;color:white;padding:6px 14px;"
                "border-radius:20px;font-weight:600;'>◼ STOPPED</span></div>",
                unsafe_allow_html=True,
            )


def render_controls(cfg: Config) -> None:
    """Render Start / Stop / Clear buttons."""
    col_start, col_stop, col_clear, col_spacer = st.columns([1, 1, 1, 4])

    with col_start:
        if st.button(
            "▶ Start",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.is_running,
        ):
            with st.spinner("Initializing pipeline…"):
                t = threading.Thread(target=_start_pipeline, args=(cfg,), daemon=True)
                t.start()
                t.join(timeout=30)
                if t.is_alive():
                    st.session_state.error = (
                        "Pipeline initialization timed out after 30 s. "
                        "Check that your microphone is connected and models are accessible."
                    )
            st.rerun()

    with col_stop:
        if st.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=not st.session_state.is_running,
        ):
            with st.spinner("Stopping pipeline…"):
                _stop_pipeline()
            st.rerun()

    with col_clear:
        if st.button("🗑 Clear feed", use_container_width=True):
            st.session_state.events = []
            st.session_state.total_segments = 0
            st.rerun()


def render_status_metrics(cfg: Config) -> None:
    """Show at-a-glance statistics."""
    elapsed = (
        int(time.time() - st.session_state.start_time)
        if st.session_state.start_time and st.session_state.is_running
        else 0
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Status", "🟢 Running" if st.session_state.is_running else "🔴 Stopped")
    m2.metric("Target language", LANGUAGES.get(cfg.translation.target_language, cfg.translation.target_language))
    m3.metric("Segments translated", st.session_state.total_segments)
    m4.metric("Uptime", f"{elapsed // 60:02d}:{elapsed % 60:02d}" if elapsed else "—")
    m5.metric("ASR model", cfg.asr.model_size)


def render_feed() -> None:
    """Render the live translation feed."""
    st.subheader("📋 Translation Feed")

    if st.session_state.error:
        st.error(f"❌ Pipeline error: {st.session_state.error}")

    if not st.session_state.events:
        st.info(
            "No translations yet. "
            "Press **▶ Start** and speak into your microphone to begin."
        )
        return

    # Display each event as a card
    for event in st.session_state.events[:50]:  # show most recent 50
        src = event.get("source_language", "?")
        tgt = event.get("target_language", "?")
        src_text = event.get("source_text", "")
        tgt_text = event.get("translated_text", "")
        audio_ms = event.get("audio_duration_ms", 0)
        proc_ms = event.get("processing_time_ms", 0)

        with st.container(border=True):
            col_src, col_arrow, col_tgt = st.columns([5, 1, 5])
            with col_src:
                st.markdown(f"**🗣 [{src.upper()}]** — *heard*")
                st.write(src_text)
            with col_arrow:
                st.markdown(
                    "<div style='text-align:center;font-size:2rem;padding-top:0.5rem'>→</div>",
                    unsafe_allow_html=True,
                )
            with col_tgt:
                st.markdown(f"**🔊 [{tgt.upper()}]** — *translated*")
                st.write(tgt_text)
            st.caption(
                f"Audio: {audio_ms:.0f} ms  |  Processing: {proc_ms:.0f} ms"
            )


def render_how_to() -> None:
    """Render a collapsible quick-start guide."""
    with st.expander("ℹ️ How to use this app", expanded=False):
        st.markdown(
            """
**Step 1 – Configure**
> Use the sidebar on the left to choose your target language, ASR model size,
> TTS backend, and audio settings.  All settings are locked while the pipeline
> is running.

**Step 2 – Start**
> Click the **▶ Start** button.  The system loads the required models
> (this may take a few seconds the first time) and begins listening to your
> microphone.

**Step 3 – Speak**
> Talk naturally into your microphone.  The app automatically detects speech,
> transcribes it, translates it, and plays the translated audio through your
> speakers.  Each completed utterance appears in the *Translation Feed* below.

**Step 4 – Stop**
> Click **⏹ Stop** to shut down the pipeline and release the microphone.

---

**Troubleshooting**
- Make sure your microphone is connected and not muted.
- If you see a *pipeline error*, check the terminal output for details.
- For offline use choose `pyttsx3` (TTS) and `marian` (translation).
- For best accuracy use a larger Whisper model such as `small` or `medium`.
"""
        )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Sidebar (returns configured Config object)
    cfg = render_sidebar()

    # 2. Header
    render_header()
    st.markdown("---")

    # 3. Controls
    render_controls(cfg)
    st.markdown("---")

    # 4. Status metrics
    render_status_metrics(cfg)
    st.markdown("---")

    # 5. How-to guide
    render_how_to()

    # 6. Drain any events queued by the pipeline thread
    _drain_queue()

    # 7. Live feed
    render_feed()

    # 8. Auto-refresh every second while the pipeline is running
    if st.session_state.is_running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
