"""
utils/logger.py – Centralised logging configuration for the S2ST system.

Every module obtains its logger via:

    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Keep track of whether the root logger has already been configured so that
# repeated imports do not add duplicate handlers.
_configured: bool = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """
    Configure the root logger.

    Call this once at application start (e.g. from main.py or api/app.py).
    Subsequent calls are no-ops.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to a log file.  If *None*, logs go only to
                  stdout.
        fmt: Log-record format string.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(fmt=fmt, datefmt=_DATE_FORMAT)

    handlers: list[logging.Handler] = []

    # --- stdout handler ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    # --- optional file handler ---
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised (level=%s, file=%s)", level, log_file or "none"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.

    If the root logger has not been set up yet this will call
    :func:`setup_logging` with default settings so the application is still
    usable even if *main.py* has not been executed (e.g. during testing).

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
