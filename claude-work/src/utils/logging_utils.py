"""Logging setup helpers."""

import logging
import os
import sys


def configure_logging() -> None:
    """
    Configure root logging.

    Controlled by LOG_LEVEL env var (default: INFO).
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Avoid clobbering existing handlers (e.g., tests or embedding apps).
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
