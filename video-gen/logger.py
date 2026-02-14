"""Logging setup — console (INFO) + file (DEBUG) handlers."""

import logging
import os


def setup_logger(session_id: str) -> logging.Logger:
    """Create and configure logger with console and file outputs."""
    logger = logging.getLogger(f"video_agent.{session_id}")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # Console handler — INFO level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler — DEBUG level
    log_dir = os.path.join("output", session_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "agent.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
