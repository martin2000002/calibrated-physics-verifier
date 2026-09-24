"""Logging setup; called once by the CLI entry point (no configuration at import time)."""

from __future__ import annotations

import logging
from typing import TextIO

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str, stream: TextIO | None = None) -> logging.Logger:
    """Attach one handler to the ``physverify`` logger, replacing any previous one."""
    name = level.upper()
    if name not in LOG_LEVELS:
        raise ValueError(f"log level must be one of {LOG_LEVELS}, got {level!r}")
    logger = logging.getLogger("physverify")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(name)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"physverify.{name}")
