"""Centralised logging configuration for the Enterprise Knowledge-Base.

Provides a factory function that returns a named logger writing to both
the console (stdout) and a rotating log file at ``logs/system.log``.

Usage::

    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("System started")
"""

import logging
import os
import sys
from typing import Optional

_LOG_DIR = "logs"
_LOG_FILE = os.path.join(_LOG_DIR, "system.log")
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger instance.

    On first call for a given *name*, the logger is set up with two handlers:

    1. **Console** — writes to ``sys.stdout`` at INFO level.
    2. **File** — appends to ``logs/system.log`` (UTF-8, auto-created).

    Subsequent calls with the same *name* return the same logger without
    adding duplicate handlers.

    Args:
        name: Logger name.  Pass ``__name__`` from the calling module for
              automatic hierarchical naming.  Defaults to ``"EnterpriseKB"``.

    Returns:
        A ``logging.Logger`` ready to use.
    """
    logger = logging.getLogger(name or "EnterpriseKB")

    # Avoid adding handlers more than once.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (create directory if needed)
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file at %s", _LOG_FILE)

    return logger