from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "eleanity") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("eleanity")
        if not root.handlers:
            root.addHandler(handler)
            root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Structured-ish log line for CI and local debugging."""

    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n")
        if " " in text or "=" in text:
            text = repr(text)
        parts.append(f"{key}={text}")
    logger.info(" ".join(parts))
