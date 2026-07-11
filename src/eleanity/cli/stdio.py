from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TextIO


def configure_cli_stdio(streams: Iterable[TextIO] | None = None) -> None:
    """Make rich CLI output safe on legacy and redirected encodings."""

    targets = streams if streams is not None else (sys.stdout, sys.stderr)
    for stream in targets:
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            # Captured/closed streams may not be reconfigurable. Rich can still
            # write to them using their existing policy.
            continue
