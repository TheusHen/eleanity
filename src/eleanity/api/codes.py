"""Public exit-code contract (same as CLI)."""

from __future__ import annotations

from typing import Any

from eleanity.models.schemas import ParityResult

EXIT_OK = 0
EXIT_DIVERGENT = 1
EXIT_CONFIG = 2

__all__ = [
    "EXIT_OK",
    "EXIT_DIVERGENT",
    "EXIT_CONFIG",
    "exit_from_diagnosis",
    "exit_from_batch",
]


def exit_from_diagnosis(
    diagnosis: Any,
    *,
    gate_passed: bool | None = None,
) -> int:
    """Map a diagnosis (+ optional gate result) to process exit codes 0/1/2."""

    status = getattr(diagnosis, "status", None)
    value = str(getattr(status, "value", status or ""))
    if value == ParityResult.ERROR.value:
        return EXIT_CONFIG
    if gate_passed is False:
        return EXIT_DIVERGENT
    if value == ParityResult.DIVERGENT.value:
        return EXIT_DIVERGENT
    return EXIT_OK


def exit_from_batch(failed: int, *, had_error: bool = False) -> int:
    if had_error:
        return EXIT_CONFIG
    if failed:
        return EXIT_DIVERGENT
    return EXIT_OK
