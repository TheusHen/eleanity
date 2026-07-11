from __future__ import annotations

from typing import Any

from eleanity.models.schemas import ParityResult

EXIT_OK = 0
EXIT_DIVERGENT = 1
EXIT_CONFIG = 2


def exit_from_diagnosis(
    diagnosis: Any,
    *,
    gate_passed: bool | None = None,
) -> int:
    """Unified exit codes for compare/test/ci/batch."""

    status = getattr(diagnosis, "status", None)
    value = status.value if hasattr(status, "value") else str(status or "")
    if value == ParityResult.ERROR.value:
        return EXIT_CONFIG
    if gate_passed is False:
        return EXIT_DIVERGENT
    if value == ParityResult.DIVERGENT.value:
        return EXIT_DIVERGENT
    # Limited coverage / inconclusive are still success for exit unless gates failed
    return EXIT_OK


def exit_from_batch(failed: int, *, had_error: bool = False) -> int:
    if had_error:
        return EXIT_CONFIG
    if failed:
        return EXIT_DIVERGENT
    return EXIT_OK
