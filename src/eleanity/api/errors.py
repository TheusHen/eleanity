"""Public API exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eleanity.api.types import CompareOutcome


class EleanityAPIError(Exception):
    """Base error for the public Python API."""


class ConfigError(EleanityAPIError):
    """Invalid configuration, missing dependency, or ERROR status (exit 2)."""


class ParityError(EleanityAPIError):
    """Parity DIVERGENT or gate failure (exit 1)."""

    def __init__(self, message: str, *, outcome: CompareOutcome | None = None):
        super().__init__(message)
        self.outcome = outcome


class NotFoundError(EleanityAPIError):
    """Run, scenario, or resource not found."""
