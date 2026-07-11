"""Eleanity — same model, same input, find the first divergence.

Product surfaces:
  - CLI: ``eleanity`` console script
  - Python API: :class:`eleanity.Eleanity` client (+ low-level helpers in :mod:`eleanity.api`)
"""

from eleanity.version import __version__
from eleanity.api import (  # noqa: I001 — version first avoids circular imports
    EXIT_CONFIG,
    EXIT_DIVERGENT,
    EXIT_OK,
    CompareOutcome,
    ConfigError,
    DoctorReport,
    Eleanity,
    EleanityAPIError,
    NotFoundError,
    ParityError,
    TestReport,
    compare_traces,
    make_scenario,
    observe_backend,
)

__all__ = [
    "__version__",
    "Eleanity",
    "CompareOutcome",
    "TestReport",
    "DoctorReport",
    "EleanityAPIError",
    "ConfigError",
    "ParityError",
    "NotFoundError",
    "EXIT_OK",
    "EXIT_DIVERGENT",
    "EXIT_CONFIG",
    "observe_backend",
    "compare_traces",
    "make_scenario",
]
