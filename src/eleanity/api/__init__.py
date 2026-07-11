"""Public programmatic API for Eleanity.

High-level (B)::

    from eleanity import Eleanity
    client = Eleanity()
    result = client.compare(model="demo", backends=["fake", "fake"])

Low-level (C)::

    from eleanity.api import observe_backend, compare_traces, make_scenario
"""

from eleanity.api.client import Eleanity
from eleanity.api.codes import (
    EXIT_CONFIG,
    EXIT_DIVERGENT,
    EXIT_OK,
    exit_from_batch,
    exit_from_diagnosis,
)
from eleanity.api.errors import ConfigError, EleanityAPIError, NotFoundError, ParityError
from eleanity.api.lowlevel import (
    adapter_for,
    available_adapters,
    compare_trace_layers,
    compare_traces,
    create_adapter,
    diagnose_traces,
    evaluate_gates,
    load_scenario_file,
    load_scenarios,
    load_suite,
    make_scenario,
    observe,
    observe_backend,
    register_adapter,
)
from eleanity.api.types import (
    BackendHealth,
    CompareOutcome,
    DoctorReport,
    ScenarioResult,
    TestReport,
)

__all__ = [
    # client
    "Eleanity",
    # results
    "CompareOutcome",
    "TestReport",
    "ScenarioResult",
    "DoctorReport",
    "BackendHealth",
    # errors / codes
    "EleanityAPIError",
    "ConfigError",
    "ParityError",
    "NotFoundError",
    "EXIT_OK",
    "EXIT_DIVERGENT",
    "EXIT_CONFIG",
    "exit_from_diagnosis",
    "exit_from_batch",
    # low-level
    "observe",
    "observe_backend",
    "compare_traces",
    "compare_trace_layers",
    "diagnose_traces",
    "evaluate_gates",
    "make_scenario",
    "load_scenarios",
    "load_scenario_file",
    "load_suite",
    "create_adapter",
    "adapter_for",
    "register_adapter",
    "available_adapters",
]
