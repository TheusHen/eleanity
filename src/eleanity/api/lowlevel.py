"""Low-level stable building blocks (layer C).

Use these when you need observe-only, custom adapters, or layer diffs
without the full :class:`~eleanity.api.client.Eleanity` client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eleanity.adapters import adapter_for, available_adapters, create_adapter, register_adapter
from eleanity.core.observe import observe as _observe_pipeline
from eleanity.core.run import compare_trace_layers
from eleanity.diagnosers import diagnose
from eleanity.gates.engine import evaluate_gates
from eleanity.models.schemas import Comparison, Message, ModelSpec, ObservationTrace, ParityProfile, Scenario
from eleanity.policies.engine import PolicyEngine
from eleanity.scenarios.loader import load_scenarios
from eleanity.scenarios.suites import load_suite

__all__ = [
    "observe",
    "observe_backend",
    "compare_traces",
    "compare_trace_layers",
    "diagnose_traces",
    "evaluate_gates",
    "create_adapter",
    "adapter_for",
    "register_adapter",
    "available_adapters",
    "load_scenarios",
    "load_suite",
    "make_scenario",
    "PolicyEngine",
]


def make_scenario(
    *,
    name: str = "ad-hoc",
    messages: list[dict[str, str]] | None = None,
    observe: list[str] | None = None,
    policy: str = "strict",
    model: str | None = None,
    parameters: dict[str, Any] | None = None,
    backends: list[str] | None = None,
) -> Scenario:
    """Build a :class:`~eleanity.models.schemas.Scenario` without YAML."""

    params = {"temperature": 0, "max_tokens": 32, "seed": 42}
    if parameters:
        params.update(parameters)
    spec = ModelSpec(id=model) if model else None
    return Scenario(
        name=name,
        messages=[Message(**message) for message in messages] if messages else [Message(role="user", content="Hello")],
        observe=observe
        or [
            "artifact",
            "template",
            "special_tokens",
            "tokens",
            "generation",
        ],
        parity_profile=ParityProfile(policy),
        parameters=params,
        model=spec,
        backends=backends or [],
    )


def observe(
    adapter: Any,
    scenario: Scenario,
    model: str,
    *,
    baseline_backend: str | None = None,
) -> ObservationTrace:
    """Run the observation pipeline on an already-constructed adapter."""

    return _observe_pipeline(
        adapter,
        scenario,
        model,
        baseline_backend=baseline_backend,
    )


def observe_backend(
    backend: str,
    model: str,
    scenario: Scenario | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    tokenizer_only: bool = False,
    model_spec: ModelSpec | None = None,
) -> ObservationTrace:
    """Instantiate *backend* and observe *model* under *scenario*."""

    sc = scenario or make_scenario(model=model)
    adapter = create_adapter(
        backend,
        model,
        model_spec=model_spec or sc.model,
        scenario=sc,
        tokenizer_only=tokenizer_only,
        base_url=base_url,
        api_key=api_key,
    )
    return observe(
        adapter,
        sc,
        model,
        baseline_backend=backend,
    )


def compare_traces(
    left: ObservationTrace,
    right: ObservationTrace,
    scenario: Scenario | None = None,
    *,
    policy: str | None = None,
    tolerance: float | None = None,
) -> dict[str, Comparison]:
    """Compare two observation traces layer-by-layer under a policy/scenario."""

    sc = scenario or make_scenario()
    updates: dict[str, Any] = {}
    if policy is not None:
        updates["parity_profile"] = policy
    if tolerance is not None:
        updates["tolerance"] = tolerance
    if updates:
        sc = sc.model_copy(update=updates)
    return PolicyEngine(sc).compare_layers(left, right)


def diagnose_traces(traces: list[ObservationTrace]) -> Any:
    """Run the rule diagnoser on one or more traces."""

    return diagnose(traces)


def load_scenario_file(path: str | Path) -> list[Scenario]:
    """Load all scenarios from a YAML file or directory."""

    p = Path(path)
    if p.is_dir():
        scenarios: list[Scenario] = []
        for file_path in sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")):
            scenarios.extend(load_scenarios(file_path))
        return scenarios
    return load_scenarios(p)
