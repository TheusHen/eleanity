from __future__ import annotations

from pathlib import Path

from eleanity.core.engine import RESULT_SCHEMA_VERSION, CompareEngine
from eleanity.core.observe import observe
from eleanity.core.store import (
    write_github_annotations,
    write_junit,
)
from eleanity.core.store import (
    write_result_json as _write_result_json,
)
from eleanity.models.schemas import Comparison, Message, ObservationTrace, Scenario
from eleanity.policies.engine import PolicyEngine

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "observe",
    "compare_trace_layers",
    "run_compare",
    "run_ci",
    "write_junit",
    "write_github_annotations",
]


def compare_trace_layers(
    left: ObservationTrace,
    right: ObservationTrace,
    tolerance: float,
    scenario: Scenario | None = None,
) -> dict[str, Comparison]:
    if scenario is None:
        scenario = Scenario(
            name="ad-hoc",
            messages=[Message(role="user", content="x")],
            tolerance=tolerance,
        )
    else:
        scenario = scenario.model_copy(update={"tolerance": tolerance})
    return PolicyEngine(scenario).compare_layers(left, right)


def run_compare(
    model: str,
    backends: list[str],
    scenario: Scenario | None = None,
    baseline_backend: str | None = None,
    runs_dir: Path = Path(".eleanity/runs"),
    *,
    redact_prompts: bool = False,
    junit_path: Path | None = None,
    annotations_path: Path | None = None,
    parallel: bool = True,
    tokenizer_only: bool = False,
) -> tuple[str, list[ObservationTrace], object]:
    engine = CompareEngine(runs_dir=runs_dir, parallel=parallel, tokenizer_only=tokenizer_only)
    result = engine.compare(
        model,
        backends,
        scenario=scenario,
        baseline_backend=baseline_backend,
        redact_prompts=redact_prompts,
        junit_path=junit_path,
        annotations_path=annotations_path,
        tokenizer_only=tokenizer_only,
    )
    return result.run_id, result.traces, result.diagnosis


def run_ci(
    baseline_model: str,
    candidate_model: str,
    backend: str,
    scenario: Scenario | None = None,
    runs_dir: Path = Path(".eleanity/runs"),
    *,
    junit_path: Path | None = None,
    tokenizer_only: bool = False,
) -> tuple[str, dict[str, Comparison], object]:
    engine = CompareEngine(runs_dir=runs_dir, parallel=False, tokenizer_only=tokenizer_only)
    run_id, comparisons, diagnosis, _gates = engine.ci(
        baseline_model,
        candidate_model,
        backend,
        scenario=scenario,
        junit_path=junit_path,
        tokenizer_only=tokenizer_only,
    )
    return run_id, comparisons, diagnosis


def _write_result(target: Path, payload: dict, *, redact_prompts: bool = False) -> None:
    _write_result_json(target, payload, redact_prompts=redact_prompts)
