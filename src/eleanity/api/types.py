"""Public result types for the programmatic API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eleanity.api.codes import EXIT_OK, exit_from_diagnosis
from eleanity.core.engine import CompareResult
from eleanity.gates.engine import GateEvaluation
from eleanity.models.schemas import ObservationTrace, ParityResult, Scenario


def _status_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


@dataclass
class CompareOutcome:
    """High-level compare/test/ci result — stable for pipeline embedding."""

    run_id: str
    status: str
    first_divergence: str | None
    coverage: float | None
    confidence: float | None
    gates_passed: bool | None
    model: str
    backends: list[str]
    baseline: str | None
    policy: str
    scenario_name: str
    traces: list[ObservationTrace]
    diagnosis: Any
    comparisons: dict[str, Any]
    consensus: dict[str, Any]
    path: Path | None
    timings: dict[str, float] = field(default_factory=dict)
    impact: Any = None
    formal_status: str | None = None
    verified_layers: list[str] = field(default_factory=list)
    not_verified_layers: list[Any] = field(default_factory=list)
    reproduction_command: str | None = None
    gate_evaluation: GateEvaluation | None = None
    engine_result: CompareResult | None = field(default=None, repr=False)

    @property
    def passed(self) -> bool:
        """True when status is a pass family and gates did not fail."""

        if self.gates_passed is False:
            return False
        return self.status in {
            ParityResult.PASS.value,
            ParityResult.PASS_WITH_TOLERANCE.value,
            ParityResult.PASS_WITH_LIMITED_COVERAGE.value,
        }

    @property
    def divergent(self) -> bool:
        return self.status == ParityResult.DIVERGENT.value

    @property
    def exit_code(self) -> int:
        return exit_from_diagnosis(self.diagnosis, gate_passed=self.gates_passed)

    def raise_for_status(self) -> CompareOutcome:
        """Raise :class:`ParityError` if not :attr:`passed` (config/divergent)."""

        from eleanity.api.errors import ConfigError, ParityError

        code = self.exit_code
        if code == EXIT_OK:
            return self
        if code == 2:
            raise ConfigError(f"compare error status={self.status} run_id={self.run_id}")
        raise ParityError(
            f"parity failed status={self.status} first_divergence={self.first_divergence} run_id={self.run_id}",
            outcome=self,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "formal_status": self.formal_status,
            "first_divergence": self.first_divergence,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "gates_passed": self.gates_passed,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "model": self.model,
            "backends": list(self.backends),
            "baseline": self.baseline,
            "policy": self.policy,
            "scenario_name": self.scenario_name,
            "impact": self.impact,
            "verified_layers": list(self.verified_layers),
            "not_verified_layers": list(self.not_verified_layers),
            "timings": dict(self.timings),
            "path": str(self.path) if self.path else None,
            "reproduction_command": self.reproduction_command,
            "consensus": self.consensus,
        }

    def quiet_line(self) -> str:
        cov = self.coverage if self.coverage is not None else ""
        conf = self.confidence if self.confidence is not None else ""
        fd = self.first_divergence or "none"
        gates = self.gates_passed if self.gates_passed is not None else ""
        return (
            f"status={self.status} impact={_impact_label(self.impact)} "
            f"coverage={cov} confidence={conf} first_divergence={fd} "
            f"gates={gates} run_id={self.run_id}"
        )


def _impact_label(impact: Any) -> str:
    if impact is None:
        return ""
    if isinstance(impact, dict):
        return str(impact.get("impact") or impact.get("level") or "")
    return str(getattr(impact, "impact", impact))


def outcome_from_engine(
    result: CompareResult,
    *,
    model: str,
    backends: list[str],
    baseline: str | None,
    policy: str,
    scenario: Scenario | None,
) -> CompareOutcome:
    diagnosis = result.diagnosis
    status = _status_str(getattr(diagnosis, "status", None))
    cov = getattr(diagnosis, "coverage", None)
    coverage_pct: float | None = None
    if isinstance(cov, dict):
        raw = cov.get("required_coverage_percent")
        coverage_pct = float(raw) if raw is not None else None
    elif isinstance(cov, (int, float)):
        coverage_pct = float(cov)

    impact = getattr(diagnosis, "impact", None)
    formal = getattr(diagnosis, "formal_status", None)
    gates = result.gate_evaluation
    gates_passed = gates.passed if gates is not None else None

    repro = None
    # Prefer reproduction from on-disk result if present
    if result.path is not None:
        result_json = result.path / "result.json"
        if result_json.is_file():
            try:
                import json

                payload = json.loads(result_json.read_text(encoding="utf-8"))
                repro = payload.get("reproduction_command")
            except (OSError, ValueError):
                repro = None

    return CompareOutcome(
        run_id=result.run_id,
        status=status,
        first_divergence=getattr(diagnosis, "first_divergence", None),
        coverage=coverage_pct,
        confidence=getattr(diagnosis, "confidence", None),
        gates_passed=gates_passed,
        model=model,
        backends=list(backends),
        baseline=baseline,
        policy=policy,
        scenario_name=(scenario.name if scenario else "compare"),
        traces=list(result.traces),
        diagnosis=diagnosis,
        comparisons=result.comparisons,
        consensus=result.consensus or {},
        path=result.path,
        timings=dict(result.timings or {}),
        impact=impact,
        formal_status=_status_str(formal) if formal else status,
        verified_layers=list(getattr(diagnosis, "verified_layers", None) or []),
        not_verified_layers=list(getattr(diagnosis, "not_verified_layers", None) or []),
        reproduction_command=repro,
        gate_evaluation=gates,
        engine_result=result,
    )


@dataclass
class ScenarioResult:
    name: str
    outcome: CompareOutcome

    @property
    def status(self) -> str:
        return self.outcome.status

    @property
    def run_id(self) -> str:
        return self.outcome.run_id

    @property
    def exit_code(self) -> int:
        return self.outcome.exit_code


@dataclass
class TestReport:
    """Result of running many scenarios (``client.test``)."""

    results: list[ScenarioResult] = field(default_factory=list)
    fail_fast: bool = False

    @property
    def passed(self) -> bool:
        return all(r.outcome.passed for r in self.results)

    @property
    def exit_code(self) -> int:
        from eleanity.api.codes import exit_from_batch

        failed = sum(1 for r in self.results if not r.outcome.passed)
        had_error = any(r.outcome.status == ParityResult.ERROR.value for r in self.results)
        return exit_from_batch(failed, had_error=had_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.passed,
            "exit_code": self.exit_code,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "run_id": r.run_id,
                    "exit": r.exit_code,
                    "first_divergence": r.outcome.first_divergence,
                }
                for r in self.results
            ],
        }


@dataclass
class BackendHealth:
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    ok: bool
    version: str
    python: str
    adapters: list[str]
    project: str | None
    checks: dict[str, str] = field(default_factory=dict)
    backends: list[BackendHealth] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "python": self.python,
            "adapters": list(self.adapters),
            "project": self.project,
            "checks": dict(self.checks),
            "backends": [b.to_dict() for b in self.backends],
        }
