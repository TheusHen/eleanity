"""Self-consistency / determinism protocol before cross-backend comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eleanity.core.engine import CompareEngine
from eleanity.models.schemas import ParityResult, Scenario
from eleanity.spec.parity import FormalParityStatus


@dataclass
class StabilityReport:
    backend: str
    model: str
    repetitions: int
    agreements: int
    rate: float
    pairwise_status: list[str] = field(default_factory=list)
    first_divergence_layers: list[str | None] = field(default_factory=list)
    self_consistent: bool = True
    run_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "repetitions": self.repetitions,
            "agreements": self.agreements,
            "rate": self.rate,
            "self_consistent": self.self_consistent,
            "pairwise_status": self.pairwise_status,
            "first_divergence_layers": self.first_divergence_layers,
            "run_ids": self.run_ids,
            "notes": self.notes,
        }


@dataclass
class CrossStabilityReport:
    backend_a: StabilityReport
    backend_b: StabilityReport
    cross_agreement_rate: float | None
    cross_status: str | None
    attributable: bool
    conclusion: str
    cross_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_a": self.backend_a.to_dict(),
            "backend_b": self.backend_b.to_dict(),
            "cross_agreement_rate": self.cross_agreement_rate,
            "cross_status": self.cross_status,
            "attributable": self.attributable,
            "conclusion": self.conclusion,
            "cross_run_id": self.cross_run_id,
        }


def stabilize_backend(
    engine: CompareEngine,
    model: str,
    backend: str,
    *,
    scenario: Scenario | None = None,
    repetitions: int = 5,
    threshold: float = 1.0,
) -> StabilityReport:
    """Run backend against itself N times (rep1 vs rep2, …)."""

    if repetitions < 2:
        repetitions = 2
    # Collect N traces via pairwise self-compares
    agreements = 0
    pairs = 0
    statuses: list[str] = []
    layers: list[str | None] = []
    run_ids: list[str] = []
    notes: list[str] = []

    for i in range(repetitions - 1):
        result = engine.compare(
            model,
            [backend, backend],
            scenario=scenario,
            baseline_backend=backend,
        )
        run_ids.append(result.run_id)
        status = result.diagnosis.status
        value = status.value if hasattr(status, "value") else str(status)
        statuses.append(value)
        layers.append(result.diagnosis.first_divergence)
        pairs += 1
        if value in {ParityResult.PASS.value, ParityResult.PASS_WITH_TOLERANCE.value}:
            agreements += 1
        elif value in {ParityResult.NOT_OBSERVABLE.value, ParityResult.INCOMPARABLE.value}:
            notes.append(f"pair[{i}] inconclusive: {value}")
        else:
            notes.append(f"pair[{i}] divergent/error: {value} @ {result.diagnosis.first_divergence}")

    rate = agreements / pairs if pairs else 0.0
    return StabilityReport(
        backend=backend,
        model=model,
        repetitions=repetitions,
        agreements=agreements,
        rate=round(rate, 4),
        pairwise_status=statuses,
        first_divergence_layers=layers,
        self_consistent=rate + 1e-12 >= threshold,
        run_ids=run_ids,
        notes=notes,
    )


def compare_with_stability(
    engine: CompareEngine,
    model: str,
    backends: list[str],
    *,
    scenario: Scenario | None = None,
    repetitions: int = 3,
    threshold: float = 1.0,
    require_self_consistency: bool = True,
) -> tuple[Any, CrossStabilityReport]:
    """Stabilize each unique backend, then cross-compare if attributable."""

    unique = []
    for b in backends:
        if b not in unique:
            unique.append(b)
    if len(unique) < 2:
        unique = [unique[0], unique[0]] if unique else ["fake", "fake"]

    a, b = unique[0], unique[1]
    report_a = stabilize_backend(engine, model, a, scenario=scenario, repetitions=repetitions, threshold=threshold)
    report_b = stabilize_backend(engine, model, b, scenario=scenario, repetitions=repetitions, threshold=threshold)

    attributable = report_a.self_consistent and report_b.self_consistent
    cross_result = None
    cross_status = None
    cross_rate = None

    if not require_self_consistency or attributable:
        cross_result = engine.compare(model, [a, b], scenario=scenario, baseline_backend=a)
        cross_status = (
            cross_result.diagnosis.status.value
            if hasattr(cross_result.diagnosis.status, "value")
            else str(cross_result.diagnosis.status)
        )
        cross_rate = 1.0 if cross_status in {"PASS", "PASS_WITH_TOLERANCE"} else 0.0
        conclusion = (
            f"Backend A stability: {report_a.rate:.0%} · "
            f"Backend B stability: {report_b.rate:.0%} · "
            f"Cross-backend: {cross_status}."
        )
        if not attributable:
            conclusion += (
                " One or both backends are internally non-reproducible; "
                "cross-backend divergence cannot be attributed conclusively."
            )
            # Force formal inconclusive annotation on diagnosis path (caller may override)
    else:
        conclusion = (
            f"Backend A stability: {report_a.rate:.0%} · "
            f"Backend B stability: {report_b.rate:.0%}. "
            "Cross-backend comparison skipped: require-self-consistency failed. "
            "Backend is internally non-reproducible; divergence cannot be attributed."
        )
        cross_status = FormalParityStatus.INCONCLUSIVE.value

    report = CrossStabilityReport(
        backend_a=report_a,
        backend_b=report_b,
        cross_agreement_rate=cross_rate,
        cross_status=cross_status,
        attributable=attributable,
        conclusion=conclusion,
        cross_run_id=cross_result.run_id if cross_result else None,
    )
    return cross_result, report
