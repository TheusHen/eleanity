from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eleanity.config.project import GateRule
from eleanity.core.coverage import POLICY_MIN_COVERAGE, policy_min_coverage
from eleanity.models.schemas import ParityResult


STATUS_RANK = {
    ParityResult.PASS: 0,
    ParityResult.PASS_WITH_TOLERANCE: 1,
    ParityResult.PASS_WITH_LIMITED_COVERAGE: 2,
    ParityResult.NOT_REQUESTED: 3,
    ParityResult.NOT_OBSERVABLE: 3,
    ParityResult.NOT_SUPPORTED: 3,
    ParityResult.INCONCLUSIVE: 3,
    ParityResult.INCOMPARABLE: 3,
    ParityResult.DIVERGENT: 4,
    ParityResult.ERROR: 5,
}


def _as_status(value: Any) -> ParityResult:
    if isinstance(value, ParityResult):
        return value
    try:
        return ParityResult(str(value))
    except ValueError:
        return ParityResult.ERROR


@dataclass
class GateResult:
    name: str
    passed: bool
    layer: str | None = None
    status: ParityResult | None = None
    message: str = ""
    backend: str | None = None


@dataclass
class GateEvaluation:
    passed: bool
    results: list[GateResult] = field(default_factory=list)
    summary: str = ""

    @property
    def exit_code(self) -> int:
        if self.passed:
            return 0
        if any(r.status == ParityResult.ERROR for r in self.results if not r.passed):
            return 2
        return 1


def evaluate_gates(
    gates: list[GateRule],
    comparisons: dict[str, dict[str, Any]],
    *,
    diagnosis_status: ParityResult | str | None = None,
    coverage: dict[str, Any] | None = None,
    policy: str | None = None,
    min_coverage: float | None = None,
) -> GateEvaluation:
    """Evaluate production gates against pairwise comparison maps.

    Also enforces minimum required-layer coverage when ``coverage`` is provided.
    ``comparisons`` shape: {backend_key: {layer: {result: STATUS, details: ...}}}
    """

    results: list[GateResult] = []

    # Coverage gate (always when coverage payload present)
    if coverage is not None:
        threshold = (
            min_coverage
            if min_coverage is not None
            else policy_min_coverage(policy or coverage.get("policy") or "strict")
        )
        cov_pct = float(coverage.get("required_coverage_percent") or 0.0)
        ok_cov = cov_pct + 1e-9 >= threshold * 100.0
        results.append(
            GateResult(
                name="min-coverage",
                passed=ok_cov,
                status=ParityResult.PASS if ok_cov else ParityResult.PASS_WITH_LIMITED_COVERAGE,
                message=(
                    f"required-layer coverage {cov_pct}% "
                    f"(min {threshold * 100:.0f}%)"
                    + ("" if ok_cov else " — below minimum")
                ),
            )
        )

    if not gates:
        status = _as_status(diagnosis_status) if diagnosis_status else ParityResult.PASS
        ok = status in {
            ParityResult.PASS,
            ParityResult.PASS_WITH_TOLERANCE,
            ParityResult.PASS_WITH_LIMITED_COVERAGE,
            ParityResult.NOT_OBSERVABLE,
            ParityResult.INCONCLUSIVE,
            ParityResult.NOT_REQUESTED,
            ParityResult.NOT_SUPPORTED,
        }
        # If coverage gate failed, overall fails
        cov_failed = any(r.name == "min-coverage" and not r.passed for r in results)
        if cov_failed:
            ok = False
        results.append(
            GateResult(
                name="default-diagnosis",
                passed=ok and not cov_failed,
                status=status,
                message="No custom gates configured; used diagnosis status + coverage.",
            )
        )
        passed = all(r.passed for r in results)
        failed = [r for r in results if not r.passed]
        summary = (
            "all gates passed"
            if passed
            else f"{len(failed)} gate failure(s): " + "; ".join(r.message for r in failed[:5])
        )
        return GateEvaluation(passed=passed, results=results, summary=summary)

    for gate in gates:
        allowed = set(gate.allow or [])
        max_rank = STATUS_RANK.get(gate.max_status, 0)
        gate_ok = True
        for backend, layers in comparisons.items():
            for layer in gate.layers:
                entry = layers.get(layer)
                if not entry:
                    status = ParityResult.NOT_OBSERVABLE
                    if status in allowed or not gate.required:
                        continue
                    if STATUS_RANK.get(status, 99) <= max_rank:
                        continue
                    gate_ok = False
                    results.append(
                        GateResult(
                            name=gate.name,
                            passed=False,
                            layer=layer,
                            backend=backend,
                            status=status,
                            message=f"layer {layer} missing for {backend}",
                        )
                    )
                    continue
                status = _as_status(entry.get("result", "NOT_OBSERVABLE"))
                if status in allowed:
                    continue
                # Limited coverage / not requested do not fail soft gates by default
                if status in {
                    ParityResult.NOT_REQUESTED,
                    ParityResult.PASS_WITH_LIMITED_COVERAGE,
                } and not gate.required:
                    continue
                if STATUS_RANK.get(status, 99) > max_rank:
                    gate_ok = False
                    results.append(
                        GateResult(
                            name=gate.name,
                            passed=False,
                            layer=layer,
                            backend=backend,
                            status=status,
                            message=(
                                f"{backend}/{layer}={status.value} exceeds max_status="
                                f"{gate.max_status.value}"
                            ),
                        )
                    )
        if gate_ok:
            results.append(
                GateResult(
                    name=gate.name,
                    passed=True,
                    message=f"gate {gate.name} passed",
                )
            )

    passed = all(r.passed for r in results)
    failed = [r for r in results if not r.passed]
    summary = (
        "all gates passed"
        if passed
        else f"{len(failed)} gate failure(s): " + "; ".join(r.message for r in failed[:5])
    )
    return GateEvaluation(passed=passed, results=results, summary=summary)
