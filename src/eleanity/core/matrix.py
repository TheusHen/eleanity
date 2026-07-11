"""Pairwise comparison matrices and multi-backend consensus."""

from __future__ import annotations

from collections import Counter
from typing import Any

from eleanity.models.schemas import ObservationTrace, ParityResult, Scenario
from eleanity.policies.engine import PolicyEngine


def build_pairwise_matrix(
    traces: list[ObservationTrace],
    scenario: Scenario,
    baseline_index: int = 0,
) -> dict[str, dict[str, Any]]:
    """Compare every non-baseline trace against the baseline; key by unique backend label."""

    if not traces:
        return {}
    engine = PolicyEngine(scenario)
    baseline = traces[baseline_index]
    matrix: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    for index, trace in enumerate(traces):
        if index == baseline_index:
            continue
        seen[trace.backend] += 1
        count = seen[trace.backend]
        key = trace.backend if count == 1 else f"{trace.backend}#{count}"
        comparisons = engine.compare_layers(baseline, trace)
        matrix[key] = {layer: comparison.model_dump(mode="json") for layer, comparison in comparisons.items()}
    return matrix


def consensus_summary(
    traces: list[ObservationTrace],
    scenario: Scenario,
) -> dict[str, Any]:
    """Multi-backend consensus: which layers agree across all comparable pairs."""

    if len(traces) < 2:
        return {"pairs": 0, "layers": {}, "status": "INCOMPARABLE"}

    engine = PolicyEngine(scenario)
    layer_votes: dict[str, list[str]] = {}
    pairs = 0
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            pairs += 1
            comparisons = engine.compare_layers(traces[i], traces[j])
            for layer, comparison in comparisons.items():
                layer_votes.setdefault(layer, []).append(comparison.result.value)

    layer_status: dict[str, dict[str, Any]] = {}
    for layer, votes in layer_votes.items():
        counts = Counter(votes)
        priority = [
            ParityResult.ERROR.value,
            ParityResult.DIVERGENT.value,
            ParityResult.INCOMPARABLE.value,
            ParityResult.NOT_OBSERVABLE.value,
            ParityResult.PASS_WITH_TOLERANCE.value,
            ParityResult.PASS.value,
        ]
        worst = next((p for p in priority if counts.get(p)), ParityResult.NOT_OBSERVABLE.value)
        layer_status[layer] = {
            "status": worst,
            "votes": dict(counts),
            "agree_pass": counts.get(ParityResult.PASS.value, 0)
            + counts.get(ParityResult.PASS_WITH_TOLERANCE.value, 0),
            "pairs": len(votes),
        }

    overall = ParityResult.PASS.value
    for meta in layer_status.values():
        if meta["status"] == ParityResult.ERROR.value:
            overall = ParityResult.ERROR.value
            break
        if meta["status"] == ParityResult.DIVERGENT.value:
            overall = ParityResult.DIVERGENT.value
    if overall == ParityResult.PASS.value and not layer_status:
        overall = ParityResult.NOT_OBSERVABLE.value

    return {"pairs": pairs, "layers": layer_status, "status": overall}
