from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eleanity.models.schemas import Comparison, LayerObservation, LayerState, ObservationTrace, ParityResult
from eleanity.policies.engine import PolicyEngine
from eleanity.models.schemas import Scenario


def save_golden(trace: ObservationTrace, directory: Path | str, name: str | None = None) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    filename = name or f"{trace.scenario_name}__{trace.backend}.json"
    path = root / filename
    path.write_text(json.dumps(trace.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_golden(path: Path | str) -> ObservationTrace:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ObservationTrace.model_validate(data)


def compare_against_golden(
    live: ObservationTrace,
    golden: ObservationTrace,
    scenario: Scenario,
) -> dict[str, Comparison]:
    return PolicyEngine(scenario).compare_layers(golden, live)


def golden_gate(
    live: ObservationTrace,
    golden_path: Path | str,
    scenario: Scenario,
    *,
    layers: list[str] | None = None,
) -> dict[str, Any]:
    golden = load_golden(golden_path)
    comparisons = compare_against_golden(live, golden, scenario)
    focus = layers or list(comparisons)
    divergent = [
        layer
        for layer in focus
        if layer in comparisons and comparisons[layer].result == ParityResult.DIVERGENT
    ]
    return {
        "passed": not divergent,
        "divergent_layers": divergent,
        "comparisons": {k: v.model_dump(mode="json") for k, v in comparisons.items()},
        "golden_backend": golden.backend,
        "live_backend": live.backend,
    }
