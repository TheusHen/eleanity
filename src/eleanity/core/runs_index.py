from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    run_id: str
    path: Path
    run_type: str
    scenario: str
    model: str
    baseline: str
    status: str
    first_divergence: str | None
    created_at: str
    duration_ms: float | None
    backends: list[str]


def _load_result(path: Path) -> dict[str, Any] | None:
    result = path / "result.json"
    if not result.is_file():
        return None
    try:
        return json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_runs(runs_dir: Path | str = ".eleanity/runs") -> list[RunSummary]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    summaries: list[RunSummary] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        data = _load_result(child)
        if not data:
            continue
        traces = data.get("traces") or []
        diagnosis = data.get("diagnosis") or {}
        scenario = data.get("scenario") or {}
        model = (
            data.get("baseline_model")
            or (scenario.get("model") or {}).get("id")
            or (traces[0].get("artifact_fingerprint") or {}).get("model_ref")
            if traces
            else "—"
        )
        durations = [t.get("duration_ms") for t in traces if t.get("duration_ms") is not None]
        created = traces[0].get("created_at") if traces else ""
        summaries.append(
            RunSummary(
                run_id=str(data.get("run_id") or child.name),
                path=child,
                run_type=str(data.get("run_type") or "compare"),
                scenario=str(scenario.get("name") or (traces[0].get("scenario_name") if traces else "—")),
                model=str(model or "—"),
                baseline=str(data.get("baseline_backend") or "—"),
                status=str(diagnosis.get("status") or "—"),
                first_divergence=diagnosis.get("first_divergence"),
                created_at=str(created or ""),
                duration_ms=sum(durations) if durations else None,
                backends=[str(t.get("backend")) for t in traces],
            )
        )
    return summaries


def load_run(run_id: str, runs_dir: Path | str = ".eleanity/runs") -> dict[str, Any]:
    root = Path(runs_dir)
    path = root / run_id / "result.json"
    if not path.is_file():
        # allow prefix match
        matches = [p for p in root.glob("*") if p.is_dir() and p.name.startswith(run_id)]
        if len(matches) == 1:
            path = matches[0] / "result.json"
        else:
            raise FileNotFoundError(f"run not found: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def diff_runs(left_id: str, right_id: str, runs_dir: Path | str = ".eleanity/runs") -> dict[str, Any]:
    left = load_run(left_id, runs_dir)
    right = load_run(right_id, runs_dir)
    ld = left.get("diagnosis") or {}
    rd = right.get("diagnosis") or {}
    left_layers = _layer_status_map(left)
    right_layers = _layer_status_map(right)
    all_layers = sorted(set(left_layers) | set(right_layers))
    layer_delta = []
    for layer in all_layers:
        a = left_layers.get(layer)
        b = right_layers.get(layer)
        if a != b:
            layer_delta.append({"layer": layer, "left": a, "right": b})
    return {
        "left_run_id": left.get("run_id"),
        "right_run_id": right.get("run_id"),
        "left_status": ld.get("status"),
        "right_status": rd.get("status"),
        "left_first_divergence": ld.get("first_divergence"),
        "right_first_divergence": rd.get("first_divergence"),
        "status_changed": ld.get("status") != rd.get("status"),
        "first_divergence_changed": ld.get("first_divergence") != rd.get("first_divergence"),
        "layer_delta": layer_delta,
        "left_created_at": (left.get("traces") or [{}])[0].get("created_at"),
        "right_created_at": (right.get("traces") or [{}])[0].get("created_at"),
        "generated_at": datetime.now().isoformat(),
    }


def _layer_status_map(data: dict[str, Any]) -> dict[str, str]:
    """Aggregate worst status per layer across comparison candidates."""

    from eleanity.gates.engine import STATUS_RANK
    from eleanity.models.schemas import ParityResult

    comparisons = data.get("comparisons") or {}
    worst: dict[str, str] = {}
    for _backend, layers in comparisons.items():
        for layer, entry in (layers or {}).items():
            status = str((entry or {}).get("result") or "NOT_OBSERVABLE")
            try:
                rank = STATUS_RANK[ParityResult(status)]
            except ValueError:
                rank = 2
            prev = worst.get(layer)
            if prev is None:
                worst[layer] = status
            else:
                try:
                    if rank > STATUS_RANK[ParityResult(prev)]:
                        worst[layer] = status
                except ValueError:
                    worst[layer] = status
    return worst
