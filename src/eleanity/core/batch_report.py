from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from eleanity.core.engine import CompareEngine, CompareResult
from eleanity.models.schemas import Scenario


@dataclass
class BatchJob:
    model: str
    backends: list[str]
    scenario_name: str


@dataclass
class BatchReport:
    batch_id: str
    path: Path
    results: list[CompareResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def run_multi_model_batch(
    jobs: list[tuple[str, list[str], Scenario]],
    *,
    engine: CompareEngine | None = None,
    output_dir: Path | str = ".eleanity/batches",
    baseline_backend: str | None = None,
    tokenizer_only: bool = False,
    redact_prompts: bool = False,
) -> BatchReport:
    """Run many (model × backends × scenario) compares and write an aggregate report."""

    engine = engine or CompareEngine(parallel=True)
    batch_id = str(uuid4())
    root = Path(output_dir) / batch_id
    root.mkdir(parents=True, exist_ok=True)

    results: list[CompareResult] = []
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    divergence_counts: Counter[str] = Counter()

    for model, backends, scenario in jobs:
        result = engine.compare(
            model,
            backends,
            scenario=scenario,
            baseline_backend=baseline_backend if baseline_backend in backends else backends[0],
            tokenizer_only=tokenizer_only,
            redact_prompts=redact_prompts,
        )
        results.append(result)
        status = str(getattr(result.diagnosis, "status", None) or "UNKNOWN")
        if hasattr(result.diagnosis.status, "value"):
            status = result.diagnosis.status.value
        first = getattr(result.diagnosis, "first_divergence", None)
        status_counts[status] += 1
        if first:
            divergence_counts[str(first)] += 1
        gates_passed = result.gate_evaluation.passed if result.gate_evaluation else None
        rows.append(
            {
                "run_id": result.run_id,
                "model": model,
                "scenario": scenario.name,
                "backends": backends,
                "status": status,
                "first_divergence": first,
                "gates_passed": gates_passed,
                "total_duration_ms": sum(result.timings.values()) if result.timings else None,
            }
        )

    summary = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_count": len(jobs),
        "status_counts": dict(status_counts),
        "divergence_layer_counts": dict(divergence_counts),
        "failed": status_counts.get("DIVERGENT", 0) + status_counts.get("ERROR", 0),
        "passed": status_counts.get("PASS", 0) + status_counts.get("PASS_WITH_TOLERANCE", 0),
        "rows": rows,
    }
    (root / "batch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "batch.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return BatchReport(batch_id=batch_id, path=root, results=results, summary=summary)


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Eleanity batch {summary['batch_id']}",
        "",
        f"- jobs: {summary['job_count']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- status_counts: `{summary['status_counts']}`",
        f"- divergence_layers: `{summary['divergence_layer_counts']}`",
        "",
        "| model | scenario | status | first_divergence | run_id |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('scenario')} | {row.get('status')} | "
            f"{row.get('first_divergence') or '—'} | `{str(row.get('run_id'))[:8]}` |"
        )
    lines.append("")
    return "\n".join(lines)
