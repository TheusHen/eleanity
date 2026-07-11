from __future__ import annotations

import json
import sys
from typing import Any, Literal

from rich.console import Console

from eleanity.cli.errors import EleanityError
from eleanity.cli.exitcodes import EXIT_CONFIG, exit_from_diagnosis
from eleanity.reporters.terminal import print_terminal
from eleanity.version import __version__

OutputFormat = Literal["text", "json", "quiet"]


def emit_error(error: EleanityError | Exception, *, fmt: OutputFormat = "text") -> int:
    if isinstance(error, EleanityError):
        if fmt == "json":
            payload = {
                "ok": False,
                "code": error.code,
                "message": error.message,
                "hint": error.hint,
                "exit_code": error.exit_code,
            }
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return error.exit_code
    if fmt == "json":
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "ELEANITY_E012",
                    "message": str(error),
                    "exit_code": EXIT_CONFIG,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    else:
        print(f"ELEANITY_E012: {error}", file=sys.stderr)
    return EXIT_CONFIG


def canonical_summary(
    *,
    run_id: str,
    status: str,
    first_divergence: str | None,
    propagation: float | None,
    gates_passed: bool | None,
    model: str | None = None,
    scenario: str | None = None,
    baseline: str | None = None,
    candidate: str | None = None,
    hypothesis: str | None = None,
    next_action: str | None = None,
    total_ms: float | None = None,
    impact: str | None = None,
    formal_status: str | None = None,
) -> dict[str, Any]:
    return {
        "eleanity": __version__,
        "run_id": run_id,
        "status": status,
        "formal_status": formal_status or status,
        "impact": impact,
        "first_divergence": first_divergence,
        "propagation_percent": propagation,
        "gates_passed": gates_passed,
        "model": model,
        "scenario": scenario,
        "baseline": baseline,
        "candidate": candidate,
        "hypothesis": hypothesis,
        "next_action": next_action,
        "total_ms": total_ms,
    }


def print_canonical_text(summary: dict[str, Any], console: Console | None = None) -> None:
    out = console or Console()
    out.print()
    out.print("[bold]── summary ──[/bold]")
    out.print(f"status:            {summary.get('status')}")
    if summary.get("formal_status") and summary.get("formal_status") != summary.get("status"):
        out.print(f"formal_status:     {summary.get('formal_status')}")
    out.print(f"impact:            {summary.get('impact') or 'n/a'}")
    out.print(f"first_divergence:  {summary.get('first_divergence') or 'none'}")
    prop = summary.get("propagation_percent")
    out.print(f"propagation:       {prop if prop is not None else 'n/a'}")
    gates = summary.get("gates_passed")
    out.print(f"gates:             {'PASS' if gates is True else 'FAIL' if gates is False else 'n/a'}")
    out.print(f"run_id:            {summary.get('run_id')}")
    if summary.get("hypothesis"):
        out.print(f"cause:             {summary['hypothesis']}")
    if summary.get("next_action"):
        out.print(f"next:              {summary['next_action']}")
    if summary.get("total_ms") is not None:
        out.print(f"total_ms:          {summary['total_ms']}")


def emit_compare_result(
    *,
    fmt: OutputFormat,
    traces,
    diagnosis,
    run_id: str,
    scenario=None,
    model: str | None = None,
    policy: str | None = None,
    gate_evaluation=None,
    timings: dict | None = None,
    comparisons: dict | None = None,
    quiet_layers: bool = False,
) -> int:
    status = getattr(diagnosis, "status", None)
    status_value = status.value if hasattr(status, "value") else str(status or "UNKNOWN")
    first = getattr(diagnosis, "first_divergence", None)
    prop = getattr(diagnosis, "propagation_percent", None)
    gates_passed = gate_evaluation.passed if gate_evaluation is not None else None
    causes = getattr(diagnosis, "probable_causes", None) or []
    actions = getattr(diagnosis, "suggested_actions", None) or []
    hypothesis = causes[0].message if causes else getattr(diagnosis, "hypothesis", None)
    next_action = actions[0] if actions else getattr(diagnosis, "next_test", None)
    total_ms = sum(timings.values()) if timings else None
    baseline = traces[0].backend if traces else None
    candidate = traces[1].backend if traces and len(traces) > 1 else None
    scenario_name = scenario.name if scenario is not None else (traces[0].scenario_name if traces else None)

    impact_obj = getattr(diagnosis, "impact", None) or {}
    impact_value = impact_obj.get("impact") if isinstance(impact_obj, dict) else None
    formal = getattr(diagnosis, "formal_status", None) or status_value

    summary = canonical_summary(
        run_id=run_id,
        status=status_value,
        first_divergence=first,
        propagation=prop,
        gates_passed=gates_passed,
        model=model,
        scenario=scenario_name,
        baseline=baseline,
        candidate=candidate,
        hypothesis=hypothesis,
        next_action=next_action,
        total_ms=total_ms,
        impact=impact_value,
        formal_status=formal,
    )

    coverage = getattr(diagnosis, "coverage", None) or {}
    conf = getattr(diagnosis, "confidence", None)
    # Prefer reproduction from first trace path is not available here — callers embed in diagnosis cmds

    if fmt == "json":
        from eleanity.core.coverage import format_timings

        payload = {
            "ok": status_value
            in {
                "PASS",
                "PASS_WITH_TOLERANCE",
                "PASS_WITH_LIMITED_COVERAGE",
                "NOT_OBSERVABLE",
                "INCOMPARABLE",
                "INCONCLUSIVE",
                "NOT_REQUESTED",
                "NOT_SUPPORTED",
                "UNSUPPORTED",
            },
            "summary": {
                **summary,
                "confidence": conf,
                "coverage_percent": coverage.get("required_coverage_percent"),
                "verified_layers": getattr(diagnosis, "verified_layers", None),
                "tolerance_reasons": getattr(diagnosis, "tolerance_reasons", None),
            },
            "diagnosis": diagnosis.model_dump(mode="json") if hasattr(diagnosis, "model_dump") else {},
            "impact": impact_obj if impact_obj else None,
            "coverage": coverage,
            "verified_layers": getattr(diagnosis, "verified_layers", None),
            "not_verified_layers": getattr(diagnosis, "not_verified_layers", None),
            "tolerance_reasons": getattr(diagnosis, "tolerance_reasons", None),
            "artifact_divergent_fields": getattr(diagnosis, "artifact_divergent_fields", None),
            "confidence": conf,
            "practical_commands": getattr(diagnosis, "practical_commands", None),
            "gates": (
                {
                    "passed": gate_evaluation.passed,
                    "summary": gate_evaluation.summary,
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "message": r.message,
                            "status": r.status.value if r.status else None,
                        }
                        for r in gate_evaluation.results
                    ],
                }
                if gate_evaluation
                else None
            ),
            "timings_ms": timings,
            "timings": format_timings(timings),
            "comparisons": comparisons,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif fmt == "quiet":
        cov = coverage.get("required_coverage_percent")
        print(
            f"status={status_value} impact={impact_value or 'n/a'} "
            f"coverage={cov if cov is not None else 'n/a'} "
            f"confidence={conf if conf is not None else 'n/a'} "
            f"first_divergence={first or 'none'} gates={gates_passed} run_id={run_id}"
        )
    else:
        # Single unified report — no duplicate summary block
        print_terminal(
            traces,
            diagnosis,
            scenario=scenario,
            model=model,
            policy=policy,
            comparisons=comparisons,
            timings=timings,
            gate_evaluation=gate_evaluation,
            run_id=run_id,
        )

    return exit_from_diagnosis(diagnosis, gate_passed=gates_passed)
