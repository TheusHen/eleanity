from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eleanity.core.coverage import format_timings
from eleanity.models.schemas import LAYER_ORDER
from eleanity.version import __version__


def _status_style(status: str) -> str:
    return {
        "PASS": "green",
        "PASS_WITH_TOLERANCE": "yellow",
        "PASS_WITH_LIMITED_COVERAGE": "yellow",
        "DIVERGENT": "red bold",
        "INCONCLUSIVE": "magenta",
        "NOT_OBSERVABLE": "dim",
        "NOT_REQUESTED": "dim",
        "NOT_SUPPORTED": "dim",
        "INCOMPARABLE": "dim",
        "ERROR": "red",
        "REFERENCE": "cyan",
        "OBSERVED": "green",
    }.get(status, "white")


def print_terminal(
    traces,
    diagnosis,
    *,
    scenario=None,
    model: str | None = None,
    policy: str | None = None,
    comparisons: dict | None = None,
    timings: dict | None = None,
    gate_evaluation=None,
    reproduction: str | None = None,
    run_id: str | None = None,
) -> None:
    """Single engineering-style terminal report (no duplicate summary blocks)."""

    console = Console()
    console.print(Text(f"Eleanity {__version__}", style="bold"))
    console.print()

    baseline = traces[0] if traces else None
    candidate = traces[1] if len(traces) > 1 else None
    scenario_name = scenario.name if scenario is not None else (baseline.scenario_name if baseline else "—")
    model_ref = model or (baseline.artifact_fingerprint.model_ref if baseline else "—")
    policy_name = policy or (
        getattr(scenario, "parity_profile", None).value
        if scenario is not None and getattr(scenario, "parity_profile", None)
        else "strict"
    )

    console.print(f"Scenario: {scenario_name}")
    console.print(f"Model:    {model_ref}")
    console.print(f"Policy:   {policy_name}")
    if run_id:
        console.print(f"Run:      {run_id}")
    console.print()
    if baseline and candidate:
        console.print(f"Baseline:  {baseline.backend}")
        console.print(f"Candidate: {candidate.backend}")
        console.print()
    elif baseline:
        console.print(f"Backend: {baseline.backend}")
        console.print()

    # Layer table: observation states + comparison result (separated)
    layer_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    layer_table.add_column("Layer", style="bold")
    layer_table.add_column("Baseline obs")
    layer_table.add_column("Candidate obs")
    layer_table.add_column("Compare")
    layer_table.add_column("Origin (base→cand)")

    # Resolve comparisons for first candidate
    cmp_map: dict[str, Any] = {}
    if comparisons:
        first_key = next(iter(comparisons))
        raw = comparisons[first_key]
        # raw values may be dict dumps
        cmp_map = raw
    elif baseline and candidate:
        from eleanity.policies.engine import PolicyEngine

        eng = PolicyEngine(
            scenario
            if scenario is not None
            else __import__("eleanity.models.schemas", fromlist=["Scenario"]).Scenario(
                name="x", messages=[{"role": "user", "content": "x"}]
            )
        )
        cmp_objs = eng.compare_layers(baseline, candidate)
        cmp_map = {k: v.model_dump(mode="json") for k, v in cmp_objs.items()}

    seen_layers: list[str] = []
    if baseline and candidate:
        for layer in LAYER_ORDER:
            if layer in baseline.layers or layer in candidate.layers or layer in cmp_map:
                seen_layers.append(layer)
        for layer in list(baseline.layers) + list(candidate.layers) + list(cmp_map):
            if layer not in seen_layers:
                seen_layers.append(layer)
        for layer in seen_layers:
            left = baseline.layers.get(layer)
            right = candidate.layers.get(layer)
            entry = cmp_map.get(layer) or {}
            if hasattr(entry, "result"):
                result = entry.result.value
            else:
                result = str(entry.get("result") or "—")
                entry.get("details") or {}
            lo = left.state.value if left else "—"
            ro = right.state.value if right else "—"
            origin = f"{(left.origin if left else None) or '—'} → {(right.origin if right else None) or '—'}"
            layer_table.add_row(
                layer,
                Text(lo, style=_status_style(lo)),
                Text(ro, style=_status_style(ro)),
                Text(result, style=_status_style(result)),
                Text(origin[:48], style="dim"),
            )
    console.print(layer_table)
    console.print()

    # Coverage
    coverage = getattr(diagnosis, "coverage", None) or {}
    if coverage:
        console.print(Text("Coverage", style="bold"))
        console.print(
            f"  required layers:  {coverage.get('required_coverage_percent')}% "
            f"({coverage.get('required_verified')}/{coverage.get('required_total')}) "
            f"min={coverage.get('min_coverage_percent')}%"
        )
        console.print(f"  requested layers: {coverage.get('requested_coverage_percent')}%")
        console.print()

    # Verified / Not verified
    verified = getattr(diagnosis, "verified_layers", None) or coverage.get("verified_layers") or []
    not_verified = getattr(diagnosis, "not_verified_layers", None) or coverage.get("not_verified_layers") or []
    console.print(Text("Verified", style="bold green"))
    console.print(f"  {', '.join(verified) if verified else '—'}")
    console.print()
    console.print(Text("Not verified", style="bold yellow"))
    if not_verified:
        for item in not_verified:
            if isinstance(item, dict):
                console.print(
                    f"  - {item.get('layer')}: {item.get('reason')} "
                    f"(base={item.get('baseline_state')}, cand={item.get('candidate_state')})"
                )
            else:
                console.print(f"  - {item}")
    else:
        console.print("  —")
    console.print()

    # Artifact field diffs
    art_fields = getattr(diagnosis, "artifact_divergent_fields", None) or []
    if art_fields:
        console.print(Text("Artifact divergent fields", style="bold"))
        console.print(f"  {', '.join(art_fields)}")
        console.print()

    # Tolerance reasons
    tol = getattr(diagnosis, "tolerance_reasons", None) or []
    if tol or (
        getattr(diagnosis, "status", None) and getattr(diagnosis.status, "value", None) == "PASS_WITH_TOLERANCE"
    ):
        console.print(Text("Why PASS_WITH_TOLERANCE / limited coverage", style="bold"))
        if tol:
            for reason in tol:
                console.print(f"  - {reason}")
        else:
            console.print("  - numeric/prefix thresholds of the active policy")
        console.print()

    # First divergence
    if diagnosis.first_divergence:
        detail = getattr(diagnosis, "first_divergence_detail", None)
        console.print(Text("First divergence", style="bold red"))
        console.print(f"  Layer: {diagnosis.first_divergence}")
        if detail and detail.location:
            if detail.location.character is not None:
                console.print(f"  Character: {detail.location.character}")
            if detail.location.byte is not None:
                console.print(f"  Byte: {detail.location.byte}")
            if detail.location.token_index is not None:
                console.print(f"  Token index: {detail.location.token_index}")
            if detail.location.line is not None:
                console.print(f"  Line/col: {detail.location.line}:{detail.location.column}")
        if detail and detail.baseline is not None:
            console.print(f'  Baseline:  "{detail.baseline}"')
        if detail and detail.candidate is not None:
            console.print(f'  Candidate: "{detail.candidate}"')
        console.print()

    causes = getattr(diagnosis, "probable_causes", None) or []
    if causes:
        console.print(Text("Probable cause", style="bold"))
        top = causes[0]
        conf = getattr(top, "confidence", None)
        console.print(f"  [{top.code}] conf={conf} — {top.message}")
        if getattr(top, "evidence", None):
            for k, v in list(top.evidence.items())[:4]:
                console.print(f"    evidence.{k}: {v}")
        for cause in causes[1:3]:
            console.print(f"  - [{cause.code}] conf={cause.confidence} — {cause.message}")
        console.print()

    conf = getattr(diagnosis, "confidence", None)
    if conf is not None:
        console.print(f"[bold]Diagnosis confidence:[/bold] {conf:.0%}")
        console.print()

    cmds = getattr(diagnosis, "practical_commands", None) or []
    actions = getattr(diagnosis, "suggested_actions", None) or []
    console.print(Text("Suggested next steps", style="bold"))
    shown = list(dict.fromkeys([*cmds, *actions]))[:6]
    if shown:
        for item in shown:
            console.print(f"  $ {item}" if item.startswith("eleanity ") else f"  - {item}")
    else:
        console.print("  - Expand observe layers if deeper parity is required.")
    console.print()

    # Timings
    tinfo = format_timings(timings)
    if tinfo["entries"]:
        console.print(Text("Timings", style="bold"))
        for entry in tinfo["entries"]:
            console.print(f"  {entry['name']}: {entry['ms']:.1f} ms ({entry['share_percent']}%)")
        console.print(f"  total: {tinfo['total_ms']:.1f} ms")
        if tinfo.get("delta_label"):
            console.print(f"  delta: {tinfo['delta_label']}")
        console.print()

    if gate_evaluation is not None:
        console.print(f"[bold]Gates:[/bold] {'PASS' if gate_evaluation.passed else 'FAIL'} — {gate_evaluation.summary}")
        console.print()

    impact = getattr(diagnosis, "impact", None) or {}
    if isinstance(impact, dict) and impact.get("impact"):
        console.print(f"[bold]Impact:[/bold] {impact.get('impact')} — {impact.get('rationale') or ''}")
        console.print()

    if reproduction:
        console.print(Text("Reproduce", style="bold"))
        console.print(f"  {reproduction}")
        console.print()

    status = getattr(diagnosis, "status", None)
    status_value = status.value if status is not None else "—"
    console.print(
        Panel(
            diagnosis.summary or "—",
            title=f"Diagnosis · {status_value}",
            border_style=_status_style(status_value),
        )
    )
