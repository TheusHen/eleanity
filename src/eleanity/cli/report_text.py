from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eleanity.comparators.diff import compare_prompt, compare_tokens
from eleanity.core.coverage import format_timings
from eleanity.models.schemas import ParityResult
from eleanity.playbook import get_playbook_entry
from eleanity.version import __version__


def render_text_report(data: dict[str, Any], *, redact: bool = True) -> str:
    """Full plain-text causal report aligned with the live compare terminal output."""

    console = Console(record=True, width=100, force_terminal=False)
    diagnosis = data.get("diagnosis") or {}
    scenario = data.get("scenario") or {}
    traces = data.get("traces") or []
    comparisons = data.get("comparisons") or {}
    gates = data.get("gates") or {}
    timings = data.get("timings_ms") or {}
    coverage = data.get("coverage") or diagnosis.get("coverage") or {}

    console.print(Text(f"Eleanity {__version__} · text report", style="bold"))
    console.print(f"run_id:     {data.get('run_id')}")
    console.print(f"run_type:   {data.get('run_type')}")
    console.print(f"scenario:   {scenario.get('name') or (traces[0].get('scenario_name') if traces else '—')}")
    console.print(f"policy:     {scenario.get('parity_profile') or scenario.get('parity_policy') or '—'}")
    console.print(f"baseline:   {data.get('baseline_backend') or '—'}")
    if data.get("baseline_model"):
        console.print(f"models:     {data.get('baseline_model')} → {data.get('candidate_model')}")
    console.print()

    if traces:
        layer_table = Table(title="Layers (observation ≠ comparison)", show_header=True, box=None)
        layer_table.add_column("Layer")
        layer_table.add_column("Baseline obs")
        layer_table.add_column("Candidate obs")
        layer_table.add_column("Compare")
        layer_table.add_column("Origin")
        base_layers = traces[0].get("layers") or {}
        cand = traces[1] if len(traces) > 1 else {}
        cand_layers = (cand.get("layers") or {}) if cand else {}
        # first comparison backend
        cmp_layers: dict[str, Any] = {}
        if comparisons:
            cmp_layers = next(iter(comparisons.values())) or {}
        for layer in list(base_layers.keys()) or list(cand_layers.keys()):
            bl = base_layers.get(layer) or {}
            cl = cand_layers.get(layer) or {}
            entry = cmp_layers.get(layer) or {}
            result = entry.get("result") if isinstance(entry, dict) else "—"
            origin = f"{bl.get('origin') or '—'} → {cl.get('origin') or '—'}"
            layer_table.add_row(
                layer,
                str(bl.get("state") or "—"),
                str(cl.get("state") or "—"),
                str(result or "—"),
                origin[:40],
            )
        console.print(layer_table)
        console.print()

    if coverage:
        console.print(Text("Coverage", style="bold"))
        console.print(
            f"  required:  {coverage.get('required_coverage_percent')}% (min {coverage.get('min_coverage_percent')}%)"
        )
        console.print(f"  requested: {coverage.get('requested_coverage_percent')}%")
        console.print()

    verified = data.get("verified_layers") or diagnosis.get("verified_layers") or []
    not_verified = data.get("not_verified_layers") or diagnosis.get("not_verified_layers") or []
    console.print(Text("Verified", style="bold"))
    console.print(f"  {', '.join(verified) if verified else '—'}")
    console.print(Text("Not verified", style="bold"))
    if not_verified:
        for item in not_verified:
            if isinstance(item, dict):
                console.print(f"  - {item.get('layer')}: {item.get('reason')}")
            else:
                console.print(f"  - {item}")
    else:
        console.print("  —")
    console.print()

    art_fields = data.get("artifact_divergent_fields") or diagnosis.get("artifact_divergent_fields") or []
    if art_fields:
        console.print(Text("Artifact divergent fields", style="bold"))
        console.print(f"  {', '.join(art_fields)}")
        console.print()

    tol = data.get("tolerance_reasons") or diagnosis.get("tolerance_reasons") or []
    if tol:
        console.print(Text("Why PASS_WITH_TOLERANCE / limited coverage", style="bold"))
        for reason in tol:
            console.print(f"  - {reason}")
        console.print()

    status = diagnosis.get("status") or "—"
    conf = data.get("confidence") if data.get("confidence") is not None else diagnosis.get("confidence")
    title = f"Diagnosis · {status}"
    if conf is not None:
        title += f" · confidence={conf:.0%}" if isinstance(conf, float) else f" · confidence={conf}"
    console.print(Panel(diagnosis.get("summary") or "—", title=title))
    console.print()

    first = diagnosis.get("first_divergence")
    detail = diagnosis.get("first_divergence_detail") or {}
    location = detail.get("location") or {}
    if first:
        console.print(Text("First divergence", style="bold red"))
        console.print(f"  layer:     {first}")
        if location.get("character") is not None:
            console.print(f"  character: {location.get('character')}")
        if location.get("token_index") is not None:
            console.print(f"  token:     {location.get('token_index')}")
        if detail.get("baseline") is not None:
            console.print(f'  baseline:  "{detail.get("baseline")}"')
        if detail.get("candidate") is not None:
            console.print(f'  candidate: "{detail.get("candidate")}"')
        console.print()

    if len(traces) >= 2:
        _print_layer_diffs(console, traces[0], traces[1], redact=redact)

    causes = diagnosis.get("probable_causes") or []
    if causes:
        console.print(Text("Probable causes", style="bold"))
        for cause in causes:
            console.print(f"  [{cause.get('code')}] conf={cause.get('confidence')} — {cause.get('message')}")
            entry = get_playbook_entry(str(cause.get("code") or ""))
            if entry and entry.get("actions"):
                for action in entry["actions"][:2]:
                    console.print(f"      → {action}")
        console.print()

    cmds = data.get("practical_commands") or diagnosis.get("practical_commands") or []
    actions = diagnosis.get("suggested_actions") or []
    console.print(Text("Suggested next steps", style="bold"))
    for item in list(dict.fromkeys([*cmds, *actions]))[:8]:
        console.print(f"  $ {item}" if str(item).startswith("eleanity ") else f"  - {item}")
    console.print()

    if gates:
        console.print(Text("Gates", style="bold"))
        console.print(f"  passed: {gates.get('passed')} — {gates.get('summary')}")
        for item in gates.get("results") or []:
            mark = "OK" if item.get("passed") else "FAIL"
            console.print(f"  [{mark}] {item.get('name')}: {item.get('message')}")
        console.print()

    tinfo = data.get("timings") or format_timings(timings)
    if tinfo.get("entries"):
        console.print(Text("Timings", style="bold"))
        for entry in tinfo["entries"]:
            console.print(f"  {entry['name']}: {entry['ms']:.1f} ms ({entry['share_percent']}%)")
        console.print(f"  total: {tinfo.get('total_ms')} ms")
        if tinfo.get("delta_label"):
            console.print(f"  delta: {tinfo['delta_label']}")
        console.print()

    if data.get("reproduction_command"):
        console.print(Text("Reproduce", style="bold"))
        console.print(f"  {data['reproduction_command']}")
        console.print()

    for w in diagnosis.get("warnings") or []:
        console.print(f"[yellow]warning:[/yellow] {w}")

    return console.export_text()


def _print_layer_diffs(console: Console, left: dict, right: dict, *, redact: bool) -> None:
    left_layers = left.get("layers") or {}
    right_layers = right.get("layers") or {}

    lt = left_layers.get("template") or {}
    rt = right_layers.get("template") or {}
    if lt.get("state") == "OBSERVED" and rt.get("state") == "OBSERVED":
        ltext = str((lt.get("data") or {}).get("text") or (lt.get("data") or {}).get("rendered_text") or "")
        rtext = str((rt.get("data") or {}).get("text") or (rt.get("data") or {}).get("rendered_text") or "")
        if ltext or rtext:
            cmp = compare_prompt(ltext, rtext)
            console.print(Text("Template diff", style="bold"))
            console.print(f"  result: {cmp.result.value}")
            if cmp.result == ParityResult.DIVERGENT:
                d = cmp.details
                console.print(f"  first_byte: {d.get('first_byte')}  first_char: {d.get('first_character')}")
                if not redact:
                    left_ctx = (d.get("left_context") or {}).get("snippet") or ""
                    right_ctx = (d.get("right_context") or {}).get("snippet") or ""
                    console.print(f'  baseline≈   "{left_ctx}"')
                    console.print(f'  candidate≈  "{right_ctx}"')
                else:
                    console.print("  snippets:   redacted (use --no-redact for local debug)")
            console.print()

    ltok = left_layers.get("tokens") or {}
    rtok = right_layers.get("tokens") or {}
    if ltok.get("state") == "OBSERVED" and rtok.get("state") == "OBSERVED":
        lids = (ltok.get("data") or {}).get("ids") or (ltok.get("data") or {}).get("token_ids") or []
        rids = (rtok.get("data") or {}).get("ids") or (rtok.get("data") or {}).get("token_ids") or []
        if lids or rids:
            cmp = compare_tokens(lids, rids)
            console.print(Text("Token diff", style="bold"))
            console.print(f"  result: {cmp.result.value}")
            if cmp.result == ParityResult.DIVERGENT:
                d = cmp.details
                console.print(f"  first_index:     {d.get('first_difference')}")
                console.print(f"  lengths:         {d.get('left_length')} vs {d.get('right_length')}")
            else:
                console.print(f"  count: {len(lids)} (match)")
            console.print()
