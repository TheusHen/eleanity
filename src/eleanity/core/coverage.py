"""Layer coverage, required layers per policy, and limited-coverage status."""

from __future__ import annotations

from typing import Any

from eleanity.models.schemas import (
    Comparison,
    LayerObservation,
    LayerState,
    ObservationTrace,
    ParityProfile,
    ParityResult,
    Scenario,
)

# Layers that must be mutually OBSERVED for a full-confidence verdict under each policy.
POLICY_REQUIRED_LAYERS: dict[str, list[str]] = {
    ParityProfile.STRICT.value: ["artifact", "template", "tokens", "generation"],
    ParityProfile.QUANTIZED.value: ["artifact", "template", "tokens", "generation"],
    ParityProfile.FUNCTIONAL.value: ["generation"],
    ParityProfile.API_CONFORMANCE.value: ["api", "generation"],
}

# Minimum fraction of required layers that must be verified on both sides.
POLICY_MIN_COVERAGE: dict[str, float] = {
    ParityProfile.STRICT.value: 0.75,
    ParityProfile.QUANTIZED.value: 0.50,
    ParityProfile.FUNCTIONAL.value: 0.50,
    ParityProfile.API_CONFORMANCE.value: 0.50,
}

_OBSERVED_OK = {
    LayerState.OBSERVED,
    LayerState.INFERRED,  # weak — counts for coverage with penalty in confidence only
}


def policy_required_layers(policy: str | ParityProfile) -> list[str]:
    name = policy.value if isinstance(policy, ParityProfile) else str(policy)
    return list(POLICY_REQUIRED_LAYERS.get(name, POLICY_REQUIRED_LAYERS[ParityProfile.STRICT.value]))


def policy_min_coverage(policy: str | ParityProfile) -> float:
    name = policy.value if isinstance(policy, ParityProfile) else str(policy)
    return float(POLICY_MIN_COVERAGE.get(name, 0.5))


def _layer_state(trace: ObservationTrace | None, layer: str) -> LayerState | None:
    if trace is None:
        return None
    obs = trace.layers.get(layer)
    return obs.state if obs else None


def classify_unobserved(
    *,
    requested: bool,
    left: LayerObservation | None,
    right: LayerObservation | None,
) -> ParityResult:
    """Map observation gap into a comparison-side availability result (not PASS)."""

    if not requested:
        return ParityResult.NOT_REQUESTED
    states: list[LayerState | None] = []
    for obs in (left, right):
        if obs is None:
            states.append(None)
            continue
        states.append(obs.state)
    for st in states:
        if st in {LayerState.NOT_SUPPORTED, LayerState.UNSUPPORTED}:
            return ParityResult.NOT_SUPPORTED
        if st in {LayerState.FAILED, LayerState.ERROR}:
            return ParityResult.ERROR
    for st in states:
        if st in {LayerState.NOT_EXPOSED, LayerState.NOT_OBSERVABLE, LayerState.REDACTED, None}:
            return ParityResult.NOT_OBSERVABLE
    return ParityResult.INCONCLUSIVE


def compute_coverage(
    left: ObservationTrace,
    right: ObservationTrace,
    *,
    scenario: Scenario | None = None,
    policy: str | ParityProfile | None = None,
    comparisons: dict[str, Comparison] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute layer coverage and verified / not-verified sets."""

    policy_name = (
        policy.value
        if isinstance(policy, ParityProfile)
        else (
            str(policy)
            if policy
            else (scenario.parity_profile.value if scenario is not None else ParityProfile.STRICT.value)
        )
    )
    requested = list(scenario.observe) if scenario is not None else list(set(left.layers) | set(right.layers))
    required = policy_required_layers(policy_name)
    # Union for reporting
    all_layers: list[str] = []
    for name in required + requested + list(left.layers) + list(right.layers):
        if name not in all_layers:
            all_layers.append(name)

    verified: list[str] = []
    not_verified: list[dict[str, Any]] = []
    required_verified = 0
    for layer in all_layers:
        req = layer in requested or layer in required
        ls = _layer_state(left, layer)
        rs = _layer_state(right, layer)
        both_observed = ls in _OBSERVED_OK and rs in _OBSERVED_OK
        cmp_entry = (comparisons or {}).get(layer)
        cmp_result = None
        if cmp_entry is not None:
            cmp_result = (
                cmp_entry.result.value
                if hasattr(cmp_entry, "result")
                else (cmp_entry.get("result") if isinstance(cmp_entry, dict) else None)
            )
        if both_observed and cmp_result in {
            None,
            ParityResult.PASS.value,
            ParityResult.PASS_WITH_TOLERANCE.value,
            ParityResult.DIVERGENT.value,
            ParityResult.PASS_WITH_LIMITED_COVERAGE.value,
        }:
            # comparable on both sides
            if both_observed:
                verified.append(layer)
                if layer in required:
                    required_verified += 1
                continue
        # not verified
        reason = "not_observed_on_both"
        if not req and layer not in required:
            reason = "not_requested"
        elif ls in {LayerState.NOT_SUPPORTED, LayerState.UNSUPPORTED} or rs in {
            LayerState.NOT_SUPPORTED,
            LayerState.UNSUPPORTED,
        }:
            reason = "not_supported"
        elif ls in {LayerState.NOT_EXPOSED, LayerState.NOT_OBSERVABLE} or rs in {
            LayerState.NOT_EXPOSED,
            LayerState.NOT_OBSERVABLE,
        }:
            reason = "not_exposed"
        elif ls in {LayerState.FAILED, LayerState.ERROR} or rs in {
            LayerState.FAILED,
            LayerState.ERROR,
        }:
            reason = "failed"
        not_verified.append(
            {
                "layer": layer,
                "reason": reason,
                "baseline_state": ls.value if ls else None,
                "candidate_state": rs.value if rs else None,
                "required": layer in required,
                "requested": layer in requested,
            }
        )

    required_total = len(required) or 1
    required_coverage = required_verified / required_total
    requested_set = [layer for layer in requested if layer]
    requested_verified = sum(1 for layer in verified if layer in requested_set)
    requested_coverage = (requested_verified / len(requested_set)) if requested_set else 1.0
    min_cov = policy_min_coverage(policy_name)
    meets_min = required_coverage + 1e-12 >= min_cov

    return {
        "policy": policy_name,
        "required_layers": required,
        "requested_layers": requested,
        "verified_layers": verified,
        "not_verified_layers": not_verified,
        "required_verified": required_verified,
        "required_total": len(required),
        "required_coverage_percent": round(required_coverage * 100.0, 1),
        "requested_coverage_percent": round(requested_coverage * 100.0, 1),
        "min_coverage_percent": round(min_cov * 100.0, 1),
        "meets_min_coverage": meets_min,
        "coverage_percent": round(required_coverage * 100.0, 1),
    }


def diagnosis_confidence(
    *,
    status: ParityResult | str,
    coverage: dict[str, Any],
    probable_causes: list[Any] | None = None,
) -> float:
    """Heuristic 0..1 confidence for the diagnosis."""

    value = status.value if isinstance(status, ParityResult) else str(status)
    base = {
        ParityResult.PASS.value: 0.85,
        ParityResult.PASS_WITH_TOLERANCE.value: 0.75,
        ParityResult.PASS_WITH_LIMITED_COVERAGE.value: 0.55,
        ParityResult.DIVERGENT.value: 0.9,
        ParityResult.ERROR.value: 0.95,
        ParityResult.INCONCLUSIVE.value: 0.35,
        ParityResult.NOT_OBSERVABLE.value: 0.3,
        ParityResult.INCOMPARABLE.value: 0.35,
        ParityResult.NOT_SUPPORTED.value: 0.4,
        ParityResult.NOT_REQUESTED.value: 0.2,
    }.get(value, 0.5)
    cov = float(coverage.get("required_coverage_percent") or 0) / 100.0
    conf = base * (0.5 + 0.5 * cov)
    causes = probable_causes or []
    if causes:
        top = causes[0]
        c = getattr(top, "confidence", None)
        if c is None and isinstance(top, dict):
            c = top.get("confidence")
        if isinstance(c, (int, float)):
            conf = 0.5 * conf + 0.5 * float(c)
    return round(min(1.0, max(0.0, conf)), 3)


def apply_coverage_to_status(
    status: ParityResult,
    coverage: dict[str, Any],
) -> tuple[ParityResult, list[str]]:
    """Downgrade PASS → PASS_WITH_LIMITED_COVERAGE / INCONCLUSIVE when coverage is low."""

    reasons: list[str] = []
    if status in {ParityResult.DIVERGENT, ParityResult.ERROR}:
        return status, reasons
    if status in {
        ParityResult.INCONCLUSIVE,
        ParityResult.NOT_OBSERVABLE,
        ParityResult.INCOMPARABLE,
    }:
        return ParityResult.INCONCLUSIVE if status != ParityResult.INCONCLUSIVE else status, reasons

    meets = bool(coverage.get("meets_min_coverage"))
    cov_pct = coverage.get("required_coverage_percent")
    min_pct = coverage.get("min_coverage_percent")
    missing = [item["layer"] for item in (coverage.get("not_verified_layers") or []) if item.get("required")]
    if status in {ParityResult.PASS, ParityResult.PASS_WITH_TOLERANCE, ParityResult.PASS_WITH_LIMITED_COVERAGE}:
        if not meets:
            reasons.append(
                f"Required layer coverage {cov_pct}% is below policy minimum {min_pct}% "
                f"(missing: {', '.join(missing) or 'n/a'})."
            )
            # If almost nothing verified → inconclusive
            if float(cov_pct or 0) < 25:
                return ParityResult.INCONCLUSIVE, reasons
            return ParityResult.PASS_WITH_LIMITED_COVERAGE, reasons
        if missing and status == ParityResult.PASS:
            reasons.append(f"Some required layers were not verified on both backends: {', '.join(missing)}.")
            return ParityResult.PASS_WITH_LIMITED_COVERAGE, reasons
    return status, reasons


def practical_commands_for(
    *,
    status: ParityResult | str,
    first_divergence: str | None,
    model: str | None,
    backends: list[str],
    run_id: str | None = None,
) -> list[str]:
    """Actionable CLI commands for the operator."""

    value = status.value if isinstance(status, ParityResult) else str(status)
    model = model or "MODEL"
    be = ",".join(backends) if backends else "transformers,vllm"
    cmds: list[str] = []
    if run_id:
        cmds.append(f"eleanity report {run_id} --format text")
        cmds.append(f"eleanity replay {run_id}")
    cmds.append(f"eleanity compare --model {model} --backends {be} --policy quantized --format text")
    if first_divergence in {"template", "tokens", "special_tokens"}:
        cmds.append(
            f"eleanity compare --model {model} --backends {be} --tokenizer-only "
            f"--observe artifact,template,special_tokens,tokens --format text"
        )
    if first_divergence == "artifact":
        cmds.append(f"eleanity inspect {model} --backend transformers --format json")
    if value in {ParityResult.PASS_WITH_LIMITED_COVERAGE.value, ParityResult.INCONCLUSIVE.value}:
        cmds.append("eleanity doctor --check-backends --format text")
        cmds.append("eleanity policy-spec --policy quantized  # required layers + comparator thresholds")
    if value == ParityResult.DIVERGENT.value:
        cmds.append(
            f"eleanity stabilize --backend {backends[0] if backends else 'transformers'} "
            f"--model {model} --repetitions 3"
        )
    return cmds


def format_timings(timings: dict[str, float] | None) -> dict[str, Any]:
    """Pretty timing block with percent share and pairwise delta."""

    if not timings:
        return {"entries": [], "total_ms": 0.0, "delta_percent": None}
    total = sum(timings.values()) or 1.0
    entries = []
    for name, ms in timings.items():
        entries.append(
            {
                "name": name,
                "ms": round(float(ms), 2),
                "share_percent": round(100.0 * float(ms) / total, 1),
            }
        )
    delta_percent = None
    keys = list(timings.keys())
    if len(keys) >= 2:
        a, b = float(timings[keys[0]]), float(timings[keys[1]])
        base = a if a else 1.0
        delta_percent = round(100.0 * (b - a) / base, 1)
    return {
        "entries": entries,
        "total_ms": round(sum(timings.values()), 2),
        "delta_percent": delta_percent,
        "delta_label": (
            f"{keys[1]} is {delta_percent:+.1f}% vs {keys[0]}" if delta_percent is not None and len(keys) >= 2 else None
        ),
    }


def build_reproduction_command(
    *,
    model: str,
    backends: list[str],
    baseline: str | None = None,
    policy: str | None = None,
    observe: list[str] | None = None,
    tokenizer_only: bool = False,
    backend_urls: dict[str, str] | None = None,
    no_gates: bool = False,
    scenario_name: str | None = None,
) -> str:
    parts = ["eleanity", "compare", f"--model {model}", f"--backends {','.join(backends)}"]
    if baseline:
        parts.append(f"--baseline {baseline}")
    if policy:
        parts.append(f"--policy {policy}")
    if observe:
        parts.append(f"--observe {','.join(observe)}")
    if tokenizer_only:
        parts.append("--tokenizer-only")
    if no_gates:
        parts.append("--no-gates")
    if scenario_name:
        parts.append(f"--name {scenario_name}")
    for name, url in (backend_urls or {}).items():
        parts.append(f"--backend-url {name}={url}")
    parts.append("--format text")
    return " ".join(parts)
