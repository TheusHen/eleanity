"""Enrich diagnosis + comparisons after the causal walk."""

from __future__ import annotations

from typing import Any

from eleanity.core.coverage import (
    apply_coverage_to_status,
    compute_coverage,
    diagnosis_confidence,
    practical_commands_for,
)
from eleanity.models.schemas import (
    Comparison,
    Diagnosis,
    ObservationTrace,
    ParityResult,
    Scenario,
)


ARTIFACT_COMPARE_KEYS = (
    "model_ref",
    "revision",
    "commit_sha",
    "model_hash",
    "config_hash",
    "tokenizer",
    "tokenizer_hash",
    "chat_template_hash",
    "quantization",
    "dtype",
    "architecture",
)


def collect_tolerance_reasons(
    comparisons: dict[str, dict[str, Any]] | dict[str, Comparison],
) -> list[str]:
    reasons: list[str] = []
    # comparisons may be nested by backend key
    for key, layers in comparisons.items():
        if not isinstance(layers, dict):
            continue
        # detect nested backend map vs flat layer map
        sample = next(iter(layers.values()), None) if layers else None
        if sample is not None and isinstance(sample, dict) and "result" not in sample and any(
            isinstance(v, dict) and "result" in v for v in layers.values()
        ):
            layer_map = layers
        elif sample is not None and (
            hasattr(sample, "result") or (isinstance(sample, dict) and "result" in sample)
        ):
            layer_map = layers
        else:
            layer_map = layers
        for layer, entry in layer_map.items():
            if hasattr(entry, "result"):
                result = entry.result.value if hasattr(entry.result, "value") else str(entry.result)
                details = entry.details or {}
                tol = getattr(entry, "tolerance_reason", None) or details.get("tolerance_reason") or details.get("reason")
            elif isinstance(entry, dict):
                result = str(entry.get("result") or "")
                details = entry.get("details") or {}
                tol = entry.get("tolerance_reason") or details.get("tolerance_reason") or details.get("reason")
            else:
                continue
            if result == ParityResult.PASS_WITH_TOLERANCE.value and tol:
                reasons.append(f"{layer}: {tol}")
            elif result == ParityResult.PASS_WITH_TOLERANCE.value:
                reasons.append(f"{layer}: within configured tolerance thresholds")
    # unique preserve order
    out: list[str] = []
    for r in reasons:
        if r not in out:
            out.append(r)
    return out


def artifact_divergent_fields(left: ObservationTrace, right: ObservationTrace) -> list[str]:
    la = left.layers.get("artifact")
    ra = right.layers.get("artifact")
    if not la or not ra:
        return []
    ld, rd = la.data or {}, ra.data or {}
    return [k for k in ARTIFACT_COMPARE_KEYS if ld.get(k) != rd.get(k)]


def enrich_diagnosis(
    diagnosis: Diagnosis,
    *,
    left: ObservationTrace,
    right: ObservationTrace,
    scenario: Scenario | None,
    comparisons: dict[str, Any] | None,
    model: str | None = None,
    backends: list[str] | None = None,
    run_id: str | None = None,
) -> Diagnosis:
    """Attach coverage, confidence, verified sets, tolerance reasons, commands."""

    # Flatten first candidate comparisons if nested
    flat: dict[str, Any] = {}
    if comparisons:
        first = next(iter(comparisons.values()), {})
        if isinstance(first, dict) and first and (
            "result" in first or hasattr(next(iter(first.values()), None), "result")
        ):
            flat = first
        else:
            # maybe already flat layer→comparison
            flat = first if isinstance(first, dict) else {}

    coverage = compute_coverage(
        left,
        right,
        scenario=scenario,
        policy=scenario.parity_profile if scenario else None,
        comparisons=flat,
    )
    status, cov_reasons = apply_coverage_to_status(diagnosis.status, coverage)
    tolerance_reasons = collect_tolerance_reasons(comparisons or {})
    # also from coverage downgrade
    for r in cov_reasons:
        if r not in tolerance_reasons:
            tolerance_reasons.append(r)

    conf = diagnosis_confidence(
        status=status,
        coverage=coverage,
        probable_causes=diagnosis.probable_causes,
    )
    art_fields = artifact_divergent_fields(left, right)
    cmds = practical_commands_for(
        status=status,
        first_divergence=diagnosis.first_divergence,
        model=model or (left.artifact_fingerprint.model_ref if left else None),
        backends=backends or [left.backend, right.backend],
        run_id=run_id,
    )
    # Prefer remediation from causes when present
    for cause in diagnosis.probable_causes:
        if cause.suggested_remediation and cause.suggested_remediation not in cmds:
            cmds.insert(0, cause.suggested_remediation)

    summary = diagnosis.summary
    if status == ParityResult.PASS_WITH_LIMITED_COVERAGE and status != diagnosis.status:
        summary = (
            f"{summary} Coverage limited: {coverage.get('required_coverage_percent')}% of "
            f"required layers verified (min {coverage.get('min_coverage_percent')}%)."
        )
    if status == ParityResult.INCONCLUSIVE and diagnosis.status != ParityResult.INCONCLUSIVE:
        summary = (
            "Inconclusive: insufficient mutually observed layers for a confident parity verdict. "
            f"Coverage {coverage.get('required_coverage_percent')}%."
        )

    return diagnosis.model_copy(
        update={
            "status": status,
            "formal_status": status.value,
            "confidence": conf,
            "coverage": coverage,
            "verified_layers": list(coverage.get("verified_layers") or []),
            "not_verified_layers": list(coverage.get("not_verified_layers") or []),
            "tolerance_reasons": tolerance_reasons,
            "artifact_divergent_fields": art_fields,
            "practical_commands": cmds,
            "summary": summary,
            "suggested_actions": list(dict.fromkeys([*cmds[:3], *diagnosis.suggested_actions]))[:8],
        }
    )
