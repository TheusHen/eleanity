"""Separate parity (internal) from functional impact (user-visible effect)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from eleanity.models.schemas import Comparison, ObservationTrace, ParityResult
from eleanity.spec.layers import LAYER_ORDER_V1, canonicalize_layer


class FunctionalImpact(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CATASTROPHIC = "CATASTROPHIC"


# Layers ordered by how early they tend to affect final user-visible output.
_IMPACT_WEIGHT: dict[str, FunctionalImpact] = {
    "artifact": FunctionalImpact.MEDIUM,
    "model_config": FunctionalImpact.MEDIUM,
    "tokenizer_artifact": FunctionalImpact.HIGH,
    "chat_template": FunctionalImpact.HIGH,
    "rendered_prompt": FunctionalImpact.HIGH,
    "normalization": FunctionalImpact.MEDIUM,
    "special_tokens": FunctionalImpact.HIGH,
    "input_token_ids": FunctionalImpact.HIGH,
    "generation_config": FunctionalImpact.MEDIUM,
    "prefill_logits": FunctionalImpact.LOW,
    "decode_logits": FunctionalImpact.LOW,
    "logits_processing": FunctionalImpact.MEDIUM,
    "sampling": FunctionalImpact.MEDIUM,
    "generated_token_ids": FunctionalImpact.HIGH,
    "stop_decision": FunctionalImpact.HIGH,
    "detokenization": FunctionalImpact.MEDIUM,
    "response_mapping": FunctionalImpact.HIGH,
    "streaming": FunctionalImpact.LOW,
    "usage_accounting": FunctionalImpact.NONE,
    "structured_output": FunctionalImpact.CATASTROPHIC,
    "tool_call_parsing": FunctionalImpact.CATASTROPHIC,
    "multimodal_inputs": FunctionalImpact.HIGH,
    "embeddings": FunctionalImpact.HIGH,
    "reasoning_content": FunctionalImpact.LOW,
    "speculative_decoding": FunctionalImpact.LOW,
    # legacy coarse
    "template": FunctionalImpact.HIGH,
    "tokens": FunctionalImpact.HIGH,
    "logits": FunctionalImpact.LOW,
    "generation": FunctionalImpact.HIGH,
    "structured": FunctionalImpact.CATASTROPHIC,
    "api": FunctionalImpact.HIGH,
}


_IMPACT_RANK = {
    FunctionalImpact.NONE: 0,
    FunctionalImpact.LOW: 1,
    FunctionalImpact.MEDIUM: 2,
    FunctionalImpact.HIGH: 3,
    FunctionalImpact.CATASTROPHIC: 4,
}


class ImpactAssessment(BaseModel):
    """Dual-axis companion to parity status."""

    parity: str
    impact: FunctionalImpact = FunctionalImpact.NONE
    first_divergence_layer: str | None = None
    final_token_sequence_equal: bool | None = None
    final_text_equal: bool | None = None
    structured_valid_both: bool | None = None
    first_generated_token_changed: bool | None = None
    propagation_layers: list[str] = Field(default_factory=list)
    rationale: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _gen_data(trace: ObservationTrace) -> dict[str, Any]:
    layer = trace.layers.get("generation")
    return dict(layer.data) if layer and layer.data else {}


def _token_data(trace: ObservationTrace) -> dict[str, Any]:
    layer = trace.layers.get("tokens")
    return dict(layer.data) if layer and layer.data else {}


def assess_impact(
    *,
    parity_status: str | ParityResult,
    first_divergence: str | None,
    left: ObservationTrace | None = None,
    right: ObservationTrace | None = None,
    comparisons: dict[str, Comparison] | dict[str, Any] | None = None,
) -> ImpactAssessment:
    """Compute functional impact independent of internal parity label."""

    parity = parity_status.value if isinstance(parity_status, ParityResult) else str(parity_status)
    if parity in {"PASS", "PASS_WITH_TOLERANCE"}:
        return ImpactAssessment(
            parity=parity,
            impact=FunctionalImpact.NONE,
            first_divergence_layer=None,
            final_token_sequence_equal=True,
            final_text_equal=True,
            rationale="Parity holds under policy; no functional impact attributed.",
        )

    layer = canonicalize_layer(first_divergence) if first_divergence else None
    base_impact = _IMPACT_WEIGHT.get(layer or "", FunctionalImpact.MEDIUM)

    final_ids_equal: bool | None = None
    final_text_equal: bool | None = None
    first_token_changed: bool | None = None
    structured_ok: bool | None = None
    propagation: list[str] = []

    if left is not None and right is not None:
        lg, rg = _gen_data(left), _gen_data(right)
        lids = lg.get("ids") or lg.get("token_ids") or []
        rids = rg.get("ids") or rg.get("token_ids") or []
        if lids or rids:
            final_ids_equal = list(lids) == list(rids)
            first_token_changed = (lids[0] != rids[0] if lids and rids else lids != rids) if lids and rids else True
        ltext = lg.get("text")
        rtext = rg.get("text")
        if ltext is not None or rtext is not None:
            final_text_equal = (ltext or "") == (rtext or "")

        ls = left.layers.get("structured")
        rs = right.layers.get("structured")
        if ls and rs and ls.data is not None and rs.data is not None:
            structured_ok = ls.data == rs.data

        # Propagation: layers after first divergence that also diverge
        if comparisons and layer:
            started = False
            order = list(LAYER_ORDER_V1) + ["template", "tokens", "logits", "generation", "structured", "api"]
            for name in order:
                canon = canonicalize_layer(name)
                if first_divergence and (name == first_divergence or canon == layer):
                    started = True
                if not started:
                    continue
                entry = comparisons.get(name) or comparisons.get(canon)
                if entry is None:
                    continue
                result = (
                    entry.result
                    if hasattr(entry, "result")
                    else (entry.get("result") if isinstance(entry, dict) else None)
                )
                value = result.value if hasattr(result, "value") else result
                if value == "DIVERGENT":
                    propagation.append(name)

    # Soften impact when internal divergence has no user-visible effect
    impact = base_impact
    if parity == "DIVERGENT" and final_ids_equal is True and final_text_equal is not False:
        # e.g. prefill_logits diverge but greedy decode yields same tokens
        if layer in {"prefill_logits", "decode_logits", "logits", "logits_processing", "sampling"}:
            impact = FunctionalImpact.NONE
            rationale = (
                f"First divergence at {layer}, but final token sequence is equal — "
                "internal numeric drift without functional effect."
            )
        else:
            impact = FunctionalImpact.LOW
            rationale = f"Divergence at {layer} with equal final token sequence."
    elif parity == "DIVERGENT" and first_token_changed:
        impact = max(impact, FunctionalImpact.HIGH, key=lambda x: _IMPACT_RANK[x])
        rationale = f"First divergence at {layer}; first generated token changed — downstream generation path diverged."
    elif parity == "DIVERGENT" and structured_ok is False:
        impact = FunctionalImpact.CATASTROPHIC
        rationale = f"Divergence at {layer} invalidated structured/tool output."
    elif parity in {"INCONCLUSIVE", "NOT_OBSERVABLE", "INCOMPARABLE", "UNSUPPORTED"}:
        impact = FunctionalImpact.NONE
        rationale = "Parity is inconclusive/unsupported; impact not attributed (insufficient observation)."
    elif parity == "ERROR":
        impact = FunctionalImpact.HIGH
        rationale = "Execution error; functional outcome unreliable."
    else:
        rationale = f"Impact estimated from first divergence layer '{layer}'."

    # Escalation: high-weight early layers always at least HIGH if tokens differ
    if final_ids_equal is False and layer in {
        "chat_template",
        "rendered_prompt",
        "template",
        "input_token_ids",
        "tokens",
        "special_tokens",
    }:
        impact = max(impact, FunctionalImpact.HIGH, key=lambda x: _IMPACT_RANK[x])
        rationale = f"First divergence at {layer} changed the input path; generated token sequence differs."

    return ImpactAssessment(
        parity=parity,
        impact=impact,
        first_divergence_layer=first_divergence,
        final_token_sequence_equal=final_ids_equal,
        final_text_equal=final_text_equal,
        structured_valid_both=structured_ok,
        first_generated_token_changed=first_token_changed,
        propagation_layers=propagation,
        rationale=rationale,
        evidence={
            "layer_default_impact": base_impact.value if layer else None,
        },
    )
