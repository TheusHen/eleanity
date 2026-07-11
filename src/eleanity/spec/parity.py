"""Formal parity status taxonomy and per-layer comparator specifications.

PASS_WITH_TOLERANCE is never free-form: it is only emitted when a declared
comparator mode accepts the measured drift under explicit numeric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from eleanity.models.schemas import ParityProfile, ParityResult


class FormalParityStatus(str, Enum):
    """Operational meanings of comparison outcomes (Trace Spec v1)."""

    PASS = "PASS"
    PASS_WITH_TOLERANCE = "PASS_WITH_TOLERANCE"
    DIVERGENT = "DIVERGENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


# Backward-compatible mapping from legacy ParityResult values.
_LEGACY_TO_FORMAL = {
    ParityResult.PASS: FormalParityStatus.PASS,
    ParityResult.PASS_WITH_TOLERANCE: FormalParityStatus.PASS_WITH_TOLERANCE,
    ParityResult.PASS_WITH_LIMITED_COVERAGE: FormalParityStatus.PASS_WITH_TOLERANCE,
    ParityResult.DIVERGENT: FormalParityStatus.DIVERGENT,
    ParityResult.INCOMPARABLE: FormalParityStatus.INCONCLUSIVE,
    ParityResult.INCONCLUSIVE: FormalParityStatus.INCONCLUSIVE,
    ParityResult.NOT_OBSERVABLE: FormalParityStatus.INCONCLUSIVE,
    ParityResult.NOT_REQUESTED: FormalParityStatus.INCONCLUSIVE,
    ParityResult.NOT_SUPPORTED: FormalParityStatus.UNSUPPORTED,
    ParityResult.ERROR: FormalParityStatus.ERROR,
}


STATUS_DEFINITIONS: dict[str, dict[str, str]] = {
    FormalParityStatus.PASS.value: {
        "meaning": "All observed layers that the policy requires match under their comparator mode.",
        "when": "exact modes equal; numerical modes within atol/rtol and ranking thresholds.",
        "never": "Missing observations must not produce PASS.",
    },
    FormalParityStatus.PASS_WITH_TOLERANCE.value: {
        "meaning": "Values differ but stay inside the declared numeric / ranking tolerance of the policy.",
        "when": "Comparator mode is numerical|prefix|topk and measured error ≤ thresholds.",
        "never": "exact-mode layers; or unconfigured tolerances.",
    },
    FormalParityStatus.DIVERGENT.value: {
        "meaning": "At least one required observed layer fails its comparator under the active policy.",
        "when": "exact mismatch, numerical drift beyond thresholds, prefix shorter than required, etc.",
        "never": "When either side is not OBSERVED (use INCONCLUSIVE/UNSUPPORTED instead).",
    },
    FormalParityStatus.INCONCLUSIVE.value: {
        "meaning": "Comparison cannot decide because data is partial, inferred, or self-inconsistent.",
        "when": "NOT_EXPOSED/INFERRED on required layers, or stability protocol failed.",
        "never": "Do not treat missing data as equality.",
    },
    FormalParityStatus.UNSUPPORTED.value: {
        "meaning": "Adapter or backend cannot expose the layer required by the policy.",
        "when": "Capability reports unsupported; layer state UNSUPPORTED.",
        "never": "Silent skip that looks like PASS.",
    },
    FormalParityStatus.ERROR.value: {
        "meaning": "Execution or observation failed (exception, unhealthy backend, bad config).",
        "when": "Trace errors, FAILED observation state, or engine-level exceptions.",
        "never": "Soft policy failures (those are DIVERGENT).",
    },
}


COMPARATOR_MODES = frozenset(
    {
        "exact",
        "numerical",
        "prefix",
        "topk",
        "set_equal",
        "schema",
        "ignore",
    }
)


@dataclass(frozen=True)
class ComparatorSpec:
    """Formal comparator for one layer (or sub-field)."""

    mode: str = "exact"
    atol: float | None = None
    rtol: float | None = None
    top_k: int | None = None
    top_k_agreement: float | None = None
    exact_prefix_tokens: int | None = None
    max_relative_drift: float | None = None
    required: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "atol": self.atol,
            "rtol": self.rtol,
            "top_k": self.top_k,
            "top_k_agreement": self.top_k_agreement,
            "exact_prefix_tokens": self.exact_prefix_tokens,
            "max_relative_drift": self.max_relative_drift,
            "required": self.required,
            "notes": self.notes,
        }


@dataclass
class PolicyComparatorSet:
    policy: str
    description: str
    comparators: dict[str, ComparatorSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "description": self.description,
            "comparators": {k: v.to_dict() for k, v in self.comparators.items()},
        }


def status_definition(status: str | FormalParityStatus | ParityResult) -> dict[str, str]:
    if isinstance(status, FormalParityStatus):
        key = status.value
    elif isinstance(status, ParityResult):
        key = formal_status_from_parity(status).value
    else:
        key = str(status)
    return dict(STATUS_DEFINITIONS.get(key, {"meaning": "unknown status", "when": "", "never": ""}))


def formal_status_from_parity(result: ParityResult | str) -> FormalParityStatus:
    if isinstance(result, str):
        try:
            result = ParityResult(result)
        except ValueError:
            try:
                return FormalParityStatus(result)
            except ValueError:
                return FormalParityStatus.ERROR
    if result in _LEGACY_TO_FORMAL:
        return _LEGACY_TO_FORMAL[result]
    # New values already aligned
    try:
        return FormalParityStatus(result.value)
    except ValueError:
        return FormalParityStatus.ERROR


def policy_comparator_set(profile: ParityProfile | str) -> PolicyComparatorSet:
    """Return the formal, declarative comparator table for a policy profile."""

    name = profile.value if isinstance(profile, ParityProfile) else str(profile)
    if name == ParityProfile.STRICT.value:
        return PolicyComparatorSet(
            policy=name,
            description="Byte/id exact parity. Logits only with tiny numerical tolerance when observed.",
            comparators={
                "artifact": ComparatorSpec(mode="exact", required=True),
                "chat_template": ComparatorSpec(mode="exact", required=True),
                "rendered_prompt": ComparatorSpec(mode="exact", required=True),
                "special_tokens": ComparatorSpec(mode="exact", required=True),
                "input_token_ids": ComparatorSpec(mode="exact", required=True),
                "prefill_logits": ComparatorSpec(
                    mode="numerical",
                    atol=1e-5,
                    rtol=1e-5,
                    top_k=5,
                    top_k_agreement=1.0,
                    required=False,
                    notes="Required only when both sides OBSERVED.",
                ),
                "generated_token_ids": ComparatorSpec(mode="exact", required=True),
                "finish_reason": ComparatorSpec(mode="exact", required=True),
                "stop_decision": ComparatorSpec(mode="exact", required=True),
            },
        )
    if name == ParityProfile.QUANTIZED.value:
        return PolicyComparatorSet(
            policy=name,
            description=(
                "Exact template/token ids; numerical prefill logits with atol/rtol; "
                "generation prefix match; finish_reason exact."
            ),
            comparators={
                "artifact": ComparatorSpec(
                    mode="exact",
                    required=True,
                    notes="Ignore dtype/quantization flags that differ by design.",
                ),
                "chat_template": ComparatorSpec(mode="exact", required=True),
                "rendered_prompt": ComparatorSpec(mode="exact", required=True),
                "special_tokens": ComparatorSpec(mode="exact", required=True),
                "input_token_ids": ComparatorSpec(mode="exact", required=True),
                "prefill_logits": ComparatorSpec(
                    mode="numerical",
                    atol=1.0e-4,
                    rtol=1.0e-3,
                    top_k=10,
                    top_k_agreement=0.99,
                    required=False,
                ),
                "decode_logits": ComparatorSpec(
                    mode="numerical",
                    atol=1.0e-3,
                    rtol=1.0e-2,
                    top_k=10,
                    top_k_agreement=0.95,
                    required=False,
                ),
                "generated_token_ids": ComparatorSpec(
                    mode="prefix",
                    exact_prefix_tokens=16,
                    required=True,
                ),
                "finish_reason": ComparatorSpec(mode="exact", required=True),
            },
        )
    if name == ParityProfile.FUNCTIONAL.value:
        return PolicyComparatorSet(
            policy=name,
            description="Functional outcome parity: structure, tools, stop — not logits/tokens.",
            comparators={
                "artifact": ComparatorSpec(mode="ignore", required=False),
                "chat_template": ComparatorSpec(mode="ignore", required=False),
                "input_token_ids": ComparatorSpec(mode="ignore", required=False),
                "prefill_logits": ComparatorSpec(mode="ignore", required=False),
                "generated_token_ids": ComparatorSpec(mode="ignore", required=False),
                "finish_reason": ComparatorSpec(mode="exact", required=True),
                "structured_output": ComparatorSpec(mode="schema", required=True),
                "tool_call_parsing": ComparatorSpec(mode="schema", required=True),
                "response_mapping": ComparatorSpec(mode="schema", required=False),
            },
        )
    # api_conformance
    return PolicyComparatorSet(
        policy=name,
        description="OpenAI-compatible API shape, status, usage, finish_reason, streaming order.",
        comparators={
            "api": ComparatorSpec(mode="schema", required=True),
            "streaming": ComparatorSpec(mode="exact", required=False),
            "finish_reason": ComparatorSpec(mode="exact", required=True),
            "usage_accounting": ComparatorSpec(mode="exact", required=False),
            "response_mapping": ComparatorSpec(mode="schema", required=True),
            "chat_template": ComparatorSpec(mode="ignore", required=False),
            "input_token_ids": ComparatorSpec(mode="ignore", required=False),
            "prefill_logits": ComparatorSpec(mode="ignore", required=False),
        },
    )


def apply_numerical_thresholds(
    *,
    max_abs_diff: float | None,
    max_rel_diff: float | None,
    top_k_agreement: float | None,
    spec: ComparatorSpec,
) -> FormalParityStatus:
    """Decide PASS / PASS_WITH_TOLERANCE / DIVERGENT for numerical mode."""

    if spec.mode != "numerical":
        raise ValueError("apply_numerical_thresholds requires numerical mode")
    atol = spec.atol if spec.atol is not None else 0.0
    rtol = spec.rtol if spec.rtol is not None else 0.0
    if max_abs_diff is None and max_rel_diff is None and top_k_agreement is None:
        return FormalParityStatus.INCONCLUSIVE
    if max_abs_diff is not None and max_abs_diff == 0 and (max_rel_diff is None or max_rel_diff == 0):
        if top_k_agreement is None or top_k_agreement >= (spec.top_k_agreement or 1.0):
            return FormalParityStatus.PASS
    within_abs = max_abs_diff is None or max_abs_diff <= atol
    within_rel = max_rel_diff is None or max_rel_diff <= rtol
    within_topk = True
    if spec.top_k_agreement is not None and top_k_agreement is not None:
        within_topk = top_k_agreement + 1e-12 >= spec.top_k_agreement
    if within_abs and within_rel and within_topk:
        if (max_abs_diff or 0) > 0 or (max_rel_diff or 0) > 0 or (
            top_k_agreement is not None and top_k_agreement < 1.0
        ):
            return FormalParityStatus.PASS_WITH_TOLERANCE
        return FormalParityStatus.PASS
    return FormalParityStatus.DIVERGENT


def apply_prefix_thresholds(
    equal_prefix: int,
    left_len: int,
    right_len: int,
    spec: ComparatorSpec,
) -> FormalParityStatus:
    need = spec.exact_prefix_tokens or 0
    if need <= 0:
        return FormalParityStatus.PASS if left_len == right_len and equal_prefix == left_len else FormalParityStatus.DIVERGENT
    if equal_prefix >= need:
        if left_len == right_len and equal_prefix == left_len:
            return FormalParityStatus.PASS
        return FormalParityStatus.PASS_WITH_TOLERANCE
    return FormalParityStatus.DIVERGENT
