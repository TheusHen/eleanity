from __future__ import annotations

from typing import Any

from eleanity.comparators.api import compare_api, compare_streaming
from eleanity.comparators.diff import (
    compare_generation,
    compare_logits,
    compare_prompt,
    compare_special_tokens,
    compare_tokens,
)
from eleanity.comparators.structured import compare_structured
from eleanity.models.schemas import (
    Comparison,
    LayerObservation,
    LayerState,
    ObservationTrace,
    ParityProfile,
    ParityResult,
    Scenario,
)
from eleanity.policies import policy_rules


class PolicyEngine:
    """Apply parity policy rules to layer pairs — honest about observability."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.profile = scenario.parity_profile
        self.tolerance = float(scenario.tolerance or 0.0)
        self.rules = policy_rules(self.profile)

    def compare_layers(self, left: ObservationTrace, right: ObservationTrace) -> dict[str, Comparison]:
        results: dict[str, Comparison] = {}
        for layer, a in left.layers.items():
            b = right.layers.get(layer)
            results[layer] = self.compare_layer(layer, a, b)
        # Include layers only present on right
        for layer, b in right.layers.items():
            if layer not in results:
                results[layer] = self.compare_layer(layer, None, b)
        return results

    def compare_layer(
        self,
        layer: str,
        left: LayerObservation | None,
        right: LayerObservation | None,
    ) -> Comparison:
        requested = layer in set(self.scenario.observe) or layer == "artifact"
        if (
            left is None
            or right is None
            or (
                left is not None
                and right is not None
                and (left.state != LayerState.OBSERVED or right.state != LayerState.OBSERVED)
            )
        ):
            from eleanity.core.coverage import classify_unobserved

            if left is not None and right is not None:
                if left.state in {LayerState.ERROR, LayerState.FAILED} or right.state in {
                    LayerState.ERROR,
                    LayerState.FAILED,
                }:
                    return Comparison(
                        result=ParityResult.ERROR,
                        details={"left_note": left.note, "right_note": right.note},
                        baseline_state=left.state,
                        candidate_state=right.state,
                    )
            result = classify_unobserved(requested=requested, left=left, right=right)
            return Comparison(
                result=result,
                details={
                    "left_state": left.state.value if left else None,
                    "right_state": right.state.value if right else None,
                    "left_note": left.note if left else "missing layer",
                    "right_note": right.note if right else "missing layer",
                    "left_origin": left.origin if left else None,
                    "right_origin": right.origin if right else None,
                    "requested": requested,
                },
                baseline_state=left.state if left else None,
                candidate_state=right.state if right else None,
            )

        # Policy gates
        if layer == "logits" and self.rules.get("require_logits") is False:
            # Still compare if both observed, but functional policy may soften
            pass
        if layer == "template" and self.rules.get("require_identical_prompt_bytes") is False:
            # functional/api may skip exact prompt equality — mark INCOMPARABLE for policy
            if self.profile in {ParityProfile.FUNCTIONAL, ParityProfile.API_CONFORMANCE}:
                return Comparison(
                    result=ParityResult.INCOMPARABLE,
                    details={"reason": f"policy {self.profile.value} does not require prompt bytes"},
                )
        if layer == "tokens" and self.rules.get("require_identical_token_ids") is False:
            if self.profile in {ParityProfile.FUNCTIONAL, ParityProfile.API_CONFORMANCE}:
                return Comparison(
                    result=ParityResult.INCOMPARABLE,
                    details={"reason": f"policy {self.profile.value} does not require token ids"},
                )

        return self._dispatch(layer, left.data, right.data)

    def _dispatch(self, layer: str, left: dict[str, Any], right: dict[str, Any]) -> Comparison:
        if layer == "template":
            return compare_prompt(
                str(left.get("text") or left.get("rendered_text") or ""),
                str(right.get("text") or right.get("rendered_text") or ""),
            )
        if layer == "tokens":
            return compare_tokens(
                left.get("ids") or left.get("token_ids") or [],
                right.get("ids") or right.get("token_ids") or [],
                left_strings=left.get("token_strings"),
                right_strings=right.get("token_strings"),
                left_special=left,
                right_special=right,
            )
        if layer == "special_tokens":
            return compare_special_tokens(left, right)
        if layer == "logits":
            return compare_logits(
                left.get("top_logits", []),
                right.get("top_logits", []),
                self.tolerance,
            )
        if layer == "generation":
            if self.profile == ParityProfile.FUNCTIONAL:
                # Functional: prefer stop_reason / structural signals over exact text
                if left.get("stop_reason") != right.get("stop_reason"):
                    return Comparison(
                        result=ParityResult.DIVERGENT,
                        details={
                            "reason": "stop_reason differs under functional policy",
                            "left": left.get("stop_reason"),
                            "right": right.get("stop_reason"),
                        },
                    )
                left_ids = left.get("ids") or left.get("token_ids") or []
                right_ids = right.get("ids") or right.get("token_ids") or []
                if left_ids and right_ids and self.rules.get("require_deterministic_generation"):
                    return compare_generation(left_ids, right_ids)
                return Comparison(
                    result=ParityResult.PASS,
                    details={"reason": "functional policy: stop_reason aligned"},
                )
            left_ids = list(left.get("ids") or left.get("token_ids") or [])
            right_ids = list(right.get("ids") or right.get("token_ids") or [])
            left_text = str(left.get("text") or "")
            right_text = str(right.get("text") or "")
            # OpenAI-compat servers often omit generated token ids — fall back to text.
            if (left_ids and not right_ids) or (right_ids and not left_ids):
                text_cmp = compare_prompt(left_text, right_text)
                details = {
                    **text_cmp.details,
                    "reason": "one side missing generated token ids; compared text",
                    "left_id_count": len(left_ids),
                    "right_id_count": len(right_ids),
                    "left_stop": left.get("stop_reason") or left.get("finish_reason"),
                    "right_stop": right.get("stop_reason") or right.get("finish_reason"),
                }
                if text_cmp.result == ParityResult.PASS and self.profile in {
                    ParityProfile.QUANTIZED,
                    ParityProfile.FUNCTIONAL,
                }:
                    return Comparison(
                        result=ParityResult.PASS_WITH_TOLERANCE,
                        details=details,
                        tolerance_reason=("one backend omitted generated token ids; decoded text matches"),
                    )
                return Comparison(result=text_cmp.result, details=details)
            if left_ids or right_ids:
                cmp = compare_generation(left_ids, right_ids)
                # Formal quantized policy: prefix mode with exact_prefix_tokens
                if self.profile == ParityProfile.QUANTIZED and cmp.result == ParityResult.DIVERGENT:
                    from eleanity.spec.observability import legacy_parity_from_formal
                    from eleanity.spec.parity import (
                        FormalParityStatus,
                        apply_prefix_thresholds,
                        policy_comparator_set,
                    )

                    spec = policy_comparator_set(self.profile).comparators.get("generated_token_ids")
                    if spec and spec.mode == "prefix":
                        equal_prefix = 0
                        for a, b in zip(left_ids, right_ids, strict=False):
                            if a != b:
                                break
                            equal_prefix += 1
                        formal = apply_prefix_thresholds(equal_prefix, len(left_ids), len(right_ids), spec)
                        if formal in {
                            FormalParityStatus.PASS,
                            FormalParityStatus.PASS_WITH_TOLERANCE,
                        }:
                            return Comparison(
                                result=legacy_parity_from_formal(formal),
                                details={
                                    **cmp.details,
                                    "comparator": spec.to_dict(),
                                    "equal_prefix": equal_prefix,
                                    "formal_status": formal.value,
                                },
                            )
                # If ids diverge but decoded text matches under quantized, soften to tolerance
                if (
                    self.profile == ParityProfile.QUANTIZED
                    and cmp.result == ParityResult.DIVERGENT
                    and left_text
                    and left_text == right_text
                ):
                    return Comparison(
                        result=ParityResult.PASS_WITH_TOLERANCE,
                        details={
                            **cmp.details,
                            "reason": "generated text equal despite token-id drift",
                            "text": left_text,
                        },
                    )
                return cmp
            return compare_prompt(left_text, right_text)
        if layer == "structured":
            return compare_structured(left, right)
        if layer == "streaming":
            return compare_streaming(left, right)
        if layer == "api":
            return compare_api(left, right)
        if layer == "artifact":
            keys = (
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
            left_view = {k: left.get(k) for k in keys}
            right_view = {k: right.get(k) for k in keys}
            if left_view == right_view:
                return Comparison(result=ParityResult.PASS, details={"keys": list(keys)})
            # Under quantized policy, identity/runtime packaging drift is expected
            # (HF bf16 vs LM Studio Q8, different model_ref spelling, null hashes on HTTP).
            if self.profile == ParityProfile.QUANTIZED:
                soft = {
                    "dtype",
                    "quantization",
                    "model_ref",
                    "model_hash",
                    "config_hash",
                    "commit_sha",
                    "revision",
                }
                hard_left = {
                    k: v
                    for k, v in left_view.items()
                    if k not in soft and v is not None and right_view.get(k) is not None
                }
                hard_right = {
                    k: v
                    for k, v in right_view.items()
                    if k not in soft and v is not None and left_view.get(k) is not None
                }
                if hard_left == hard_right:
                    soft_diff = [k for k in soft if left_view.get(k) != right_view.get(k)]
                    return Comparison(
                        result=ParityResult.PASS_WITH_TOLERANCE,
                        details={
                            "reason": "quantization/runtime identity differ under quantized policy",
                            "tolerance_reason": (
                                "artifact soft fields differ under quantized policy: "
                                + ", ".join(soft_diff or ["dtype/quantization"])
                            ),
                            "divergent_fields": soft_diff,
                            "left": left_view,
                            "right": right_view,
                        },
                        tolerance_reason=(
                            "artifact soft fields differ under quantized policy: "
                            + ", ".join(soft_diff or ["dtype/quantization"])
                        ),
                    )
            divergent = [k for k in keys if left_view.get(k) != right_view.get(k)]
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={
                    "divergent_keys": divergent,
                    "divergent_fields": divergent,
                    "left": left_view,
                    "right": right_view,
                },
            )
        return Comparison(
            result=ParityResult.PASS if left == right else ParityResult.DIVERGENT,
            details={},
        )
