from __future__ import annotations

from typing import Any

from eleanity.comparators.diff import compare_prompt, compare_special_tokens, compare_tokens
from eleanity.diagnosers.rules import (
    actions_for,
    diagnose_artifact,
    diagnose_generation,
    diagnose_template,
    diagnose_tokens,
)
from eleanity.models.schemas import (
    Diagnosis,
    DivergenceLocation,
    FirstDivergence,
    LayerState,
    ObservationTrace,
    ParityResult,
    ProbableCause,
    PropagationInfo,
)

ORDER = [
    "artifact",
    "template",
    "special_tokens",
    "tokens",
    "logits",
    "generation",
    "structured",
    "streaming",
    "api",
]

ARTIFACT_KEYS = {
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
}


def _snippet(text: str, index: int | None, width: int = 40) -> str:
    if not text:
        return ""
    if index is None:
        return text[:width]
    start = max(0, index)
    return text[start : start + width]


def diagnose(traces: list[ObservationTrace], comparisons: dict[str, Any] | None = None) -> Diagnosis:
    """Rule-based causal diagnoser. Deterministic — no LLM."""

    if len(traces) < 2:
        return Diagnosis(
            status=ParityResult.INCONCLUSIVE,
            hypothesis="At least two traces are required.",
            next_test="Add another backend.",
            summary="No pair to compare.",
            suggested_actions=["Provide at least two backend traces."],
        )

    # Surface hard errors first
    for trace in traces[:2]:
        if trace.errors:
            err = trace.errors[0]
            return Diagnosis(
                status=ParityResult.ERROR,
                first_divergence=err.layer or "error",
                hypothesis=err.message,
                next_test="Fix the failing backend configuration and re-run.",
                summary=f"ERROR: {err.message}",
                probable_causes=[ProbableCause(code=err.code, confidence=1.0, message=err.message)],
                suggested_actions=["Inspect adapter logs and healthcheck.", err.detail or ""],
                warnings=list(trace.warnings),
            )

    left, right = traces[0], traces[1]
    warnings = list(left.warnings) + list(right.warnings)

    for layer in ORDER:
        a, b = left.layers.get(layer), right.layers.get(layer)
        if not a or not b:
            continue
        if a.state in {LayerState.ERROR, LayerState.FAILED} or b.state in {LayerState.ERROR, LayerState.FAILED}:
            return Diagnosis(
                status=ParityResult.ERROR,
                first_divergence=layer,
                hypothesis=a.note or b.note or "Layer execution error.",
                next_test=f"Repair {layer} observation on the failing backend.",
                summary=f"ERROR on layer {layer}.",
                probable_causes=[
                    ProbableCause(
                        code="LAYER_ERROR",
                        confidence=1.0,
                        message=a.note or b.note or "error",
                    )
                ],
                suggested_actions=[f"Inspect {layer} on {left.backend} and {right.backend}."],
                warnings=warnings,
            )
        if a.state != LayerState.OBSERVED or b.state != LayerState.OBSERVED:
            continue

        if comparisons:
            comp = comparisons.get(layer)
            if comp:
                if hasattr(comp, "result"):
                    res = comp.result.value
                elif isinstance(comp, dict):
                    res = comp.get("result")
                else:
                    res = str(comp)
                if res in {ParityResult.PASS.value, ParityResult.PASS_WITH_TOLERANCE.value}:
                    continue

        if layer == "artifact":
            left_view = {key: a.data.get(key) for key in ARTIFACT_KEYS}
            right_view = {key: b.data.get(key) for key in ARTIFACT_KEYS}
            if left_view != right_view:
                # Soft identity drift is expected across HF full-precision vs local
                # quantized runtimes (LM Studio Q8, different model_ref spelling, missing hashes).
                # Soft only for packaging/identity noise across HF vs local quant runtimes.
                # revision/tokenizer_hash remain hard when both sides provide values.
                soft_keys = {
                    "model_ref",
                    "quantization",
                    "dtype",
                    "local_path",
                    "runtime_version",
                }
                # Hashes that are missing on one side (HTTP adapters) are soft; both present → hard
                soft_if_partial = {"model_hash", "config_hash", "commit_sha"}
                hard = []
                for key in ARTIFACT_KEYS:
                    lv, rv = left_view.get(key), right_view.get(key)
                    if lv == rv:
                        continue
                    if key in soft_keys:
                        continue
                    if key in soft_if_partial and (lv is None or rv is None):
                        continue
                    if lv is None or rv is None:
                        continue
                    hard.append(key)
                if not hard:
                    warnings.append(
                        "Artifact soft-mismatch (model_ref/quant/dtype/hashes) — "
                        "continuing causal walk under multi-runtime compare."
                    )
                    continue
                causes = diagnose_artifact(a.data, b.data)
                actions = actions_for(causes, layer)
                return Diagnosis(
                    status=ParityResult.DIVERGENT,
                    first_divergence=layer,
                    first_divergence_detail=FirstDivergence(
                        layer=layer,
                        baseline=str(left_view.get("model_ref")),
                        candidate=str(right_view.get("model_ref")),
                    ),
                    probable_causes=causes,
                    suggested_actions=actions,
                    hypothesis=causes[0].message if causes else "Artifact mismatch.",
                    next_test=actions[0] if actions else "Align artifacts.",
                    summary="First divergence is in the artifact layer.",
                    warnings=warnings,
                )

        elif layer == "template":
            left_text = str(a.data.get("text") or a.data.get("rendered_text") or "")
            right_text = str(b.data.get("text") or b.data.get("rendered_text") or "")
            result = compare_prompt(left_text, right_text)
            if result.result == ParityResult.DIVERGENT:
                details = result.details
                char_i = details.get("first_character")
                byte_i = details.get("first_byte")
                point = char_i if char_i is not None else details.get("first_difference")
                token_a, token_b = left.layers.get("tokens"), right.layers.get("tokens")
                token_result = None
                if (
                    token_a
                    and token_b
                    and token_a.state == LayerState.OBSERVED
                    and token_b.state == LayerState.OBSERVED
                ):
                    token_result = compare_tokens(
                        token_a.data.get("ids") or token_a.data.get("token_ids") or [],
                        token_b.data.get("ids") or token_b.data.get("token_ids") or [],
                    )
                propagation_percent = (
                    float(token_result.details.get("downstream_percent", 0.0)) if token_result else 0.0
                )
                token_index = token_result.details.get("first_difference") if token_result else None
                causes = diagnose_template(a.data, b.data, details)
                actions = actions_for(causes, layer)
                hypothesis = causes[0].message if causes else "Template divergence."
                return Diagnosis(
                    status=ParityResult.DIVERGENT,
                    first_divergence=layer,
                    first_divergence_detail=FirstDivergence(
                        layer=layer,
                        location=DivergenceLocation(
                            character=char_i,
                            byte=byte_i,
                            line=details.get("line"),
                            column=details.get("column"),
                            token_index=token_index if isinstance(token_index, int) else None,
                        ),
                        baseline=_snippet(left_text, char_i),
                        candidate=_snippet(right_text, char_i),
                    ),
                    propagation=PropagationInfo(
                        first_token_difference=token_index if isinstance(token_index, int) else None,
                        different_tokens_percent=propagation_percent,
                        downstream_different=(
                            token_result.details.get("downstream_different") if token_result else None
                        ),
                    ),
                    propagation_percent=propagation_percent,
                    probable_causes=causes,
                    suggested_actions=actions,
                    hypothesis=hypothesis,
                    next_test=actions[0] if actions else "Compare templates.",
                    summary=(
                        f"First divergence is in the chat template at character {point}. "
                        f"After that, {propagation_percent:.1f}% of tokens differ from index "
                        f"{token_index if token_index is not None else 0}. "
                        f"Likely cause: {hypothesis.rstrip('.')}."
                    ),
                    warnings=warnings,
                )

        elif layer == "special_tokens":
            result = compare_special_tokens(a.data, b.data)
            if result.result == ParityResult.DIVERGENT:
                causes = diagnose_tokens(a.data, b.data, result.details)
                # Prefer special-token oriented causes
                if "eos_token_id" in (result.details.get("divergent_keys") or []):
                    causes = [
                        ProbableCause(
                            code="EOS_DIFFERENT",
                            confidence=0.92,
                            message="EOS token id differs between backends.",
                        )
                    ] + [c for c in causes if c.code != "EOS_DIFFERENT"]
                actions = actions_for(causes, layer)
                return Diagnosis(
                    status=ParityResult.DIVERGENT,
                    first_divergence=layer,
                    first_divergence_detail=FirstDivergence(layer=layer),
                    probable_causes=causes,
                    suggested_actions=actions,
                    hypothesis=causes[0].message if causes else "Special tokens differ.",
                    next_test=actions[0] if actions else "Align special tokens.",
                    summary="First divergence is in special tokens.",
                    warnings=warnings,
                )

        elif layer == "tokens":
            left_ids = a.data.get("ids") or a.data.get("token_ids") or []
            right_ids = b.data.get("ids") or b.data.get("token_ids") or []
            result = compare_tokens(
                left_ids,
                right_ids,
                left_strings=a.data.get("token_strings"),
                right_strings=b.data.get("token_strings"),
                left_special=a.data,
                right_special=b.data,
            )
            if result.result == ParityResult.DIVERGENT:
                causes = diagnose_tokens(a.data, b.data, result.details)
                actions = actions_for(causes, layer)
                idx = result.details.get("first_difference")
                percent = float(result.details.get("downstream_percent", 0.0))
                return Diagnosis(
                    status=ParityResult.DIVERGENT,
                    first_divergence=layer,
                    first_divergence_detail=FirstDivergence(
                        layer=layer,
                        location=DivergenceLocation(token_index=idx),
                        baseline=str(result.details.get("expected_token_id")),
                        candidate=str(result.details.get("received_token_id")),
                    ),
                    propagation=PropagationInfo(
                        first_token_difference=idx if isinstance(idx, int) else None,
                        different_tokens_percent=percent,
                        downstream_different=result.details.get("downstream_different"),
                    ),
                    propagation_percent=percent,
                    probable_causes=causes,
                    suggested_actions=actions,
                    hypothesis=causes[0].message if causes else "Token divergence.",
                    next_test=actions[0] if actions else "Compare tokenizers.",
                    summary=f"First divergence is in tokens at index {idx}.",
                    warnings=warnings,
                )

        elif layer == "generation":
            left_ids = list(a.data.get("ids") or a.data.get("token_ids") or [])
            right_ids = list(b.data.get("ids") or b.data.get("token_ids") or [])
            left_text = str(a.data.get("text") or "")
            right_text = str(b.data.get("text") or "")
            # Prefer text when either side lacks generated token ids (OpenAI-compat APIs).
            if left_ids and right_ids:
                result = compare_tokens(left_ids, right_ids)
                divergent = result.result == ParityResult.DIVERGENT
                # Soft: identical decoded text overrides id-only drift
                if divergent and left_text and left_text == right_text:
                    divergent = False
                    warnings.append("Generation token ids differ but decoded text matches.")
            else:
                result = compare_prompt(left_text, right_text)
                divergent = result.result == ParityResult.DIVERGENT

            def _norm_stop(value: object) -> str:
                raw = str(value or "").strip().lower()
                if raw in {"eos", "stop", "end_turn", "end", "eos_token"}:
                    return "stop"
                return raw

            left_stop = a.data.get("stop_reason") or a.data.get("finish_reason")
            right_stop = b.data.get("stop_reason") or b.data.get("finish_reason")
            stop_divergent = _norm_stop(left_stop) != _norm_stop(right_stop)
            if stop_divergent and left_text and left_text == right_text:
                # Synonym / naming drift only — do not fail causal walk
                warnings.append(
                    f"finish_reason naming differs ({left_stop!r} vs {right_stop!r}) but generated text matches."
                )
                stop_divergent = False

            if divergent or stop_divergent:
                causes = diagnose_generation(a.data, b.data)
                actions = actions_for(causes, layer)
                return Diagnosis(
                    status=ParityResult.DIVERGENT,
                    first_divergence=layer,
                    first_divergence_detail=FirstDivergence(layer=layer),
                    probable_causes=causes,
                    suggested_actions=actions,
                    hypothesis=causes[0].message if causes else "Generation differs.",
                    next_test=actions[0] if actions else f"Isolate {layer}.",
                    summary=f"First divergence is in {layer}.",
                    warnings=warnings,
                )

        elif a.data != b.data:
            return Diagnosis(
                status=ParityResult.DIVERGENT,
                first_divergence=layer,
                first_divergence_detail=FirstDivergence(layer=layer),
                probable_causes=[
                    ProbableCause(
                        code=f"{layer.upper()}_DIVERGENT",
                        confidence=0.7,
                        message=f"Layer {layer} differs between backends.",
                    )
                ],
                suggested_actions=[f"eleanity compare --observe {layer} --backends {left.backend},{right.backend}"],
                hypothesis=f"Layer {layer} differs between backends.",
                next_test=f"Isolate {layer} with the same artifact.",
                summary=f"First divergence is in {layer}.",
                warnings=warnings,
            )

    # Check if anything was comparable
    comparable = False
    for layer in ORDER:
        a, b = left.layers.get(layer), right.layers.get(layer)
        if a and b and a.state == LayerState.OBSERVED and b.state == LayerState.OBSERVED:
            comparable = True
            break
    if not comparable:
        return Diagnosis(
            status=ParityResult.INCONCLUSIVE,
            hypothesis="No comparable layer was observed on both backends.",
            next_test="Install extras or configure endpoints to expose template/tokens.",
            summary="No mutually observable layers — verdict is INCONCLUSIVE.",
            suggested_actions=[
                "uv sync --extra transformers",
                "export ELEANITY_VLLM_URL=http://127.0.0.1:8000",
                "eleanity doctor --check-backends",
            ],
            warnings=warnings,
        )

    return Diagnosis(
        status=ParityResult.PASS,
        hypothesis="No observable divergence between traces.",
        next_test="Expand observe layers if deeper parity is required.",
        summary="No divergence found on mutually comparable layers.",
        suggested_actions=["eleanity compare --observe artifact,template,tokens,generation --format text"],
        warnings=warnings,
    )
