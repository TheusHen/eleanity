from __future__ import annotations

from typing import Any

from eleanity.models.schemas import ProbableCause


def _cause(
    code: str,
    confidence: float,
    message: str,
    *,
    evidence: dict[str, Any] | None = None,
    affected_layers: list[str] | None = None,
    suggested_remediation: str | None = None,
) -> ProbableCause:
    return ProbableCause(
        code=code,
        confidence=confidence,
        message=message,
        evidence=evidence or {},
        affected_layers=affected_layers or [],
        suggested_remediation=suggested_remediation,
    )


def diagnose_template(
    left: dict[str, Any],
    right: dict[str, Any],
    prompt_details: dict[str, Any],
) -> list[ProbableCause]:
    causes: list[ProbableCause] = []
    left_text = left.get("text") or left.get("rendered_text") or ""
    right_text = right.get("text") or right.get("rendered_text") or ""

    left_hash = left.get("chat_template_hash") or left.get("template_hash")
    right_hash = right.get("chat_template_hash") or right.get("template_hash")
    if not left_text and right_text:
        causes.append(_cause("CHAT_TEMPLATE_MISSING_BASELINE", 0.9, "Baseline has empty rendered template."))
    if left_text and not right_text:
        causes.append(_cause("CHAT_TEMPLATE_MISSING_CANDIDATE", 0.95, "Candidate has empty rendered template."))

    if left_hash and right_hash and left_hash != right_hash:
        causes.append(
            _cause(
                "CHAT_TEMPLATE_DIFFERENT",
                0.92,
                "Chat template hash differs between backends.",
            )
        )

    left_agp = left.get("add_generation_prompt")
    right_agp = right.get("add_generation_prompt")
    if left_agp is not None and right_agp is not None and left_agp != right_agp:
        causes.append(
            _cause(
                "CHAT_TEMPLATE_GENERATION_PROMPT_MISMATCH",
                0.98,
                f"add_generation_prompt differs: baseline={left_agp} candidate={right_agp}.",
                evidence={
                    "baseline_add_generation_prompt": left_agp,
                    "candidate_add_generation_prompt": right_agp,
                    "baseline_rendered_suffix": (left_text or "")[-48:],
                    "candidate_rendered_suffix": (right_text or "")[-48:],
                },
                affected_layers=["chat_template", "rendered_prompt", "input_token_ids", "generation"],
                suggested_remediation=("Set add_generation_prompt=true or configure --chat-template explicitly."),
            )
        )

    if prompt_details.get("missing_assistant_turn"):
        causes.append(
            _cause(
                "MISSING_ASSISTANT_TURN_TOKEN",
                0.94,
                "The candidate backend did not add the assistant turn start token.",
                evidence={
                    "baseline_suffix": (left_text or "")[-40:],
                    "candidate_suffix": (right_text or "")[-40:],
                    "flag": "missing_assistant_turn",
                },
                affected_layers=["chat_template", "rendered_prompt", "input_token_ids", "generation"],
                suggested_remediation=("Set add_generation_prompt=true or configure --chat-template explicitly."),
            )
        )

    if prompt_details.get("newline_difference"):
        causes.append(_cause("NEWLINE_DIVERGENT", 0.8, "Newlines differ (CRLF/LF or line count)."))
    if prompt_details.get("whitespace_only_difference"):
        causes.append(_cause("WHITESPACE_DIVERGENT", 0.78, "Only whitespace differs between prompts."))
    if prompt_details.get("unicode_normalization_difference"):
        causes.append(
            _cause(
                "UNICODE_NORMALIZATION",
                0.85,
                "Strings match under Unicode NFC but not in original form.",
            )
        )
    missing = prompt_details.get("missing_markers") or []
    if missing:
        causes.append(
            _cause(
                "SPECIAL_MARKERS_MISSING",
                0.88,
                f"Candidate missing markers: {', '.join(missing)}.",
            )
        )
    if not causes and left_text != right_text:
        causes.append(
            _cause(
                "CHAT_TEMPLATE_DIFFERENT",
                0.7,
                "Rendered chat template text differs between backends.",
            )
        )
    return causes


def diagnose_tokens(
    left: dict[str, Any],
    right: dict[str, Any],
    token_details: dict[str, Any],
) -> list[ProbableCause]:
    causes: list[ProbableCause] = []
    special = token_details.get("special_token_differences") or {}
    if "bos_token_id" in special:
        causes.append(_cause("BOS_DIFFERENT", 0.9, "BOS token id differs."))
    if "eos_token_id" in special:
        causes.append(_cause("EOS_DIFFERENT", 0.9, "EOS token id differs."))
    if special.get("pad_as_eos_candidate") or special.get("pad_as_eos_baseline"):
        causes.append(_cause("PAD_USED_AS_EOS", 0.86, "PAD is used as EOS on one backend."))
    if token_details.get("truncation_difference"):
        causes.append(_cause("TRUNCATION_DIFFERENT", 0.87, "Truncation behavior differs between backends."))
    if left.get("padding_side") and right.get("padding_side"):
        if left.get("padding_side") != right.get("padding_side"):
            causes.append(_cause("PADDING_SIDE_DIFFERENT", 0.84, "padding_side differs."))
    # Heuristic: duplicate BOS at start
    left_ids = left.get("ids") or left.get("token_ids") or []
    right_ids = right.get("ids") or right.get("token_ids") or []
    bos = left.get("bos_token_id")
    if bos is not None and len(left_ids) >= 2 and left_ids[0] == bos and left_ids[1] == bos:
        causes.append(_cause("BOS_DUPLICATED_BASELINE", 0.8, "Baseline sequence starts with duplicated BOS."))
    if bos is not None and len(right_ids) >= 2 and right_ids[0] == bos and right_ids[1] == bos:
        causes.append(_cause("BOS_DUPLICATED_CANDIDATE", 0.8, "Candidate sequence starts with duplicated BOS."))
    if not causes:
        causes.append(
            _cause(
                "TOKENIZER_OR_NORMALIZATION",
                0.72,
                "Likely tokenizer, Unicode normalization, or special-token difference.",
            )
        )
    return causes


def diagnose_artifact(left: dict[str, Any], right: dict[str, Any]) -> list[ProbableCause]:
    causes: list[ProbableCause] = []
    if left.get("revision") != right.get("revision"):
        causes.append(_cause("REVISION_DIFFERENT", 0.95, "Model revision differs."))
    if left.get("tokenizer_hash") and right.get("tokenizer_hash"):
        if left.get("tokenizer_hash") != right.get("tokenizer_hash"):
            causes.append(_cause("TOKENIZER_FILES_DIFFERENT", 0.93, "Tokenizer hash differs."))
    if left.get("chat_template_hash") != right.get("chat_template_hash"):
        if left.get("chat_template_hash") and right.get("chat_template_hash"):
            causes.append(_cause("CHAT_TEMPLATE_DIFFERENT", 0.9, "Embedded chat template hash differs."))
    quant_l, quant_r = left.get("quantization"), right.get("quantization")
    if bool(quant_l) != bool(quant_r) or (quant_l and quant_r and quant_l != quant_r):
        causes.append(
            _cause(
                "QUANTIZED_VS_FULL_PRECISION",
                0.91,
                f"Quantization differs: baseline={quant_l!r} candidate={quant_r!r}.",
            )
        )
    if left.get("lora_adapters") != right.get("lora_adapters"):
        causes.append(_cause("LORA_ABSENT_OR_DIFFERENT", 0.88, "LoRA / adapter set differs."))
    if not causes:
        causes.append(
            _cause(
                "ARTIFACT_MISMATCH",
                0.75,
                "Model, tokenizer, template, or quantization do not match.",
            )
        )
    return causes


def diagnose_generation(left: dict[str, Any], right: dict[str, Any]) -> list[ProbableCause]:
    causes: list[ProbableCause] = []
    if left.get("stop_reason") != right.get("stop_reason"):
        causes.append(
            _cause(
                "FINISH_REASON_DIFFERENT",
                0.85,
                f"stop/finish reason differs: {left.get('stop_reason')!r} vs {right.get('stop_reason')!r}.",
            )
        )
    if left.get("seed") != right.get("seed") and (left.get("seed") is not None or right.get("seed") is not None):
        causes.append(_cause("SEED_NOT_APPLIED", 0.8, "Seed differs or was not applied on one backend."))
    if not causes:
        causes.append(_cause("GENERATION_DIVERGENT", 0.7, "Generated token sequence differs."))
    return causes


def actions_for(causes: list[ProbableCause], layer: str) -> list[str]:
    actions: list[str] = []
    codes = {c.code for c in causes}
    if (
        "MISSING_ASSISTANT_TURN_TOKEN" in codes
        or "ADD_GENERATION_PROMPT_DIVERGENT" in codes
        or "CHAT_TEMPLATE_GENERATION_PROMPT_MISMATCH" in codes
    ):
        actions.append("eleanity compare --observe template,tokens --policy strict --format text")
        actions.append("Set generation.add_generation_prompt=true on both backends.")
        actions.append("Diff tokenizer chat_template / tokenizer_config.json.")
    if "CHAT_TEMPLATE_DIFFERENT" in codes or "CHAT_TEMPLATE_MISSING_CANDIDATE" in codes:
        actions.append("eleanity compare --tokenizer-only --observe template,tokens")
        actions.append("Pin the same tokenizer revision on both backends.")
    if "TOKENIZER_FILES_DIFFERENT" in codes or "TOKENIZER_OR_NORMALIZATION" in codes:
        actions.append("eleanity inspect MODEL --backend transformers --format json")
        actions.append("Compare tokenizer.json and add_special_tokens flags.")
    if "QUANTIZED_VS_FULL_PRECISION" in codes:
        actions.append("eleanity compare --policy quantized --format text")
        actions.append("Align quantization or accept PASS_WITH_TOLERANCE under quantized policy.")
    if "REVISION_DIFFERENT" in codes:
        actions.append("Pin the same model revision / commit SHA on both backends.")
    if "FINISH_REASON_DIFFERENT" in codes:
        actions.append("Align stop tokens and max_tokens across backends.")
    if "SEED_NOT_APPLIED" in codes:
        actions.append("Use temperature=0 and set seed on both backends.")
    if not actions:
        if layer == "artifact":
            actions.append("eleanity inspect MODEL --format json")
        elif layer == "template":
            actions.append("Compare apply_chat_template with identical add_generation_prompt.")
        elif layer == "tokens":
            actions.append("Compare tokenizer.json and flags add_special_tokens.")
        else:
            actions.append(f"eleanity compare --observe {layer} --format text")
    return actions
