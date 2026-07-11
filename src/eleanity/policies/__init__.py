"""Parity policy helpers and engine."""

from eleanity.models.schemas import DEFAULT_TOLERANCE, ParityProfile

__all__ = [
    "DEFAULT_TOLERANCE",
    "ParityProfile",
    "describe_policy",
    "policy_rules",
    "PolicyEngine",
]


def describe_policy(profile: ParityProfile) -> str:
    descriptions = {
        ParityProfile.STRICT: (
            "Exact byte-identical prompts, identical token IDs and special tokens; "
            "deterministic generation; logits compared with tiny tolerance when available."
        ),
        ParityProfile.QUANTIZED: (
            "Identical prompt and input tokens; small numeric drift allowed; "
            "compare top-k ranking and distribution; soft on dtype/quantization flags."
        ),
        ParityProfile.FUNCTIONAL: (
            "Does not require equal logits/tokens; validates JSON/schema/tool names/"
            "args/stop reason/functional behavior."
        ),
        ParityProfile.API_CONFORMANCE: (
            "HTTP status, usage, finish_reason, streaming order, OpenAI-compatible shape."
        ),
    }
    return descriptions[profile]


def policy_rules(profile: ParityProfile) -> dict[str, object]:
    return {
        ParityProfile.STRICT: {
            "require_identical_prompt_bytes": True,
            "require_identical_token_ids": True,
            "require_identical_special_tokens": True,
            "require_deterministic_generation": True,
            "require_logits": True,
            "logits_tolerance": DEFAULT_TOLERANCE[ParityProfile.STRICT],
        },
        ParityProfile.QUANTIZED: {
            "require_identical_prompt_bytes": True,
            "require_identical_token_ids": True,
            "require_identical_special_tokens": True,
            "require_deterministic_generation": False,
            "require_logits": True,
            "logits_tolerance": DEFAULT_TOLERANCE[ParityProfile.QUANTIZED],
            "compare_topk_ranking": True,
        },
        ParityProfile.FUNCTIONAL: {
            "require_identical_prompt_bytes": False,
            "require_identical_token_ids": False,
            "require_logits": False,
            "require_deterministic_generation": False,
            "validate_json": True,
            "validate_schema": True,
            "validate_tool_name": True,
            "validate_arguments": True,
            "validate_stop_reason": True,
            "logits_tolerance": DEFAULT_TOLERANCE[ParityProfile.FUNCTIONAL],
        },
        ParityProfile.API_CONFORMANCE: {
            "require_identical_prompt_bytes": False,
            "require_identical_token_ids": False,
            "require_logits": False,
            "http_status": True,
            "finish_reason": True,
            "usage": True,
            "streaming": True,
            "openai_compatible_shape": True,
            "logits_tolerance": DEFAULT_TOLERANCE[ParityProfile.API_CONFORMANCE],
        },
    }[profile]


def __getattr__(name: str):
    if name == "PolicyEngine":
        from eleanity.policies.engine import PolicyEngine

        return PolicyEngine
    raise AttributeError(name)
