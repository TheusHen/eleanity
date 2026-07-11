"""Granular causal layer taxonomy (Trace Spec v1) with legacy aliases."""

from __future__ import annotations

# Canonical ordered layers for first-divergence walks (fine-grained).
LAYER_ORDER_V1: tuple[str, ...] = (
    "artifact",
    "model_config",
    "tokenizer_artifact",
    "chat_template",
    "rendered_prompt",
    "normalization",
    "special_tokens",
    "input_token_ids",
    "generation_config",
    "prefill_logits",
    "decode_logits",
    "logits_processing",
    "sampling",
    "generated_token_ids",
    "stop_decision",
    "detokenization",
    "response_mapping",
    "streaming",
    "usage_accounting",
    "structured_output",
    "tool_call_parsing",
    # Multimodal / future-ready (always optional)
    "multimodal_inputs",
    "embeddings",
    "reasoning_content",
    "speculative_decoding",
)

CANONICAL_LAYERS = frozenset(LAYER_ORDER_V1)

# Legacy CLI/scenario names → canonical (or coarse bundles).
LAYER_ALIASES: dict[str, str | list[str]] = {
    "template": "rendered_prompt",
    "tokens": "input_token_ids",
    "logits": "prefill_logits",
    "forward": "prefill_logits",
    "generation": ["generated_token_ids", "stop_decision", "detokenization"],
    "stop_reason": "stop_decision",
    "structured": "structured_output",
    "api": "response_mapping",
    "rendered_prompt": "rendered_prompt",
}

# Coarse layer used by existing adapters (backward compatible observe pipeline).
COARSE_LAYERS = frozenset(
    {
        "artifact",
        "template",
        "rendered_prompt",
        "tokens",
        "special_tokens",
        "logits",
        "forward",
        "generation",
        "stop_reason",
        "structured",
        "streaming",
        "api",
    }
)

# Map fine layer → coarse observe target for adapters that only implement coarse APIs.
FINE_TO_COARSE: dict[str, str] = {
    "artifact": "artifact",
    "model_config": "artifact",
    "tokenizer_artifact": "artifact",
    "chat_template": "template",
    "rendered_prompt": "template",
    "normalization": "template",
    "special_tokens": "special_tokens",
    "input_token_ids": "tokens",
    "generation_config": "generation",
    "prefill_logits": "logits",
    "decode_logits": "logits",
    "logits_processing": "logits",
    "sampling": "generation",
    "generated_token_ids": "generation",
    "stop_decision": "generation",
    "detokenization": "generation",
    "response_mapping": "api",
    "streaming": "streaming",
    "usage_accounting": "api",
    "structured_output": "structured",
    "tool_call_parsing": "structured",
    "multimodal_inputs": "api",
    "embeddings": "api",
    "reasoning_content": "generation",
    "speculative_decoding": "generation",
}


def canonicalize_layer(name: str) -> str:
    """Return a single canonical layer name (first of expansion for bundles)."""

    key = name.strip().lower()
    mapped = LAYER_ALIASES.get(key, key)
    if isinstance(mapped, list):
        return mapped[0]
    return mapped


def expand_observe_layers(layers: list[str]) -> list[str]:
    """Expand aliases/bundles to a unique ordered list of fine layers."""

    seen: list[str] = []
    for raw in layers:
        key = raw.strip().lower()
        mapped = LAYER_ALIASES.get(key, key)
        items = mapped if isinstance(mapped, list) else [mapped]
        for item in items:
            if item not in seen:
                seen.append(item)
    # stable sort by LAYER_ORDER_V1 when known
    order = {name: i for i, name in enumerate(LAYER_ORDER_V1)}
    known = [x for x in seen if x in order]
    unknown = [x for x in seen if x not in order]
    known.sort(key=lambda x: order[x])
    return known + unknown


def to_coarse_observe(layers: list[str]) -> list[str]:
    """Convert fine/legacy observe list into coarse adapter observe set."""

    coarse: list[str] = []
    for raw in layers:
        key = raw.strip().lower()
        if key in COARSE_LAYERS:
            mapped = key
            if mapped == "rendered_prompt":
                mapped = "template"
            if mapped == "forward":
                mapped = "logits"
            if mapped == "stop_reason":
                mapped = "generation"
            if mapped not in coarse:
                coarse.append(mapped)
            continue
        fine = canonicalize_layer(key)
        c = FINE_TO_COARSE.get(fine, fine)
        allowed_coarse = {
            "artifact",
            "template",
            "tokens",
            "special_tokens",
            "logits",
            "generation",
            "structured",
            "streaming",
            "api",
        }
        if c not in coarse and c in allowed_coarse:
            coarse.append(c)
    return coarse


def layer_description(name: str) -> str:
    docs = {
        "artifact": "Model weights/tokenizer identity hashes",
        "model_config": "config.json fields affecting decoding",
        "tokenizer_artifact": "tokenizer files / vocab hash",
        "chat_template": "Jinja chat template source",
        "rendered_prompt": "Fully rendered prompt bytes/text",
        "normalization": "Unicode/normalization applied before tokenize",
        "special_tokens": "BOS/EOS/pad/unk and special maps",
        "input_token_ids": "Tokenized prompt ids",
        "generation_config": "temperature, top_p, seed, stops, …",
        "prefill_logits": "Logits after prompt prefill",
        "decode_logits": "Per-step decode logits",
        "logits_processing": "repetition_penalty, logit_bias, processors",
        "sampling": "Sampler decisions (argmax/sample path)",
        "generated_token_ids": "Output token id sequence",
        "stop_decision": "EOS / stop string / length stop",
        "detokenization": "Decoded text from ids",
        "response_mapping": "OpenAI-style response fields",
        "streaming": "SSE/chunk order and deltas",
        "usage_accounting": "prompt/completion token counts",
        "structured_output": "JSON schema / grammar constrained output",
        "tool_call_parsing": "Tool/function call parse tree",
        "multimodal_inputs": "Image/audio/video inputs",
        "embeddings": "Embedding vectors",
        "reasoning_content": "Chain-of-thought / reasoning channel",
        "speculative_decoding": "Draft/verify speculative path",
    }
    return docs.get(name, name)
