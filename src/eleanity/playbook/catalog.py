from __future__ import annotations

from typing import Any

PLAYBOOK: dict[str, dict[str, Any]] = {
    "MISSING_ASSISTANT_TURN_TOKEN": {
        "title": "Missing assistant turn marker",
        "summary": "Candidate did not emit the assistant-start special token in the rendered prompt.",
        "actions": [
            "Compare add_generation_prompt flags between backends.",
            "Inspect tokenizer.chat_template / tokenizer_config.json.",
            "For GGUF, re-convert ensuring chat template metadata is preserved.",
        ],
        "files": ["tokenizer_config.json", "chat_template.jinja"],
    },
    "ADD_GENERATION_PROMPT_DIVERGENT": {
        "title": "add_generation_prompt mismatch",
        "summary": "Backends disagree on whether to append the assistant generation prefix.",
        "actions": [
            "Force identical generation.add_generation_prompt in the scenario.",
            "Check runtime server chat template defaults.",
        ],
        "files": ["eleanity scenario generation block"],
    },
    "CHAT_TEMPLATE_DIFFERENT": {
        "title": "Chat template differs",
        "summary": "Rendered template hash or body differs across backends.",
        "actions": [
            "Diff chat templates side by side.",
            "Pin the same tokenizer revision on every backend.",
        ],
        "files": ["tokenizer_config.json"],
    },
    "CHAT_TEMPLATE_MISSING_CANDIDATE": {
        "title": "Chat template missing on candidate",
        "summary": "Candidate produced an empty rendered template.",
        "actions": [
            "Verify the model tokenizer is loaded on the candidate runtime.",
            "Confirm GGUF includes template metadata.",
        ],
        "files": ["tokenizer_config.json"],
    },
    "TOKENIZER_FILES_DIFFERENT": {
        "title": "Tokenizer files differ",
        "summary": "Tokenizer hash or vocab identity diverges.",
        "actions": [
            "Compare tokenizer.json / merges / vocab files.",
            "Avoid mixing remote HF tokenizer with local GGUF vocab silently.",
        ],
        "files": ["tokenizer.json", "vocab.json", "merges.txt"],
    },
    "TOKENIZER_OR_NORMALIZATION": {
        "title": "Tokenizer or Unicode normalization",
        "summary": "Token IDs diverge after an identical or near-identical prompt.",
        "actions": [
            "Check NFC vs NFD input normalization.",
            "Align add_special_tokens flags.",
        ],
        "files": ["tokenizer.json"],
    },
    "BOS_DIFFERENT": {
        "title": "BOS token differs",
        "summary": "Beginning-of-sequence token id mapping differs.",
        "actions": ["Align bos_token_id in tokenizer configs."],
        "files": ["tokenizer_config.json"],
    },
    "EOS_DIFFERENT": {
        "title": "EOS token differs",
        "summary": "End-of-sequence token id mapping differs.",
        "actions": ["Align eos_token_id / stop tokens across runtimes."],
        "files": ["tokenizer_config.json", "generation_config.json"],
    },
    "PAD_USED_AS_EOS": {
        "title": "PAD used as EOS",
        "summary": "One backend maps pad_token_id to eos_token_id.",
        "actions": ["Set explicit pad_token and eos_token consistently."],
        "files": ["tokenizer_config.json"],
    },
    "QUANTIZED_VS_FULL_PRECISION": {
        "title": "Quantized vs full precision",
        "summary": "Artifact quantization flags differ.",
        "actions": [
            "Use parity policy quantized if intentional.",
            "Or align GGUF/AWQ/GPTQ vs BF16/FP16 artifacts.",
        ],
        "files": ["model card", "quant config"],
    },
    "REVISION_DIFFERENT": {
        "title": "Model revision differs",
        "summary": "Checkpoint revision / commit is not pinned identically.",
        "actions": ["Pin model.revision to the same commit SHA everywhere."],
        "files": ["eleanity.yaml", "scenario model block"],
    },
    "FINISH_REASON_DIFFERENT": {
        "title": "finish/stop reason differs",
        "summary": "Generation stopped for different reasons.",
        "actions": ["Align stop sequences, max_tokens, and EOS handling."],
        "files": ["scenario parameters"],
    },
    "SEED_NOT_APPLIED": {
        "title": "Seed not applied",
        "summary": "Seed differs or a backend ignored deterministic seeding.",
        "actions": ["Verify temperature=0 and seed support on the candidate runtime."],
        "files": ["scenario parameters"],
    },
    "UNICODE_NORMALIZATION": {
        "title": "Unicode normalization difference",
        "summary": "Strings match under NFC but not in original form.",
        "actions": ["Normalize inputs to NFC before tokenization on all backends."],
        "files": ["preprocessing pipeline"],
    },
    "NEWLINE_DIVERGENT": {
        "title": "Newline divergence",
        "summary": "CRLF/LF or line count differs in rendered prompts.",
        "actions": ["Normalize newlines to LF before apply_chat_template."],
        "files": ["chat template", "message content"],
    },
    "WHITESPACE_DIVERGENT": {
        "title": "Whitespace-only divergence",
        "summary": "Only whitespace differs between rendered prompts.",
        "actions": ["Check trailing spaces and indentation in chat templates."],
        "files": ["chat template"],
    },
}


def get_playbook_entry(code: str) -> dict[str, Any] | None:
    return PLAYBOOK.get(code)


def render_playbook_markdown(code: str) -> str:
    entry = get_playbook_entry(code)
    if not entry:
        return f"# {code}\n\nNo playbook entry yet.\n"
    lines = [
        f"# {entry['title']}",
        "",
        f"**Code:** `{code}`",
        "",
        entry["summary"],
        "",
        "## Suggested actions",
        "",
    ]
    for action in entry.get("actions") or []:
        lines.append(f"- {action}")
    files = entry.get("files") or []
    if files:
        lines.extend(["", "## Related files", ""])
        for item in files:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)
