"""Capture production OpenAI-style traffic into redacted Eleanity scenarios."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _redact_content(content: Any, *, hash_content: bool) -> Any:
    if content is None:
        return None
    if isinstance(content, str):
        if hash_content:
            return f"[redacted:sha256:{_hash_text(content)} len={len(content)}]"
        return "[redacted]"
    if isinstance(content, list):
        return [_redact_content(item, hash_content=hash_content) for item in content]
    if isinstance(content, dict):
        out = dict(content)
        if "text" in out:
            out["text"] = _redact_content(out["text"], hash_content=hash_content)
        if "content" in out:
            out["content"] = _redact_content(out["content"], hash_content=hash_content)
        return out
    return "[redacted]"


def _messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    # OpenAI chat completion request shapes
    body = record.get("request") or record.get("body") or record
    messages = body.get("messages") or record.get("messages") or []
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, list):
            # multimodal — keep text parts only as placeholder
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif isinstance(part, dict) and part.get("type") in {"image_url", "image"}:
                    texts.append("[image]")
                elif isinstance(part, dict) and part.get("type") == "input_audio":
                    texts.append("[audio]")
            content = "\n".join(texts)
        out.append({"role": role, "content": str(content or "")})
    if not out:
        prompt = body.get("prompt") or record.get("prompt")
        if prompt:
            out = [{"role": "user", "content": str(prompt)}]
    return out


def _params_from_record(record: dict[str, Any]) -> dict[str, Any]:
    body = record.get("request") or record.get("body") or record
    keys = (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "response_format",
        "tools",
        "tool_choice",
    )
    params: dict[str, Any] = {}
    for key in keys:
        if key in body and body[key] is not None:
            params[key if key != "max_completion_tokens" else "max_tokens"] = body[key]
    return params


def capture_openai_jsonl(
    source: Path,
    output_dir: Path,
    *,
    redact: bool = True,
    hash_content: bool = True,
    sample: int | None = None,
    suite_name: str = "production-suite",
) -> dict[str, Any]:
    """Convert OpenAI request/response JSONL into a scenario suite directory."""

    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict[str, Any]] = []
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = _messages_from_record(record)
            if not messages:
                continue
            if redact:
                messages = [
                    {
                        "role": m["role"],
                        "content": _redact_content(m["content"], hash_content=hash_content),
                    }
                    for m in messages
                ]
            body = record.get("request") or record.get("body") or record
            model = body.get("model") or record.get("model") or "production-model"
            scenario = {
                "schema_version": "0.1",
                "name": f"{suite_name}-{count:04d}",
                "description": f"Captured from {source.name}:{line_no}",
                "model": {"id": model, "tokenizer_only": False},
                "messages": messages,
                "parameters": _params_from_record(record),
                "observe": ["artifact", "template", "tokens", "generation", "api"],
                "parity_policy": "functional",
                "redact_prompts": True,
                "metadata": {
                    "captured_from": str(source),
                    "line": line_no,
                    "redacted": redact,
                },
            }
            scenarios.append(scenario)
            count += 1
            if sample is not None and count >= sample:
                break

    suite_path = output_dir / "scenarios.yaml"
    # Write as multi-document YAML-ish via JSON lines of YAML using PyYAML
    import yaml

    suite_path.write_text(
        yaml.safe_dump_all(scenarios, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "suite": suite_name,
        "source": str(source),
        "scenarios": len(scenarios),
        "redacted": redact,
        "hash_content": hash_content,
        "path": str(suite_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
