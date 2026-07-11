"""Privacy and security controls for runs and traces."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*['\"]?([^\s'\"]+)"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT-ish
]


@dataclass
class PrivacyPolicy:
    no_store: bool = False
    redact_input: bool = False
    redact_output: bool = False
    hash_content: bool = False
    allow_remote: bool = False
    secrets_from_env: bool = True
    retention: str | None = None  # e.g. "24h", "7d"

    def to_dict(self) -> dict[str, Any]:
        return {
            "no_store": self.no_store,
            "redact_input": self.redact_input,
            "redact_output": self.redact_output,
            "hash_content": self.hash_content,
            "allow_remote": self.allow_remote,
            "secrets_from_env": self.secrets_from_env,
            "retention": self.retention,
            "content_left_machine": False,
        }


def parse_retention(value: str | None) -> timedelta | None:
    if not value:
        return None
    value = value.strip().lower()
    if value.endswith("h"):
        return timedelta(hours=int(value[:-1]))
    if value.endswith("d"):
        return timedelta(days=int(value[:-1]))
    if value.endswith("m"):
        return timedelta(minutes=int(value[:-1]))
    raise ValueError(f"unsupported retention format: {value} (use 24h, 7d, 30m)")


def scrub_secrets(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(
            lambda m: m.group(0).split(m.group(m.lastindex or 0))[0] if False else "[SECRET_REDACTED]", out
        )
        out = pattern.sub("[SECRET_REDACTED]", out)
    return out


def scrub_obj(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_secrets(value)
    if isinstance(value, list):
        return [scrub_obj(v) for v in value]
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            key_l = str(k).lower()
            if any(s in key_l for s in ("api_key", "apikey", "authorization", "secret", "password", "token")):
                if key_l in {"tokenizer", "token_ids", "tokens", "special_tokens", "token_strings", "token_index"}:
                    redacted[k] = scrub_obj(v)
                else:
                    redacted[k] = "[SECRET_REDACTED]"
            else:
                redacted[k] = scrub_obj(v)
        return redacted
    return value


def apply_privacy_to_payload(payload: dict[str, Any], policy: PrivacyPolicy) -> dict[str, Any]:
    data = scrub_obj(payload)
    if policy.redact_input or policy.redact_output:
        for trace in data.get("traces") or []:
            layers = trace.get("layers") or {}
            if policy.redact_input:
                for name in ("template", "tokens", "rendered_prompt", "input_token_ids"):
                    if name in layers and isinstance(layers[name], dict):
                        d = layers[name].get("data") or {}
                        for key in ("text", "rendered_text", "token_ids", "ids", "token_strings", "decoded_text"):
                            if key in d:
                                d[key] = None if not policy.hash_content else d[key]
                        d["content_redacted"] = True
                        layers[name]["data"] = d
                        layers[name]["state"] = layers[name].get("state") or "REDACTED"
            if policy.redact_output:
                for name in ("generation", "generated_token_ids", "detokenization", "structured", "api"):
                    if name in layers and isinstance(layers[name], dict):
                        d = layers[name].get("data") or {}
                        for key in ("text", "ids", "token_ids", "content", "arguments"):
                            if key in d:
                                d[key] = None
                        d["content_redacted"] = True
                        layers[name]["data"] = d
    data["privacy"] = policy.to_dict()
    data["redacted"] = bool(policy.redact_input or policy.redact_output)
    return data


def enforce_no_remote(policy: PrivacyPolicy) -> None:
    if policy.allow_remote:
        return
    # Block accidental remote sinks via env
    if os.environ.get("ELEANITY_FORCE_REMOTE_UPLOAD") == "1":
        raise RuntimeError("Remote upload blocked: pass --allow-remote to override")


def apply_retention(runs_dir: Path, retention: str | None) -> int:
    """Delete run directories older than retention. Returns number removed."""

    delta = parse_retention(retention)
    if delta is None:
        return 0
    cutoff = datetime.now(UTC) - delta
    removed = 0
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return 0
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def privacy_from_flags(
    *,
    no_store: bool = False,
    redact_input: bool = False,
    redact_output: bool = False,
    hash_content: bool = False,
    allow_remote: bool = False,
    retention: str | None = None,
    redact_prompts: bool = False,
) -> PrivacyPolicy:
    return PrivacyPolicy(
        no_store=no_store,
        redact_input=redact_input or redact_prompts,
        redact_output=redact_output,
        hash_content=hash_content,
        allow_remote=allow_remote,
        retention=retention,
    )
