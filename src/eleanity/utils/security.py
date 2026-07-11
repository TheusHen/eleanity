from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|private[_-]?key|secret|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|^token$|bearer)",
    re.IGNORECASE,
)
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|private[_-]?key|secret|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token)\b(\s*[:=]\s*)([^\s,;]+)"
)
BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
HOME_PATH = re.compile(r"(?i)([A-Z]:\\Users\\[^\\/]+|\/home\/[^\/]+|\/Users\/[^\/]+)")


def redact_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    rendered = SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted]", rendered)
    rendered = BEARER_TOKEN.sub("Bearer [redacted]", rendered)
    return HOME_PATH.sub(lambda m: m.group(0).rsplit("\\", 1)[0].rsplit("/", 1)[0] + "/…", rendered)


def redact_mapping(value: Any, key: str = "") -> Any:
    if key and SENSITIVE_KEY.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): redact_mapping(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def sanitize_path(path: str | Path | None, *, keep_name: bool = True) -> str | None:
    """Avoid leaking full private filesystem paths into shareable reports."""

    if path is None:
        return None
    p = Path(path)
    if keep_name:
        return f"…/{p.name}"
    return "…/[path]"


def truncate_log(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n…[truncated]…"
