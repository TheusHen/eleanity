"""Public model×runtime compatibility matrix helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eleanity.adapters import adapter_for
from eleanity.certification import certify_runtime

FEATURE_COLUMNS = (
    "tokens",
    "template",
    "logits",
    "generation",
    "tool_calls",
    "json",
    "streaming",
    "multimodal",
)


def empty_row(model: str, runtime: str, version: str | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "runtime": runtime,
        "version": version,
        "features": {f: "unknown" for f in FEATURE_COLUMNS},
        "status": "unknown",
        "notes": [],
    }


def certify_row(model: str, runtime: str) -> dict[str, Any]:
    """Build a matrix row from adapter certification + capability flags."""

    adapter = adapter_for(runtime, model)
    report = certify_runtime(adapter, model=model)
    caps = adapter.capabilities
    features = {
        "tokens": "yes" if (caps.tokenize or caps.tokenization) else "no",
        "template": "yes" if (caps.render or caps.template or caps.rendered_prompt) else "no",
        "logits": "yes" if (caps.logits or caps.logprobs) else "no",
        "generation": "yes" if caps.generation else "no",
        "tool_calls": "yes" if caps.tools else "unknown",
        "json": "yes" if caps.structured_output else "unknown",
        "streaming": "yes" if (caps.streaming or caps.stream) else "unknown",
        "multimodal": "no",
    }
    level = getattr(report, "level", None) or ("gold" if report.passed else "none")
    return {
        "model": model,
        "runtime": runtime,
        "version": getattr(adapter, "version", None),
        "features": features,
        "status": str(level).lower(),
        "certification": report.to_dict() if hasattr(report, "to_dict") else {},
        "notes": [],
    }


def render_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["Model", "Runtime", "Status"] + [f.replace("_", " ").title() for f in FEATURE_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        feats = row.get("features") or {}
        cells = [
            str(row.get("model")),
            str(row.get("runtime")),
            str(row.get("status")),
        ] + [str(feats.get(f, "—")) for f in FEATURE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_matrix(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    md = path.with_suffix(".md")
    md.write_text(render_matrix_markdown(rows), encoding="utf-8")
    return path
