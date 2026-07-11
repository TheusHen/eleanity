from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from eleanity.models.schemas import ParityResult
from eleanity.utils.hashing import text_sha256
from eleanity.utils.security import redact_mapping


def redact_prompt_layers(payload: dict) -> dict:
    for trace in payload.get("traces") or []:
        layers = trace.get("layers") or {}
        for layer_name, observation in layers.items():
            data = observation.get("data") or {}
            if layer_name == "template":
                text = data.get("text") or data.get("rendered_text")
                if text is not None:
                    data["template_hash"] = (
                        data.get("template_hash") or data.get("chat_template_hash") or text_sha256(str(text))
                    )
                    data["rendered_byte_length"] = data.get("rendered_byte_length") or len(str(text).encode("utf-8"))
                    data["rendered_char_length"] = data.get("rendered_char_length") or len(str(text))
                data["text"] = None
                data["rendered_text"] = None
                data["rendered_utf8_hex"] = None
                data["content_redacted"] = True
                observation["data"] = data
            if layer_name == "tokens":
                ids = data.get("ids") or data.get("token_ids") or []
                data["count"] = data.get("count") or len(ids)
                data["ids"] = []
                data["token_ids"] = []
                data["token_strings"] = None
                data["decoded_text"] = None
                data["content_redacted"] = True
                observation["data"] = data
            if layer_name in {"generation", "streaming", "structured"}:
                if "text" in data:
                    data["text"] = None
                if "raw_text" in data:
                    data["raw_text"] = None
                if "ids" in data:
                    data["ids"] = []
                if "token_ids" in data:
                    data["token_ids"] = []
                data["content_redacted"] = True
                observation["data"] = data
    return payload


def write_result_json(target: Path, payload: dict, *, redact_prompts: bool = False) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    body = payload
    if redact_prompts:
        body = redact_prompt_layers(json.loads(json.dumps(payload)))
    path = target / "result.json"
    path.write_text(
        json.dumps(redact_mapping(body), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_junit(path: Path, run_id: str, diagnosis) -> Path:
    suite = Element(
        "testsuite",
        name="eleanity",
        tests="1",
        failures="1" if getattr(diagnosis, "status", None) == ParityResult.DIVERGENT else "0",
        errors="1" if getattr(diagnosis, "status", None) == ParityResult.ERROR else "0",
    )
    case = SubElement(suite, "testcase", classname="eleanity.compare", name=run_id)
    status = getattr(diagnosis, "status", None)
    if status == ParityResult.DIVERGENT:
        failure = SubElement(
            case,
            "failure",
            message=str(getattr(diagnosis, "summary", "DIVERGENT")),
            type="DIVERGENT",
        )
        failure.text = getattr(diagnosis, "hypothesis", "")
    elif status == ParityResult.ERROR:
        error = SubElement(
            case,
            "error",
            message=str(getattr(diagnosis, "summary", "ERROR")),
            type="ERROR",
        )
        error.text = getattr(diagnosis, "hypothesis", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_github_annotations(path: Path, diagnosis) -> Path:
    lines: list[str] = []
    status = getattr(diagnosis, "status", None)
    summary = getattr(diagnosis, "summary", "")
    if status == ParityResult.DIVERGENT:
        layer = getattr(diagnosis, "first_divergence", "unknown")
        lines.append(f"::error title=Eleanity divergence::first_divergence={layer} — {summary}")
    elif status == ParityResult.ERROR:
        lines.append(f"::error title=Eleanity error::{summary}")
    elif status in {ParityResult.PASS, ParityResult.PASS_WITH_TOLERANCE}:
        lines.append(f"::notice title=Eleanity pass::{summary}")
    else:
        lines.append(f"::warning title=Eleanity::{summary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
