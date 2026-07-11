"""Internal HTML report builder (not part of the shipped CLI product surface).

The product I/O is CLI text/json/quiet/sarif. This module remains for unit tests
and local experiments only — it is not exposed via CLI commands.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

LAYER_META = {
    "artifact": {
        "label": "Artifact",
        "short_label": "Artifact",
        "description": "Checkpoint, tokenizer, quantization, and load flags.",
    },
    "template": {
        "label": "Template / rendered prompt",
        "short_label": "Template",
        "description": "apply_chat_template output before tokenization.",
    },
    "special_tokens": {
        "label": "Special tokens",
        "short_label": "Special",
        "description": "bos/eos/pad/unk and extra tokenizer specials.",
    },
    "tokens": {
        "label": "Token IDs",
        "short_label": "Tokens",
        "description": "ID sequence produced by the tokenizer.",
    },
    "logits": {
        "label": "Forward / logits",
        "short_label": "Logits",
        "description": "Top-k of the last step before sampling.",
    },
    "generation": {
        "label": "Generation",
        "short_label": "Generation",
        "description": "Output tokens and stop_reason.",
    },
    "structured": {
        "label": "Structured output / tools",
        "short_label": "Structured",
        "description": "JSON, tool calling, and structured contracts.",
    },
    "streaming": {
        "label": "Streaming",
        "short_label": "Stream",
        "description": "Chunks, ordering, and stream semantics.",
    },
    "api": {
        "label": "API contract",
        "short_label": "API",
        "description": "Response shape and endpoint semantics.",
    },
}

RESULT_META = {
    "REFERENCE": {"label": "Reference", "short_label": "REF"},
    "PASS": {"label": "Exact parity", "short_label": "PASS"},
    "PASS_WITH_TOLERANCE": {
        "label": "Within tolerance",
        "short_label": "TOL",
    },
    "DIVERGENT": {"label": "Divergent", "short_label": "DIV"},
    "INCOMPARABLE": {"label": "Incomparable", "short_label": "INC"},
    "NOT_OBSERVABLE": {
        "label": "Not observable",
        "short_label": "N/O",
    },
}

STATE_LABELS = {
    "OBSERVED": "OBSERVED",
    "INCOMPARABLE": "INCOMPARABLE",
    "NOT_OBSERVABLE": "NOT_OBSERVABLE",
}

BACKEND_LABELS = {
    "transformers": "transformers",
    "vllm": "vllm",
    "llamacpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "fake": "Fake adapter",
}

POLICY_LABELS = {
    "strict": "strict",
    "quantized": "quantized",
    "functional": "functional",
    "api_conformance": "api_conformance",
}

DETAIL_LABELS = {
    "first_difference": "first_diff",
    "left_length": "len_ref",
    "right_length": "len_cand",
    "downstream_different": "downstream_n",
    "downstream_percent": "downstream_%",
    "max_delta": "max_delta",
    "tolerance": "tolerance",
    "path": "json_path",
    "divergent_keys": "keys",
}

FINGERPRINT_FIELDS = (
    ("model_ref", "model_ref"),
    ("revision", "revision"),
    ("model_hash", "model_hash"),
    ("tokenizer", "tokenizer"),
    ("tokenizer_hash", "tokenizer_hash"),
    ("chat_template_hash", "chat_template_hash"),
    ("quantization", "quantization"),
    ("dtype", "dtype"),
    ("backend_flags", "backend_flags"),
)

STATUS_PRIORITY = {
    "PASS": 1,
    "PASS_WITH_TOLERANCE": 2,
    "NOT_OBSERVABLE": 3,
    "INCOMPARABLE": 4,
    "DIVERGENT": 5,
}

SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|private[_-]?key|secret|access[_-]?token|refresh[_-]?token|auth[_-]?token|^token$)",
    re.IGNORECASE,
)
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|private[_-]?key|secret|access[_-]?token|refresh[_-]?token|auth[_-]?token)\b(\s*[:=]\s*)([^\s,;]+)"
)
BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")

TEMPLATE_DIR = Path(__file__).with_name("templates")
ENVIRONMENT = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)
ENVIRONMENT.filters["json_pretty"] = lambda value: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def _backend_label(value: str) -> str:
    return BACKEND_LABELS.get(value.lower(), value)


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        parsed = parsed.astimezone(UTC)
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return value


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        if not value:
            return "[]" if isinstance(value, list) else "{}"
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _safe_mapping(value: Any, key: str = "") -> Any:
    if key and SENSITIVE_KEY.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _safe_mapping(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_safe_mapping(item) for item in value]
    return value


def _redact_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    rendered = SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted]", rendered)
    return BEARER_TOKEN.sub("Bearer [redacted]", rendered)


def _content_digest(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _compact_value(value: Any) -> str:
    rendered = _display_value(value)
    if rendered == "—" or len(rendered) <= 36:
        return rendered
    if re.fullmatch(r"[a-fA-F0-9]{40,}", rendered):
        return f"{rendered[:12]}…{rendered[-8:]}"
    return rendered if len(rendered) <= 72 else f"{rendered[:64]}…"


def _format_detail(key: str, value: Any) -> str:
    if key == "downstream_percent" and isinstance(value, (int, float)):
        return f"{value:.1f}%"
    if key in {"max_delta", "tolerance"} and isinstance(value, (int, float)):
        return f"{value:.6g}"
    if value is None:
        return "—"
    return _display_value(value)


def _fingerprint_view(fingerprint: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key, label in FINGERPRINT_FIELDS:
        safe_value = _safe_mapping(fingerprint.get(key), key)
        rows.append(
            {
                "key": key,
                "label": label,
                "value": _display_value(safe_value),
                "compact": _compact_value(safe_value),
            }
        )
    return rows


def _data_view(layer: str, data: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "kind": layer if layer in LAYER_META else "raw",
        "digest": _content_digest(data),
    }

    if layer == "artifact":
        view["fields"] = _fingerprint_view(data)
    elif layer == "template":
        text = str(data.get("text", ""))
        view.update(
            {
                "characters": len(text),
                "bytes": len(text.encode("utf-8")),
                "template_hash": data.get("template_hash") or _content_digest({"text": text}),
                "add_generation_prompt": data.get("add_generation_prompt"),
                "content_omitted": bool(text),
            }
        )
    elif layer == "special_tokens":
        view.update(
            {
                "fields": [
                    {"label": key, "value": _display_value(value), "compact": _compact_value(value)}
                    for key, value in data.items()
                    if key
                    in {
                        "bos_token_id",
                        "eos_token_id",
                        "pad_token_id",
                        "unk_token_id",
                        "vocab_size",
                        "model_max_length",
                        "additional_special_tokens",
                        "chat_template_hash",
                    }
                ],
                "content_omitted": False,
            }
        )
    elif layer == "tokens":
        ids = data.get("ids", [])
        ids = ids if isinstance(ids, list) else []
        view.update(
            {
                "count": data.get("count", len(ids)),
                "special_token_count": data.get("special_token_count"),
                "content_omitted": bool(ids),
            }
        )
    elif layer == "logits":
        token_ids = data.get("top_ids", [])
        logits = data.get("top_logits", [])
        token_ids = token_ids if isinstance(token_ids, list) else []
        logits = logits if isinstance(logits, list) else []
        rows = []
        for index in range(max(len(token_ids), len(logits))):
            logit = logits[index] if index < len(logits) else "—"
            if isinstance(logit, float):
                logit = f"{logit:.6g}"
            rows.append(
                {
                    "rank": index + 1,
                    "token_id": token_ids[index] if index < len(token_ids) else "—",
                    "logit": logit,
                }
            )
        view["rows"] = rows
        view["device"] = data.get("device")
    elif layer == "generation":
        text = str(data.get("text", ""))
        ids = data.get("ids", [])
        ids = ids if isinstance(ids, list) else []
        view.update(
            {
                "text_characters": len(text),
                "stop_reason": _display_value(data.get("stop_reason")),
                "token_count": len(ids) or data.get("completion_token_count") or 0,
                "seed": data.get("seed"),
                "content_omitted": bool(text or ids),
            }
        )
    elif layer in {"structured", "api", "streaming"} or layer not in LAYER_META:
        view.update(
            {
                "field_count": len(data),
                "content_omitted": bool(data),
            }
        )
    return view


def _observation_summary(layer: str, observation: dict[str, Any]) -> str:
    state = observation.get("state", "NOT_OBSERVABLE")
    if state != "OBSERVED":
        return _redact_text(observation.get("note")) or STATE_LABELS.get(state, state)

    data = observation.get("data") or {}
    if layer == "artifact":
        model = data.get("model_ref") or "—"
        qualifiers = [data.get("dtype"), data.get("quantization")]
        suffix = " · ".join(str(item) for item in qualifiers if item)
        return f"{model}{' · ' + suffix if suffix else ''}"
    if layer == "template":
        text = str(data.get("text", ""))
        return f"{len(text.encode('utf-8'))} B rendered · hash={str(data.get('template_hash') or _content_digest({'text': text}))[:12]}"
    if layer == "special_tokens":
        return f"eos={data.get('eos_token_id')} pad={data.get('pad_token_id')} vocab={data.get('vocab_size')}"
    if layer == "tokens":
        return f"n={data.get('count', len(data.get('ids', [])))} special={data.get('special_token_count', '—')}"
    if layer == "logits":
        return f"top{len(data.get('top_logits', []))} · device={data.get('device', '—')}"
    if layer == "generation":
        return f"n={len(data.get('ids', []))} stop={_display_value(data.get('stop_reason'))}"
    return f"{len(data)} fields" if data else "observed"


def _result_view(result: str) -> dict[str, str]:
    result = result if result in RESULT_META else "NOT_OBSERVABLE"
    return {"code": result, **RESULT_META[result]}


def _comparison_view(comparison: dict[str, Any] | None) -> dict[str, Any]:
    comparison = comparison or {}
    result = _result_view(str(comparison.get("result", "NOT_OBSERVABLE")))
    details = comparison.get("details") or {}
    safe_details = []
    for key, value in details.items():
        if key not in DETAIL_LABELS and not isinstance(value, (int, float, bool, type(None), list, str)):
            continue
        if key in {"left", "right"} and isinstance(value, dict):
            continue
        safe_value = _redact_text(value) if isinstance(value, str) else value
        safe_details.append(
            {
                "key": key,
                "label": DETAIL_LABELS.get(key, key),
                "value": _format_detail(key, safe_value),
            }
        )
    result["details"] = safe_details
    return result


def _ordered_layers(traces: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for trace in traces:
        for layer in trace.get("layers") or {}:
            if layer not in found:
                found.append(layer)
    canonical = [layer for layer in LAYER_META if layer in found]
    return canonical + [layer for layer in found if layer not in canonical]


def _environment_view(data: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    env = data.get("environment") or {}
    if not env and traces:
        env = traces[0].get("environment") or {}
    packages = env.get("packages") or {}
    package_rows = [{"name": name, "version": version or "—"} for name, version in sorted(packages.items()) if version]
    return {
        "python_version": env.get("python_version") or "—",
        "platform": env.get("platform") or "—",
        "machine": env.get("machine") or "—",
        "cuda_available": env.get("cuda_available"),
        "cuda_version": env.get("cuda_version") or "—",
        "gpu_name": env.get("gpu_name") or "—",
        "torch_version": env.get("torch_version") or "—",
        "packages": package_rows,
        "available": bool(env),
    }


def _playbook_entries(causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from eleanity.playbook import get_playbook_entry

    entries = []
    for cause in causes:
        code = str(cause.get("code") or "")
        entry = get_playbook_entry(code) or {}
        entries.append(
            {
                "code": code,
                "confidence": cause.get("confidence"),
                "message": _redact_text(cause.get("message")) or entry.get("summary") or code,
                "title": entry.get("title") or code,
                "actions": entry.get("actions") or [],
                "files": entry.get("files") or [],
            }
        )
    return entries


def _build_charts(
    *,
    metrics_pass: int,
    metrics_tol: int,
    metrics_div: int,
    metrics_inc: int,
    metrics_no: int,
    layer_views: list[dict[str, Any]],
    backend_views: list[dict[str, Any]],
    timings: dict[str, Any],
    propagation: float,
    causes: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build presentation-ready chart series (rendered as pure SVG/CSS in the template)."""

    status_series = [
        {"key": "PASS", "label": "PASS", "value": metrics_pass, "color": "ok"},
        {"key": "TOL", "label": "TOL", "value": metrics_tol, "color": "warn"},
        {"key": "DIV", "label": "DIV", "value": metrics_div, "color": "bad"},
        {"key": "INC", "label": "INC", "value": metrics_inc, "color": "neutral"},
        {"key": "N/O", "label": "N/O", "value": metrics_no, "color": "muted"},
    ]
    status_total = sum(item["value"] for item in status_series) or 1
    circ = 251.327  # 2 * pi * 40
    status_cum = 0.0
    donut_segments: list[dict[str, Any]] = []
    for item in status_series:
        pct = item["value"] / status_total * 100
        item["percent"] = round(pct, 1)
        if item["value"] > 0:
            dash = pct / 100 * circ
            donut_segments.append(
                {
                    "color": item["color"],
                    "dash": round(dash, 2),
                    "gap": round(circ - dash, 2),
                    "offset": round(-(status_cum / 100 * circ), 2),
                }
            )
            status_cum += pct

    # Horizontal bar chart for timings
    timing_items: list[dict[str, Any]] = []
    if timings:
        for name, value in timings.items():
            timing_items.append(
                {
                    "label": _backend_label(str(name)),
                    "value": round(float(value or 0), 1),
                }
            )
    else:
        for index, trace in enumerate(traces):
            label = backend_views[index]["label"] if index < len(backend_views) else str(trace.get("backend"))
            timing_items.append(
                {
                    "label": label,
                    "value": round(float(trace.get("duration_ms") or 0), 1),
                }
            )
    max_t = max((item["value"] for item in timing_items), default=1.0) or 1.0
    for item in timing_items:
        item["percent"] = round(item["value"] / max_t * 100, 1)

    # Layer pipeline scores for heatmap-style strip
    layer_series = []
    score_map = {
        "PASS": 1.0,
        "REFERENCE": 1.0,
        "PASS_WITH_TOLERANCE": 0.7,
        "NOT_OBSERVABLE": 0.35,
        "INCOMPARABLE": 0.35,
        "DIVERGENT": 0.05,
        "ERROR": 0.0,
    }
    for layer in layer_views:
        code = layer["result"]["code"]
        layer_series.append(
            {
                "key": layer["key"],
                "id": layer["id"],
                "label": layer["short_label"],
                "code": code,
                "score": score_map.get(code, 0.3),
                "is_origin": layer.get("is_first_divergence", False),
            }
        )

    # Cause confidence bars
    cause_series = []
    for cause in causes[:6]:
        conf = cause.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_f = 0.0
        cause_series.append(
            {
                "code": cause.get("code") or "CAUSE",
                "confidence": conf_f,
                "percent": round(min(max(conf_f, 0.0), 1.0) * 100, 1),
                "message": _redact_text(cause.get("message")) or "",
            }
        )

    # Logits from first observed backend
    logits_series = []
    for trace in traces:
        logits = (trace.get("layers") or {}).get("logits") or {}
        if logits.get("state") != "OBSERVED":
            continue
        data = logits.get("data") or {}
        ids = data.get("top_ids") or []
        vals = data.get("top_logits") or []
        if not vals:
            continue
        max(abs(float(v)) for v in vals) or 1.0
        min_v = min(float(v) for v in vals)
        max_v = max(float(v) for v in vals)
        span = (max_v - min_v) or 1.0
        for rank, (tid, val) in enumerate(zip(ids, vals, strict=False), start=1):
            fval = float(val)
            logits_series.append(
                {
                    "rank": rank,
                    "token_id": tid,
                    "logit": round(fval, 4),
                    "percent": round((fval - min_v) / span * 100, 1),
                }
            )
        break

    # Backend health radar-ish: observed ratio
    backend_health = []
    for backend in backend_views:
        total = backend.get("total_layers") or 1
        observed = backend.get("observed") or 0
        backend_health.append(
            {
                "label": backend["short_label"],
                "observed": observed,
                "total": total,
                "percent": round(observed / total * 100, 1),
                "status": backend.get("summary_code") or "NOT_OBSERVABLE",
            }
        )

    prop = min(max(float(propagation or 0.0), 0.0), 100.0)
    return {
        "status_series": status_series,
        "status_total": status_total,
        "donut_segments": donut_segments,
        "timing_series": timing_items,
        "layer_series": layer_series,
        "cause_series": cause_series,
        "logits_series": logits_series,
        "backend_health": backend_health,
        "propagation": round(prop, 1),
        "propagation_remaining": round(100.0 - prop, 1),
        "propagation_dash": round(prop / 100.0 * circ, 2),
    }


def build_report_context(data: dict[str, Any]) -> dict[str, Any]:
    """Build a presentation-safe view model from a serialized Eleanity run."""

    traces = data.get("traces") or []
    if not traces:
        raise ValueError("Eleanity HTML reports require at least one trace")

    layers = _ordered_layers(traces)
    comparisons = data.get("comparisons") or {}
    diagnosis = data.get("diagnosis") or {}
    first_divergence = diagnosis.get("first_divergence")
    backend_counts = Counter(str(trace.get("backend", "runtime")) for trace in traces)
    backend_seen: Counter[str] = Counter()

    backend_views: list[dict[str, Any]] = []
    comparison_results: list[str] = []
    observed_count = 0

    # Support both plain backend keys and disambiguated keys (backend#2) from result.json.
    comparison_by_order: list[dict[str, Any]] = []
    for _key, value in comparisons.items():
        if isinstance(value, dict):
            comparison_by_order.append(value)

    for index, trace in enumerate(traces):
        backend = str(trace.get("backend", "runtime"))
        backend_seen[backend] += 1
        is_baseline = index == 0
        base_label = _backend_label(backend)
        if backend_counts[backend] > 1:
            suffix = "reference" if is_baseline else f"candidate {backend_seen[backend] - 1}"
            display_label = f"{base_label} · {suffix}"
        else:
            display_label = base_label

        trace_layers = trace.get("layers") or {}
        if is_baseline:
            candidate_comparisons = {}
        else:
            candidate_index = index - 1
            if candidate_index < len(comparison_by_order):
                candidate_comparisons = comparison_by_order[candidate_index]
            else:
                # Fallback for legacy single-key maps.
                alt_key = backend if backend_seen[backend] == 1 else f"{backend}#{backend_seen[backend]}"
                candidate_comparisons = comparisons.get(backend) or comparisons.get(alt_key) or {}
        observation_views: dict[str, dict[str, Any]] = {}
        comparison_views: dict[str, dict[str, Any]] = {}

        for layer in layers:
            observation = trace_layers.get(layer) or {
                "state": "NOT_OBSERVABLE",
                "data": {},
                "note": "layer absent in this trace",
            }
            state = str(observation.get("state", "NOT_OBSERVABLE"))
            if state == "OBSERVED":
                observed_count += 1
            observation_views[layer] = {
                "state": state,
                "state_label": STATE_LABELS.get(state, state),
                "note": _redact_text(observation.get("note")),
                "summary": _observation_summary(layer, observation),
                "view": _data_view(layer, observation.get("data") or {}),
            }

            if is_baseline:
                comparison_view = _result_view("REFERENCE")
                comparison_view["details"] = []
            else:
                comparison_view = _comparison_view(candidate_comparisons.get(layer))
                comparison_results.append(comparison_view["code"])
            comparison_views[layer] = comparison_view

        fingerprint = trace.get("artifact_fingerprint") or {}
        backend_views.append(
            {
                "id": f"{_slug(backend)}-{index + 1}",
                "name": backend,
                "label": display_label,
                "short_label": base_label,
                "is_baseline": is_baseline,
                "scenario_name": trace.get("scenario_name") or "unnamed",
                "trace_id": trace.get("trace_id") or "—",
                "trace_version": trace.get("trace_version") or "—",
                "created_at": _format_timestamp(trace.get("created_at")),
                "duration_ms": trace.get("duration_ms"),
                "model_ref": fingerprint.get("model_ref") or "—",
                "fingerprint": _fingerprint_view(fingerprint),
                "observations": observation_views,
                "comparisons": comparison_views,
                "observed": sum(item["state"] == "OBSERVED" for item in observation_views.values()),
                "total_layers": len(layers),
            }
        )

    first_index = layers.index(first_divergence) if first_divergence in layers else None
    layer_views: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        candidate_statuses = [
            backend["comparisons"][layer]["code"] for backend in backend_views if not backend["is_baseline"]
        ]
        aggregate = (
            max(candidate_statuses, key=lambda item: STATUS_PRIORITY.get(item, 0))
            if candidate_statuses
            else "NOT_OBSERVABLE"
        )
        if layer == first_divergence:
            aggregate = "DIVERGENT"

        if first_index is None:
            phase = "neutral"
        elif layer_index < first_index:
            phase = "upstream"
        elif layer_index == first_index:
            phase = "origin"
        else:
            phase = "downstream"

        meta = LAYER_META.get(
            layer,
            {
                "label": layer,
                "short_label": layer,
                "description": "Camada adicional registrada pelo trace.",
            },
        )
        result = _result_view(aggregate)
        layer_views.append(
            {
                "key": layer,
                "id": _slug(layer),
                "index": layer_index + 1,
                "number": f"{layer_index + 1:02d}",
                "phase": phase,
                "is_first_divergence": layer == first_divergence,
                **meta,
                "result": result,
                "backends": [
                    {
                        "id": backend["id"],
                        "label": backend["label"],
                        "short_label": backend["short_label"],
                        "is_baseline": backend["is_baseline"],
                        "observation": backend["observations"][layer],
                        "comparison": backend["comparisons"][layer],
                    }
                    for backend in backend_views
                ],
                "candidate_comparisons": [
                    {
                        "backend": backend["label"],
                        **backend["comparisons"][layer],
                    }
                    for backend in backend_views
                    if not backend["is_baseline"]
                ],
            }
        )

    for backend in backend_views:
        if backend["is_baseline"]:
            backend.update(
                {
                    "summary_code": "REFERENCE",
                    "summary_label": "baseline pairwise",
                    "comparison_coverage": "—",
                    "first_divergence": "baseline",
                }
            )
            continue

        results = [backend["comparisons"][layer]["code"] for layer in layers]
        comparable = [result for result in results if result not in {"NOT_OBSERVABLE", "INCOMPARABLE"}]
        backend_first = next(
            (
                LAYER_META.get(layer, {}).get("label", layer)
                for layer in layers
                if backend["comparisons"][layer]["code"] == "DIVERGENT"
            ),
            None,
        )
        if backend_first:
            summary_code = "DIVERGENT"
            summary_label = "divergent"
        elif not comparable:
            summary_code = "NOT_OBSERVABLE"
            summary_label = "insufficient evidence"
        elif "PASS_WITH_TOLERANCE" in comparable:
            summary_code = "PASS_WITH_TOLERANCE"
            summary_label = "within policy"
        else:
            summary_code = "PASS"
            summary_label = "parity on comparable layers"
        backend.update(
            {
                "summary_code": summary_code,
                "summary_label": summary_label,
                "comparison_coverage": f"{len(comparable)}/{len(layers)}",
                "first_divergence": backend_first or "none",
            }
        )

    total_observations = len(traces) * len(layers)
    total_comparisons = max(0, len(traces) - 1) * len(layers)
    comparable_results = [result for result in comparison_results if result not in {"NOT_OBSERVABLE", "INCOMPARABLE"}]
    divergent_count = comparison_results.count("DIVERGENT")
    tolerance_count = comparison_results.count("PASS_WITH_TOLERANCE")
    pass_count = comparison_results.count("PASS")

    if first_divergence or divergent_count:
        overall_code = "DIVERGENT"
        overall_label = "DIVERGENT"
        overall_title = f"first_divergence = {first_divergence or 'unknown'}"
    elif not comparable_results:
        overall_code = "NOT_OBSERVABLE"
        overall_label = "NOT_OBSERVABLE"
        overall_title = "no conclusive layer comparison"
    elif any(result == "PASS_WITH_TOLERANCE" for result in comparable_results):
        overall_code = "PASS_WITH_TOLERANCE"
        overall_label = "PASS_WITH_TOLERANCE"
        overall_title = "no divergence outside policy tolerance"
    else:
        overall_code = "PASS"
        overall_label = "PASS"
        overall_title = "no divergence on comparable layers"

    scenario_name = traces[0].get("scenario_name") or "unnamed"
    model_refs = list(
        dict.fromkeys(
            str((trace.get("artifact_fingerprint") or {}).get("model_ref"))
            for trace in traces
            if (trace.get("artifact_fingerprint") or {}).get("model_ref")
        )
    )
    if data.get("baseline_model") and data.get("candidate_model"):
        model_label = f"{data['baseline_model']} → {data['candidate_model']}"
        report_mode = "model conformance"
    elif len(model_refs) == 1:
        model_label = model_refs[0]
        report_mode = "runtime parity"
    elif model_refs:
        model_label = f"{len(model_refs)} models"
        report_mode = "model conformance"
    else:
        model_label = "—"
        report_mode = "conformance"

    propagation = diagnosis.get("propagation_percent")
    propagation_value = float(propagation or 0.0)
    first_layer_label = LAYER_META.get(first_divergence, {}).get("label", first_divergence or "none")
    scenario = data.get("scenario") or {}
    profile = scenario.get("parity_profile") or scenario.get("parity_policy")
    tolerance = scenario.get("tolerance")
    parameters = _safe_mapping(scenario.get("parameters") or {})
    requested_layers = scenario.get("observe") or []
    model_block = scenario.get("model") or {}

    matrix_first_divergence = next(
        (layer["key"] for layer in layer_views if layer["result"]["code"] == "DIVERGENT"),
        None,
    )
    integrity_notice = None
    if first_divergence and matrix_first_divergence and first_divergence != matrix_first_divergence:
        integrity_notice = (
            "Persisted diagnosis and aggregated matrix point to different layers. "
            "Review the comparison pair and result.json before using as a gate."
        )

    return {
        "run_id": data.get("run_id") or "run-sem-id",
        "run_short": str(data.get("run_id") or "run-sem-id")[:8],
        "scenario_name": scenario_name,
        "model_label": model_label,
        "report_mode": report_mode,
        "created_at": _format_timestamp(traces[0].get("created_at")),
        "baseline_backend": backend_views[0]["label"],
        "diagnosis": {
            "summary": _redact_text(diagnosis.get("summary")) or "—",
            "hypothesis": _redact_text(diagnosis.get("hypothesis")) or "—",
            "next_test": _redact_text(diagnosis.get("next_test")) or "—",
            "status": diagnosis.get("status") or overall_code,
            "first_divergence": first_divergence,
            "first_divergence_label": first_layer_label,
            "first_divergence_detail": diagnosis.get("first_divergence_detail"),
            "propagation": diagnosis.get("propagation") or {},
            "propagation_percent": propagation_value,
            "propagation_display": f"{propagation_value:.1f}%",
            "propagation_available": first_divergence in {"template", "tokens"},
            "probable_causes": [
                {
                    "code": item.get("code"),
                    "confidence": item.get("confidence"),
                    "message": _redact_text(item.get("message")) or "—",
                }
                for item in (diagnosis.get("probable_causes") or [])
            ],
            "suggested_actions": [_redact_text(item) or "—" for item in (diagnosis.get("suggested_actions") or [])],
            "warnings": [
                _redact_text(item) or "—" for item in (diagnosis.get("warnings") or data.get("warnings") or [])
            ],
            "scope_note": (
                f"Causal diagnosis: {backend_views[0]['label']} × {backend_views[1]['label']}; "
                "matrix includes all candidates."
                if len(backend_views) > 2
                else (
                    "Causal diagnosis refers to this run pair."
                    if len(backend_views) == 2
                    else "Sem segundo trace para formar par."
                )
            ),
        },
        "reproduction_command": data.get("reproduction_command")
        or f"eleanity report {data.get('run_id') or 'RUN_ID'} --format html",
        "consensus": data.get("consensus") or {},
        "capabilities": data.get("capabilities") or {},
        "gates": data.get("gates") or {},
        "timings_ms": data.get("timings_ms") or {},
        "total_duration_ms": data.get("total_duration_ms"),
        "tokenizer_only": data.get("tokenizer_only", False),
        "playbook": _playbook_entries(diagnosis.get("probable_causes") or []),
        "overall": {
            "code": overall_code,
            "label": overall_label,
            "title": overall_title,
            "summary": _redact_text(diagnosis.get("summary")) or "—",
        },
        "metrics": {
            "backend_count": len(traces),
            "candidate_count": max(0, len(traces) - 1),
            "layer_count": len(layers),
            "observed_count": observed_count,
            "total_observations": total_observations,
            "observability_percent": round(observed_count / total_observations * 100, 1) if total_observations else 0.0,
            "comparable_count": len(comparable_results),
            "total_comparisons": total_comparisons,
            "coverage_percent": round(len(comparable_results) / total_comparisons * 100, 1)
            if total_comparisons
            else 0.0,
            "divergent_count": divergent_count,
            "pass_count": pass_count,
            "tolerance_count": tolerance_count,
            "not_observable_count": comparison_results.count("NOT_OBSERVABLE"),
            "incomparable_count": comparison_results.count("INCOMPARABLE"),
        },
        "charts": _build_charts(
            metrics_pass=pass_count,
            metrics_tol=tolerance_count,
            metrics_div=divergent_count,
            metrics_inc=comparison_results.count("INCOMPARABLE"),
            metrics_no=comparison_results.count("NOT_OBSERVABLE"),
            layer_views=layer_views,
            backend_views=backend_views,
            timings=data.get("timings_ms") or {},
            propagation=propagation_value,
            causes=diagnosis.get("probable_causes") or [],
            traces=traces,
        ),
        "layers": layer_views,
        "backends": backend_views,
        "environment": _environment_view(data, traces),
        "scenario": {
            "profile": POLICY_LABELS.get(str(profile), _display_value(profile)),
            "profile_code": _display_value(profile),
            "tolerance": _display_value(tolerance),
            "schema_version": scenario.get("schema_version") or "—",
            "description": scenario.get("description") or "",
            "model": {
                "id": model_block.get("id") or model_label,
                "revision": model_block.get("revision") or "—",
                "dtype": model_block.get("dtype") or "—",
                "device_map": model_block.get("device_map") or "—",
                "trust_remote_code": model_block.get("trust_remote_code"),
            },
            "parameters": [{"label": str(key), "value": _display_value(value)} for key, value in parameters.items()],
            "requested_layers": [LAYER_META.get(str(layer), {}).get("label", str(layer)) for layer in requested_layers],
            "metadata_available": bool(scenario),
        },
        "source_schema_version": data.get("schema_version") or "legacy",
        "report_command": data.get("reproduction_command")
        or f"eleanity report {data.get('run_id') or 'RUN_ID'} --format html",
        "integrity_notice": integrity_notice,
    }


def render_html(data: dict[str, Any], source_filename: str = "result.json") -> str:
    """Render one self-contained, offline-safe HTML diagnostic panel."""

    template = ENVIRONMENT.get_template("report.html.j2")
    context = build_report_context(data)
    context["source_filename"] = Path(source_filename).name
    return template.render(**context)


def write_html(result_path: Path, output_path: Path | None = None) -> Path:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    output = output_path or result_path.with_name("report.html")
    output.write_text(render_html(data, result_path.name), encoding="utf-8")
    return output
