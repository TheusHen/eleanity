"""Eleanity Trace Specification v1 — product format for interop and archives."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eleanity import __version__
from eleanity.spec.capsule import ExecutionCapsule
from eleanity.spec.impact import ImpactAssessment
from eleanity.spec.parity import FormalParityStatus, formal_status_from_parity, policy_comparator_set

TRACE_SCHEMA_VERSION = "1.0.0"
TRACE_SCHEMA_ID = "https://eleanity.dev/schemas/eleanity-trace-v1.schema.json"


def build_trace_document(
    *,
    run_id: str,
    result_payload: dict[str, Any],
    capsules: dict[str, ExecutionCapsule | dict[str, Any]] | None = None,
    impact: ImpactAssessment | dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
    privacy_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Trace Spec v1 document from an engine result payload."""

    diagnosis = result_payload.get("diagnosis") or {}
    status_raw = diagnosis.get("status") or "ERROR"
    formal = formal_status_from_parity(status_raw)
    scenario = result_payload.get("scenario") or {}
    policy = scenario.get("parity_profile") or scenario.get("parity_policy") or "strict"
    traces = result_payload.get("traces") or []

    subjects: dict[str, Any] = {}
    baseline = result_payload.get("baseline_backend")
    for index, trace in enumerate(traces):
        role = "baseline" if (baseline and trace.get("backend") == baseline) or index == 0 else f"candidate_{index}"
        if index == 1 and "candidate" not in subjects:
            role = "candidate"
        subjects[role] = {
            "backend": trace.get("backend"),
            "trace_id": trace.get("trace_id"),
            "artifact_fingerprint": trace.get("artifact_fingerprint"),
            "environment": trace.get("environment"),
            "duration_ms": trace.get("duration_ms"),
            "layers": {
                name: {
                    "state": (layer or {}).get("state"),
                    "note": (layer or {}).get("note"),
                    # data may be redacted upstream
                    "data": (layer or {}).get("data"),
                }
                for name, layer in (trace.get("layers") or {}).items()
            },
            "errors": trace.get("errors") or [],
            "warnings": trace.get("warnings") or [],
            "execution_capsule": (
                capsules.get(trace.get("backend") or role)
                if capsules
                else None
            ),
        }
        cap = subjects[role]["execution_capsule"]
        if hasattr(cap, "model_dump"):
            subjects[role]["execution_capsule"] = cap.model_dump(mode="json")

    impact_payload = impact.model_dump(mode="json") if hasattr(impact, "model_dump") else impact

    first = diagnosis.get("first_divergence_detail") or {}
    if not first and diagnosis.get("first_divergence"):
        first = {"layer": diagnosis.get("first_divergence")}

    doc: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "$schema": TRACE_SCHEMA_ID,
        "eleanity_version": __version__,
        "run_id": run_id,
        "run_type": result_payload.get("run_type") or "compare",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "name": policy,
            "comparators": policy_comparator_set(policy).to_dict()["comparators"],
        },
        "execution_capsule": _shared_capsule(capsules, result_payload),
        "subjects": subjects,
        "observations": {
            "requested_layers": scenario.get("observe") or [],
            "per_subject_states": {
                role: {
                    layer: meta.get("state")
                    for layer, meta in (subj.get("layers") or {}).items()
                }
                for role, subj in subjects.items()
            },
        },
        "comparisons": result_payload.get("comparisons") or {},
        "first_divergence": first,
        "propagation": _propagation_list(diagnosis, impact_payload),
        "parity": {
            "status": formal.value,
            "legacy_status": status_raw,
            "status_definition": {
                "PASS": "required observed layers match under comparator modes",
                "PASS_WITH_TOLERANCE": "within declared numeric/prefix thresholds only",
                "DIVERGENT": "required observed layer failed comparator",
                "INCONCLUSIVE": "insufficient observation or self-inconsistency",
                "UNSUPPORTED": "adapter cannot expose required layer",
                "ERROR": "execution/observation failure",
            }.get(formal.value),
        },
        "impact": impact_payload,
        "stability": stability,
        "diagnostics": _diagnostics_list(diagnosis),
        "gates": result_payload.get("gates"),
        "timings_ms": result_payload.get("timings_ms"),
        "reproduction_command": result_payload.get("reproduction_command"),
        "privacy": privacy_manifest or {
            "redacted": bool(result_payload.get("redacted")),
            "content_left_machine": False,
        },
        "extensions": {},
    }
    doc["document_hash"] = _hash_doc(doc)
    return doc


def _shared_capsule(
    capsules: dict[str, Any] | None,
    result_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if capsules:
        first = next(iter(capsules.values()))
        if hasattr(first, "model_dump"):
            return first.model_dump(mode="json")
        if isinstance(first, dict):
            return first
    return result_payload.get("execution_capsule")


def _propagation_list(diagnosis: dict[str, Any], impact: dict[str, Any] | None) -> list[Any]:
    items: list[Any] = []
    prop = diagnosis.get("propagation") or {}
    if prop:
        items.append({"type": "token_propagation", **prop})
    if impact and impact.get("propagation_layers"):
        items.append({"type": "layer_propagation", "layers": impact["propagation_layers"]})
    return items


def _diagnostics_list(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cause in diagnosis.get("probable_causes") or []:
        out.append(
            {
                "code": cause.get("code"),
                "confidence": cause.get("confidence"),
                "message": cause.get("message"),
                "evidence": cause.get("evidence") or {},
                "affected_layers": cause.get("affected_layers") or [],
                "suggested_remediation": cause.get("suggested_remediation")
                or (diagnosis.get("suggested_actions") or [None])[0],
            }
        )
    return out


def _hash_doc(doc: dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != "document_hash"}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_trace_document(doc: dict[str, Any]) -> list[str]:
    """Lightweight structural validation (JSON Schema optional via jsonschema)."""

    errors: list[str] = []
    if doc.get("schema_version") != TRACE_SCHEMA_VERSION:
        # allow future minor via prefix
        ver = str(doc.get("schema_version") or "")
        if not ver.startswith("1."):
            errors.append(f"unsupported schema_version: {ver}")
    for key in ("run_id", "subjects", "parity", "comparisons"):
        if key not in doc:
            errors.append(f"missing required field: {key}")
    parity = doc.get("parity") or {}
    status = parity.get("status")
    try:
        if status:
            FormalParityStatus(status)
    except ValueError:
        errors.append(f"invalid parity.status: {status}")
    if not isinstance(doc.get("subjects"), dict) or not doc.get("subjects"):
        errors.append("subjects must be a non-empty object")
    # Optional JSON Schema validation
    try:
        import jsonschema

        schema_path = Path(__file__).resolve().parents[3] / "schemas" / "eleanity-trace-v1.schema.json"
        if schema_path.is_file():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(doc, schema)
    except ImportError:
        pass
    except Exception as error:  # schema errors
        errors.append(str(error))
    return errors


def migrate_result_to_v1(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy result.json (schema 0.x) into Trace Spec v1."""

    return build_trace_document(
        run_id=str(result_payload.get("run_id") or "unknown"),
        result_payload=result_payload,
    )


def write_trace_v1(path: Path, doc: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
