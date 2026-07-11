from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_sarif(data: dict[str, Any]) -> dict[str, Any]:
    """Build SARIF 2.1.0 document from an Eleanity result payload."""

    diagnosis = data.get("diagnosis") or {}
    status = str(diagnosis.get("status") or "UNKNOWN")
    run_id = str(data.get("run_id") or "unknown")
    first = diagnosis.get("first_divergence")
    level = "none"
    if status in {"DIVERGENT", "ERROR"}:
        level = "error"
    elif status in {"NOT_OBSERVABLE", "INCOMPARABLE", "PASS_WITH_TOLERANCE"}:
        level = "warning"
    elif status == "PASS":
        level = "note"

    results: list[dict[str, Any]] = []
    if status != "PASS":
        message = diagnosis.get("summary") or diagnosis.get("hypothesis") or status
        location_text = first or "parity"
        results.append(
            {
                "ruleId": f"eleanity/{status.lower()}",
                "level": level,
                "message": {"text": str(message)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f"eleanity://runs/{run_id}"},
                            "region": {"startLine": 1, "message": {"text": location_text}},
                        }
                    }
                ],
                "properties": {
                    "first_divergence": first,
                    "status": status,
                    "probable_causes": diagnosis.get("probable_causes") or [],
                    "suggested_actions": diagnosis.get("suggested_actions") or [],
                },
            }
        )

    for cause in diagnosis.get("probable_causes") or []:
        results.append(
            {
                "ruleId": f"eleanity/cause/{cause.get('code', 'UNKNOWN')}",
                "level": "warning" if status != "ERROR" else "error",
                "message": {"text": str(cause.get("message") or cause.get("code"))},
                "properties": {
                    "confidence": cause.get("confidence"),
                    "code": cause.get("code"),
                },
            }
        )

    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        rid = str(item["ruleId"])
        if rid in seen:
            continue
        seen.add(rid)
        rules.append(
            {
                "id": rid,
                "name": rid.split("/")[-1],
                "shortDescription": {"text": rid},
                "fullDescription": {"text": "Eleanity runtime parity finding"},
                "helpUri": f"https://github.com/eleanity/eleanity/blob/main/docs/playbook/{rid.split('/')[-1]}.md",
            }
        )

    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "eleanity",
                        "informationUri": "https://github.com/eleanity/eleanity",
                        "version": "0.3.0",
                        "rules": rules,
                    }
                },
                "results": results,
                "properties": {
                    "run_id": run_id,
                    "baseline_backend": data.get("baseline_backend"),
                    "reproduction_command": data.get("reproduction_command"),
                },
            }
        ],
    }


def write_sarif(result_path: Path, output_path: Path | None = None) -> Path:
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    out = output_path or Path(result_path).with_name("results.sarif")
    out.write_text(json.dumps(build_sarif(data), indent=2), encoding="utf-8")
    return out
