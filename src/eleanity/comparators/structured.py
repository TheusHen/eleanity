from __future__ import annotations

import contextlib
import json
from typing import Any

from eleanity.comparators.diff import compare_json
from eleanity.models.schemas import Comparison, ParityResult


def validate_json_schema(instance: Any, schema: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Validate instance against a JSON Schema (Draft 2020-12 / 7 via jsonschema)."""

    if schema is None:
        return True, None
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if not errors:
            return True, None
        err = errors[0]
        path = ".".join(str(p) for p in err.path) or "$"
        return False, f"{path}: {err.message}"
    except ImportError:
        # Fallback: only type checks for trivial schemas
        expected = schema.get("type")
        if expected == "object" and not isinstance(instance, dict):
            return False, "expected object"
        if expected == "array" and not isinstance(instance, list):
            return False, "expected array"
        return True, None
    except Exception as error:  # pragma: no cover
        return False, str(error)


def extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("tool_calls")
    if not raw:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def normalize_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    fn = item.get("function") if isinstance(item.get("function"), dict) else item
    name = fn.get("name") if isinstance(fn, dict) else item.get("name")
    arguments = None
    if isinstance(fn, dict):
        arguments = fn.get("arguments")
    if arguments is None:
        arguments = item.get("arguments")
    if isinstance(arguments, str):
        with contextlib.suppress(json.JSONDecodeError):
            arguments = json.loads(arguments)
    return {"name": name, "arguments": arguments, "id": item.get("id")}


def compare_structured(left: dict[str, Any], right: dict[str, Any]) -> Comparison:
    """Functional structured-output comparison (JSON / schema / tools / stop)."""

    details: dict[str, Any] = {}

    # --- JSON validity ---
    left_json = left.get("is_json")
    right_json = right.get("is_json")
    if left_json is not None or right_json is not None:
        details["left_is_json"] = left_json
        details["right_is_json"] = right_json
        if left_json and not right_json:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, "reason": "candidate is not valid JSON"},
            )
        if right_json and not left_json:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, "reason": "baseline is not valid JSON"},
            )

    left_parsed = left.get("parsed")
    right_parsed = right.get("parsed")

    # --- JSON Schema validation (per side) ---
    schema = left.get("json_schema") or right.get("json_schema")
    if schema and left_parsed is not None:
        ok, err = validate_json_schema(left_parsed, schema)
        details["left_schema_valid"] = ok
        details["left_schema_error"] = err
        if not ok:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, "reason": f"baseline schema invalid: {err}"},
            )
    if schema and right_parsed is not None:
        ok, err = validate_json_schema(right_parsed, schema)
        details["right_schema_valid"] = ok
        details["right_schema_error"] = err
        if not ok:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, "reason": f"candidate schema invalid: {err}"},
            )

    # Explicit schema_valid flags from adapters
    if left.get("schema_valid") is False or right.get("schema_valid") is False:
        return Comparison(
            result=ParityResult.DIVERGENT,
            details={
                **details,
                "left_schema_valid": left.get("schema_valid"),
                "right_schema_valid": right.get("schema_valid"),
                "reason": "schema validation failed on one side",
            },
        )

    # --- Parsed JSON structural equality ---
    if left_parsed is not None and right_parsed is not None:
        required = left.get("required_keys") or right.get("required_keys") or []
        if required:
            for key in required:
                if key not in left_parsed or key not in right_parsed:
                    return Comparison(
                        result=ParityResult.DIVERGENT,
                        details={**details, "reason": f"missing required key: {key}"},
                    )
        json_cmp = compare_json(left_parsed, right_parsed)
        if json_cmp.result == ParityResult.DIVERGENT:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, **json_cmp.details, "reason": "parsed JSON differs"},
            )

    # --- Tool calling ---
    left_tools = [normalize_tool_call(t) for t in extract_tool_calls(left)]
    right_tools = [normalize_tool_call(t) for t in extract_tool_calls(right)]
    if left_tools or right_tools:
        left_names = [t["name"] for t in left_tools]
        right_names = [t["name"] for t in right_tools]
        details["left_tool_names"] = left_names
        details["right_tool_names"] = right_names
        if left_names != right_names:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={**details, "reason": "tool names differ"},
            )
        # Argument compare (order-sensitive for now)
        for index, (a, b) in enumerate(zip(left_tools, right_tools, strict=False)):
            if a.get("arguments") != b.get("arguments"):
                path_cmp = compare_json(a.get("arguments"), b.get("arguments"))
                return Comparison(
                    result=ParityResult.DIVERGENT,
                    details={
                        **details,
                        **path_cmp.details,
                        "reason": f"tool arguments differ at index {index}",
                        "tool_name": a.get("name"),
                    },
                )
        expected_tools = left.get("expected_tool_names") or right.get("expected_tool_names")
        if expected_tools and left_names != list(expected_tools):
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={
                    **details,
                    "reason": "tool names do not match expected_tool_names",
                    "expected": expected_tools,
                },
            )

    # --- Stop reason ---
    if left.get("stop_reason") != right.get("stop_reason"):
        # Functional: stop reason is meaningful; still divergent
        if left.get("stop_reason") is not None or right.get("stop_reason") is not None:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={
                    **details,
                    "left_stop_reason": left.get("stop_reason"),
                    "right_stop_reason": right.get("stop_reason"),
                    "reason": "stop_reason differs",
                },
            )

    return Comparison(result=ParityResult.PASS, details=details)


def build_structured_observation(
    *,
    text: str | None = None,
    tool_calls: Any = None,
    stop_reason: str | None = None,
    json_schema: dict[str, Any] | None = None,
    required_keys: list[str] | None = None,
    expected_tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """Helper for adapters: parse text, validate schema, package tool calls."""

    raw = (text or "").strip()
    parsed = None
    parse_error = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            if "```" in raw:
                parts = raw.split("```")
                if len(parts) >= 2:
                    block = parts[1]
                    if block.lstrip().startswith("json"):
                        block = block.split("\n", 1)[-1]
                    try:
                        parsed = json.loads(block.strip())
                        parse_error = None
                    except json.JSONDecodeError as nested:
                        parse_error = str(nested)
            else:
                parse_error = str(error)
    schema_valid = None
    schema_error = None
    if json_schema is not None and parsed is not None:
        schema_valid, schema_error = validate_json_schema(parsed, json_schema)
    return {
        "raw_text": raw,
        "parsed": parsed,
        "is_json": parsed is not None,
        "parse_error": parse_error,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "json_schema": json_schema,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "required_keys": required_keys or [],
        "expected_tool_names": expected_tool_names or [],
    }
