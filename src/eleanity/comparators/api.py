from __future__ import annotations

from typing import Any

from eleanity.models.schemas import Comparison, ParityResult


def compare_api(left: dict[str, Any], right: dict[str, Any]) -> Comparison:
    """API contract comparison for OpenAI-compatible endpoints."""

    issues: list[str] = []
    details: dict[str, Any] = {}

    for key in ("http_status", "finish_reason", "has_usage", "openai_shape", "health_ok"):
        details[f"left_{key}"] = left.get(key)
        details[f"right_{key}"] = right.get(key)
        if key in left and key in right and left.get(key) != right.get(key):
            issues.append(key)

    if left.get("http_status") and int(left.get("http_status") or 0) >= 400:
        issues.append("baseline_http_error")
    if right.get("http_status") and int(right.get("http_status") or 0) >= 400:
        issues.append("candidate_http_error")

    # Usage coherence
    left_usage = left.get("usage") if isinstance(left.get("usage"), dict) else {}
    right_usage = right.get("usage") if isinstance(right.get("usage"), dict) else {}
    if left.get("has_usage") and right.get("has_usage"):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if field in left_usage and field in right_usage:
                details[f"left_{field}"] = left_usage.get(field)
                details[f"right_{field}"] = right_usage.get(field)
        # Incoherent usage: total < prompt+completion (when all present)
        for label, usage in (("baseline", left_usage), ("candidate", right_usage)):
            p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
            if isinstance(p, int) and isinstance(c, int) and isinstance(t, int) and t < p + c:
                issues.append(f"{label}_usage_incoherent")

    details["issues"] = issues
    if issues:
        return Comparison(result=ParityResult.DIVERGENT, details=details)
    return Comparison(result=ParityResult.PASS, details=details)


def compare_streaming(left: dict[str, Any], right: dict[str, Any]) -> Comparison:
    """Robust streaming contract comparison."""

    issues: list[str] = []
    details: dict[str, Any] = {
        "left_chunk_count": left.get("chunk_count"),
        "right_chunk_count": right.get("chunk_count"),
        "left_ordered": left.get("ordered"),
        "right_ordered": right.get("ordered"),
        "left_finish_reason": left.get("finish_reason"),
        "right_finish_reason": right.get("finish_reason"),
        "left_event_types": left.get("event_types"),
        "right_event_types": right.get("event_types"),
    }

    if left.get("ordered") is False or right.get("ordered") is False:
        issues.append("stream_order_broken")

    # DONE marker / terminal event
    left_events = list(left.get("event_types") or [])
    right_events = list(right.get("event_types") or [])
    if left_events and "done" not in left_events and left.get("finish_reason") is None:
        issues.append("baseline_missing_terminal")
    if right_events and "done" not in right_events and right.get("finish_reason") is None:
        issues.append("candidate_missing_terminal")

    if left.get("finish_reason") != right.get("finish_reason"):
        if left.get("finish_reason") is not None or right.get("finish_reason") is not None:
            issues.append("finish_reason")

    # Non-JSON events are a contract smell
    if "non_json" in left_events or "non_json" in right_events:
        issues.append("non_json_sse_frames")

    # Empty stream when other side produced content
    left_text = str(left.get("text") or "")
    right_text = str(right.get("text") or "")
    if left_text and not right_text and (right.get("chunk_count") or 0) == 0:
        issues.append("candidate_empty_stream")
    if right_text and not left_text and (left.get("chunk_count") or 0) == 0:
        issues.append("baseline_empty_stream")

    # First content chunk latency ratio (optional)
    left_ttft = left.get("ttft_ms")
    right_ttft = right.get("ttft_ms")
    if isinstance(left_ttft, (int, float)) and isinstance(right_ttft, (int, float)):
        details["left_ttft_ms"] = left_ttft
        details["right_ttft_ms"] = right_ttft

    details["issues"] = issues
    if issues:
        return Comparison(result=ParityResult.DIVERGENT, details=details)
    return Comparison(result=ParityResult.PASS, details=details)
