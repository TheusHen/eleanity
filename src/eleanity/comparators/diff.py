from __future__ import annotations

import unicodedata
from typing import Any

from eleanity.models.schemas import Comparison, ParityResult


def _first(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _line_col_from_char(text: str, index: int) -> tuple[int, int]:
    if index < 0:
        return 1, 1
    prefix = text[:index]
    line = prefix.count("\n") + 1
    last_nl = prefix.rfind("\n")
    col = index - last_nl if last_nl >= 0 else index + 1
    return line, col


def _context(text: str, index: int, window: int = 24) -> dict[str, str]:
    start = max(0, index - window)
    end = min(len(text), index + window)
    return {
        "before": text[start:index],
        "after": text[index:end],
        "snippet": text[start:end],
    }


def _detect_special_markers(text: str) -> list[str]:
    markers = []
    for marker in (
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
    ):
        if marker in text:
            markers.append(marker)
    return markers


def compare_prompt(left: str, right: str) -> Comparison:
    """Byte/char/line-aware rendered prompt comparison."""

    left = left or ""
    right = right or ""
    left_bytes = list(left.encode("utf-8"))
    right_bytes = list(right.encode("utf-8"))
    byte_index = _first(left_bytes, right_bytes)
    char_index = _first(list(left), list(right))

    if byte_index is None and char_index is None:
        return Comparison(
            result=ParityResult.PASS,
            details={
                "first_difference": None,
                "first_byte": None,
                "first_character": None,
                "left_length": len(left_bytes),
                "right_length": len(right_bytes),
            },
        )

    # Prefer character index for diagnostics; keep first_difference as byte for legacy tests.
    first_diff = byte_index if byte_index is not None else char_index
    char_i = char_index if char_index is not None else 0
    line, col = _line_col_from_char(left if char_i < len(left) else right, char_i)

    left_ctx = _context(left, char_i)
    right_ctx = _context(right, char_i)

    left_nfc = unicodedata.normalize("NFC", left)
    right_nfc = unicodedata.normalize("NFC", right)
    left_nfd = unicodedata.normalize("NFD", left)
    right_nfd = unicodedata.normalize("NFD", right)

    whitespace_only = left.replace(" ", "").replace("\t", "").replace("\r", "").replace(
        "\n", ""
    ) == right.replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    newline_diff = left.count("\n") != right.count("\n") or ("\r\n" in left) != ("\r\n" in right)
    unicode_norm_diff = (left != left_nfc or right != right_nfc) and left_nfc == right_nfc

    left_markers = set(_detect_special_markers(left))
    right_markers = set(_detect_special_markers(right))
    missing_markers = sorted(left_markers - right_markers)
    extra_markers = sorted(right_markers - left_markers)

    # Prefer special-token assistant turn markers over bare "assistant" text.
    assistant_markers = (
        "<|im_start|>assistant",
        "<|assistant|>",
        "<|im_start|>assistant\n",
    )
    left_has_assistant = any(m in left for m in assistant_markers)
    right_has_assistant = any(m in right for m in assistant_markers)
    # Also treat trailing generation prompt without special tokens when baseline has markers.
    if left_has_assistant and not right_has_assistant:
        missing_assistant_turn = True
    elif left.rstrip().endswith(("<|im_start|>assistant", "<|im_start|>assistant\n")) and not right_has_assistant:
        missing_assistant_turn = True
    else:
        missing_assistant_turn = False

    return Comparison(
        result=ParityResult.DIVERGENT,
        details={
            "first_difference": first_diff,
            "first_byte": byte_index,
            "first_character": char_index,
            "line": line,
            "column": col,
            "left_length": len(left_bytes),
            "right_length": len(right_bytes),
            "left_char_length": len(left),
            "right_char_length": len(right),
            "left_context": left_ctx,
            "right_context": right_ctx,
            "whitespace_only_difference": whitespace_only and left != right,
            "newline_difference": newline_diff,
            "unicode_nfc_equal": left_nfc == right_nfc,
            "unicode_nfd_equal": left_nfd == right_nfd,
            "unicode_normalization_difference": unicode_norm_diff,
            "missing_markers": missing_markers,
            "extra_markers": extra_markers,
            "missing_assistant_turn": missing_assistant_turn,
            "left_roles_hint": [r for r in ("system", "user", "assistant", "tool") if r in left],
            "right_roles_hint": [r for r in ("system", "user", "assistant", "tool") if r in right],
            "baseline_snippet": left_ctx.get("snippet"),
            "candidate_snippet": right_ctx.get("snippet"),
        },
    )


def compare_tokens(
    left: list[int],
    right: list[int],
    *,
    left_strings: list[str] | None = None,
    right_strings: list[str] | None = None,
    left_special: dict[str, Any] | None = None,
    right_special: dict[str, Any] | None = None,
) -> Comparison:
    """Token-ID comparison with prefix/suffix, ops, and special-token diagnostics."""

    left = list(left or [])
    right = list(right or [])
    index = _first(left, right)

    # Equal prefix / suffix lengths
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    suffix = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        suffix += 1
        if prefix + suffix > min(len(left), len(right)):
            suffix -= 1
            break

    if index is None:
        special_diff = _special_token_diff(left_special, right_special)
        if special_diff:
            return Comparison(
                result=ParityResult.DIVERGENT,
                details={
                    "first_difference": None,
                    "downstream_different": 0,
                    "downstream_percent": 0.0,
                    "special_token_differences": special_diff,
                    "equal_prefix": prefix,
                    "equal_suffix": suffix,
                },
            )
        return Comparison(
            result=ParityResult.PASS,
            details={
                "first_difference": None,
                "downstream_different": 0,
                "downstream_percent": 0.0,
                "equal_prefix": prefix,
                "equal_suffix": suffix,
                "total_differences": 0,
            },
        )

    tail = max(len(left), len(right)) - index
    different = sum(a != b for a, b in zip(left[index:], right[index:])) + abs(
        len(left[index:]) - len(right[index:])
    )
    percent = round(different / tail * 100, 2) if tail else 0.0

    # Classify ops after first divergence (simple LCS-free heuristic)
    inserted = max(0, len(right) - len(left))
    removed = max(0, len(left) - len(right))
    substituted = sum(1 for a, b in zip(left[index:], right[index:]) if a != b)

    expected_id = left[index] if index < len(left) else None
    received_id = right[index] if index < len(right) else None
    expected_str = (
        left_strings[index]
        if left_strings is not None and index < len(left_strings)
        else None
    )
    received_str = (
        right_strings[index]
        if right_strings is not None and index < len(right_strings)
        else None
    )

    special_diff = _special_token_diff(left_special, right_special)
    left_trunc = bool((left_special or {}).get("truncated"))
    right_trunc = bool((right_special or {}).get("truncated"))

    return Comparison(
        result=ParityResult.DIVERGENT,
        details={
            "first_difference": index,
            "expected_token_id": expected_id,
            "received_token_id": received_id,
            "expected_token_string": expected_str,
            "received_token_string": received_str,
            "downstream_different": different,
            "downstream_percent": percent,
            "total_differences": different,
            "equal_prefix": prefix,
            "equal_suffix": suffix,
            "inserted": inserted,
            "removed": removed,
            "substituted": substituted,
            "special_token_differences": special_diff,
            "truncation_difference": left_trunc != right_trunc,
            "left_length": len(left),
            "right_length": len(right),
        },
    )


def _special_token_diff(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    if not left and not right:
        return {}
    left = left or {}
    right = right or {}
    keys = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "added_special_tokens",
        "padding_side",
        "truncation_side",
    )
    diff = {}
    for key in keys:
        if left.get(key) != right.get(key):
            diff[key] = {"baseline": left.get(key), "candidate": right.get(key)}
    if left.get("pad_token_id") is not None and left.get("pad_token_id") == left.get("eos_token_id"):
        if right.get("pad_token_id") != right.get("eos_token_id"):
            diff["pad_as_eos_baseline"] = True
    if right.get("pad_token_id") is not None and right.get("pad_token_id") == right.get(
        "eos_token_id"
    ):
        if left.get("pad_token_id") != left.get("eos_token_id"):
            diff["pad_as_eos_candidate"] = True
    return diff


def compare_generation(left: list[int], right: list[int]) -> Comparison:
    return compare_tokens(left, right)


def compare_json(left: Any, right: Any) -> Comparison:
    if left == right:
        return Comparison(result=ParityResult.PASS)

    def walk(a: Any, b: Any, path: str = "$") -> str:
        if type(a) is not type(b):
            return path
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    return f"{path}.{key}"
                found = walk(a[key], b[key], f"{path}.{key}")
                if found:
                    return found
        elif isinstance(a, list):
            for index, (x, y) in enumerate(zip(a, b)):
                found = walk(x, y, f"{path}[{index}]")
                if found:
                    return found
            if len(a) != len(b):
                return f"{path}[{min(len(a), len(b))}]"
        elif a != b:
            return path
        return ""

    return Comparison(result=ParityResult.DIVERGENT, details={"path": walk(left, right)})


def compare_logits(left: list[float], right: list[float], tolerance: float) -> Comparison:
    if not left and not right:
        return Comparison(result=ParityResult.NOT_OBSERVABLE, details={"reason": "empty logits"})
    if not left or not right:
        return Comparison(
            result=ParityResult.NOT_OBSERVABLE,
            details={"reason": "one side missing logits"},
        )
    pairs = list(zip(left, right))
    delta = max((abs(a - b) for a, b in pairs), default=0.0)
    # Ranking agreement (top-k order when ids not provided — value ranks only)
    left_rank = sorted(range(len(left)), key=lambda i: left[i], reverse=True)
    right_rank = sorted(range(len(right)), key=lambda i: right[i], reverse=True)
    rank_agree = sum(1 for a, b in zip(left_rank, right_rank) if a == b)
    rank_ratio = rank_agree / max(len(left_rank), 1)
    details = {
        "max_delta": delta,
        "tolerance": tolerance,
        "rank_agreement": round(rank_ratio, 4),
        "compared_k": min(len(left), len(right)),
    }
    if delta == 0:
        return Comparison(result=ParityResult.PASS, details=details)
    if delta <= tolerance:
        return Comparison(result=ParityResult.PASS_WITH_TOLERANCE, details=details)
    return Comparison(result=ParityResult.DIVERGENT, details=details)


def compare_special_tokens(left: dict[str, Any], right: dict[str, Any]) -> Comparison:
    diff = _special_token_diff(left, right)
    if not diff:
        return Comparison(result=ParityResult.PASS, details={})
    return Comparison(result=ParityResult.DIVERGENT, details={"divergent_keys": list(diff), **diff})
