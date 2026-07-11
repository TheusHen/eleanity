from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

from eleanity.utils.hashing import file_sha256, text_sha256
from eleanity.utils.security import sanitize_path

# GGUF value types (subset)
GGUF_TYPE = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# Metadata keys useful for parity / chat template diagnostics
INTERESTING_PREFIXES = (
    "general.",
    "tokenizer.",
    "llama.",
    "qwen.",
    "chat_template",
)


def _read_string(handle: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    if length > 50_000_000:
        raise ValueError(f"unreasonable GGUF string length: {length}")
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("truncated GGUF string")
    return data.decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, vtype: int, *, depth: int = 0) -> Any:
    if depth > 4:
        raise ValueError("GGUF array nesting too deep")
    if vtype == 0:
        return struct.unpack("<B", handle.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", handle.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", handle.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", handle.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", handle.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", handle.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", handle.read(4))[0]
    if vtype == 7:
        return bool(struct.unpack("<B", handle.read(1))[0])
    if vtype == 8:
        return _read_string(handle)
    if vtype == 10:
        return struct.unpack("<Q", handle.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", handle.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", handle.read(8))[0]
    if vtype == 9:
        (atype,) = struct.unpack("<I", handle.read(4))
        (count,) = struct.unpack("<Q", handle.read(8))
        # Cap arrays to keep memory bounded
        limit = min(int(count), 1024)
        values = [_read_value(handle, atype, depth=depth + 1) for _ in range(limit)]
        # Skip remainder if truncated
        if count > limit:
            for _ in range(int(count) - limit):
                _read_value(handle, atype, depth=depth + 1)
            return {"_type": GGUF_TYPE.get(atype, str(atype)), "_truncated": True, "values": values, "count": count}
        return values
    raise ValueError(f"unsupported GGUF type {vtype}")


def inspect_gguf(path: Path | str, *, max_kv: int = 256, deep: bool = True) -> dict[str, Any]:
    """Parse GGUF header + metadata KV pairs for parity fingerprints.

    Deep mode walks KV values correctly (types 0–12) and extracts chat_template,
    tokenizer model, architecture, and quantization-related fields when present.
    """

    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"file not found: {p}", "path": sanitize_path(p)}

    size = p.stat().st_size
    info: dict[str, Any] = {
        "ok": True,
        "path": sanitize_path(p),
        "size_bytes": size,
        "sha256": file_sha256(p) if size < 512 * 1024 * 1024 else None,
        "format": "gguf",
        "deep": deep,
    }
    try:
        with p.open("rb") as handle:
            magic = handle.read(4)
            if magic != b"GGUF":
                info["ok"] = False
                info["error"] = f"not a GGUF file (magic={magic!r})"
                return info
            version = struct.unpack("<I", handle.read(4))[0]
            tensor_count = struct.unpack("<Q", handle.read(8))[0]
            kv_count = struct.unpack("<Q", handle.read(8))[0]
            info.update({"version": version, "tensor_count": tensor_count, "kv_count": kv_count})

            metadata: dict[str, Any] = {}
            keys: list[str] = []
            n = min(int(kv_count), max_kv)
            for _ in range(n):
                key = _read_string(handle)
                keys.append(key)
                (vtype,) = struct.unpack("<I", handle.read(4))
                if deep:
                    value = _read_value(handle, vtype)
                    # Store interesting keys fully; others as type tags / short strings
                    if any(key.startswith(prefix) or prefix in key for prefix in INTERESTING_PREFIXES):
                        if isinstance(value, str) and len(value) > 4000:
                            metadata[key] = {
                                "preview": value[:500],
                                "length": len(value),
                                "sha256": text_sha256(value),
                            }
                        else:
                            metadata[key] = value
                    elif isinstance(value, str):
                        metadata[key] = value if len(value) < 200 else f"<str len={len(value)}>"
                    elif isinstance(value, (int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = f"<{GGUF_TYPE.get(vtype, vtype)}>"
                else:
                    # Shallow: skip value body is hard without type sizes — use deep skip
                    _read_value(handle, vtype)

            info["metadata_keys"] = keys
            info["metadata"] = metadata
            # Convenience fields for artifact fingerprints
            chat = metadata.get("tokenizer.chat_template") or metadata.get("chat_template")
            if isinstance(chat, dict) and "sha256" in chat:
                info["chat_template_hash"] = chat["sha256"]
            elif isinstance(chat, str):
                info["chat_template_hash"] = text_sha256(chat)
            info["architecture"] = metadata.get("general.architecture")
            info["name"] = metadata.get("general.name")
            info["quantization_version"] = metadata.get("general.quantization_version")
            info["file_type"] = metadata.get("general.file_type")
            info["tokenizer_model"] = metadata.get("tokenizer.ggml.model")
            bos = metadata.get("tokenizer.ggml.bos_token_id")
            eos = metadata.get("tokenizer.ggml.eos_token_id")
            info["special_tokens"] = {"bos_token_id": bos, "eos_token_id": eos}
            # Fingerprint of selected parity-critical metadata
            critical = {
                k: metadata[k]
                for k in sorted(metadata)
                if k.startswith("tokenizer.") or k in {"general.architecture", "general.file_type"}
            }
            info["parity_fingerprint"] = text_sha256(json.dumps(critical, sort_keys=True, default=str))
            if int(kv_count) > n:
                info["notes"] = f"Parsed first {n}/{kv_count} KV pairs (max_kv cap)."
    except (OSError, ValueError, struct.error) as error:
        info["ok"] = False
        info["error"] = str(error)
    return info


def gguf_report_json(path: Path | str, *, deep: bool = True) -> str:
    return json.dumps(inspect_gguf(path, deep=deep), indent=2, ensure_ascii=False)


def gguf_to_artifact_fields(path: Path | str) -> dict[str, Any]:
    """Map GGUF inspect output onto ArtifactFingerprint-compatible fields."""

    data = inspect_gguf(path, deep=True)
    if not data.get("ok"):
        return {"quantization": "GGUF", "gguf_metadata": data, "backend_flags": {"gguf_error": data.get("error")}}
    return {
        "quantization": "GGUF",
        "architecture": data.get("architecture"),
        "chat_template_hash": data.get("chat_template_hash"),
        "model_hash": data.get("parity_fingerprint") or data.get("sha256"),
        "special_tokens": data.get("special_tokens") or {},
        "gguf_metadata": {
            "version": data.get("version"),
            "tensor_count": data.get("tensor_count"),
            "kv_count": data.get("kv_count"),
            "name": data.get("name"),
            "file_type": data.get("file_type"),
            "tokenizer_model": data.get("tokenizer_model"),
            "parity_fingerprint": data.get("parity_fingerprint"),
            "size_bytes": data.get("size_bytes"),
        },
    }
