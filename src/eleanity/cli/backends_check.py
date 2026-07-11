from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

from eleanity.adapters import adapter_for, available_adapters
from eleanity.cli.errors import EleanityError, missing_dep, missing_url
from eleanity.cli.resolve import ResolvedCompare


HTTP_BACKENDS = {
    "vllm": "ELEANITY_VLLM_URL",
    "llamacpp": "ELEANITY_LLAMACPP_URL",
    "ollama": "ELEANITY_OLLAMA_URL",
    "sglang": "ELEANITY_SGLANG_URL",
    "tgi": "ELEANITY_TGI_URL",
    "openai": "ELEANITY_OPENAI_URL",
}

DEP_HINTS = {
    "transformers": ("transformers", "transformers"),
    "vllm": ("vllm", "vllm"),
    "llamacpp": ("llama_cpp", "llamacpp"),
}


@dataclass
class BackendCheck:
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


def check_backends(
    names: list[str],
    *,
    model: str = "demo",
    resolved: ResolvedCompare | None = None,
    require_healthy: bool = False,
) -> list[BackendCheck]:
    results: list[BackendCheck] = []
    for name in names:
        key = name.lower().strip()
        if key not in available_adapters():
            results.append(BackendCheck(name=key, ok=False, detail="unknown adapter"))
            continue
        if key == "fake":
            results.append(BackendCheck(name=key, ok=True, detail="offline adapter"))
            continue
        if key == "transformers":
            if importlib.util.find_spec("transformers") is None:
                results.append(
                    BackendCheck(
                        name=key,
                        ok=False,
                        detail="not installed — uv sync --extra transformers",
                    )
                )
            else:
                results.append(BackendCheck(name=key, ok=True, detail="package installed"))
            continue

        # HTTP family
        url = None
        if resolved and key in resolved.backend_urls:
            url = resolved.backend_urls[key]
        elif resolved and key in resolved.backend_profiles:
            url = resolved.backend_profiles[key].base_url
        env = HTTP_BACKENDS.get(key)
        if not url and env:
            url = os.getenv(env)
        if not url and key == "openai":
            url = os.getenv("OPENAI_BASE_URL")
        if not url and key == "ollama":
            url = os.getenv("ELEANITY_OLLAMA_URL") or "http://127.0.0.1:11434"

        if not url:
            results.append(
                BackendCheck(
                    name=key,
                    ok=False,
                    detail=f"URL unset ({HTTP_BACKENDS.get(key, 'set base_url')})",
                )
            )
            continue

        try:
            adapter = adapter_for(key, model, base_url=url)
            health = getattr(adapter, "healthcheck", None)
            if callable(health):
                hr = health()
                results.append(
                    BackendCheck(
                        name=key,
                        ok=bool(hr.ok),
                        detail=str(hr.detail),
                        latency_ms=hr.latency_ms,
                    )
                )
            else:
                results.append(BackendCheck(name=key, ok=True, detail=f"url={url}"))
        except Exception as error:
            results.append(BackendCheck(name=key, ok=False, detail=str(error)))

    if require_healthy:
        bad = [r for r in results if not r.ok and r.name != "fake"]
        # Only require health for backends that need URL when they're selected
        critical = [r for r in bad if r.name in names]
        if critical and any(r.name in HTTP_BACKENDS for r in critical):
            first = critical[0]
            if "URL unset" in first.detail:
                raise missing_url(first.name, HTTP_BACKENDS.get(first.name, "URL"))
            raise EleanityError(
                code="ELEANITY_E004",
                message=f"backend unhealthy: {first.name} — {first.detail}",
                hint="eleanity doctor --check-backends",
                exit_code=2,
            )
    return results


def ensure_local_dep(backend: str) -> None:
    if backend == "transformers" and importlib.util.find_spec("transformers") is None:
        raise missing_dep("transformers", "transformers")
