from __future__ import annotations

import os
import threading
from collections.abc import Callable
from importlib import metadata
from typing import Any

from eleanity.models.schemas import ModelSpec, Scenario
from eleanity.utils.logging import get_logger, log_event

logger = get_logger("eleanity.adapters.registry")

AdapterFactory = Callable[[str, ModelSpec | None], Any]

_REGISTRY: dict[str, AdapterFactory] = {}
_PLUGINS_LOADED = False
_LOAD_LOCK = threading.Lock()


def register_adapter(name: str, factory: AdapterFactory) -> None:
    key = name.lower()
    if key in _REGISTRY and _REGISTRY[key] is factory:
        return
    _REGISTRY[key] = factory
    log_event(logger, "adapter_registered", name=key)


def _openai_like(
    name: str,
    env_url: str,
    env_key: str | None = None,
    *,
    tokenize_env: str | None = None,
):
    from eleanity.adapters.openai_compat import OpenAICompatAdapter

    def factory(model: str, spec: ModelSpec | None):
        url = os.getenv(env_url, "").rstrip("/")
        key = os.getenv(env_key) if env_key else None
        tokenize = os.getenv(tokenize_env) if tokenize_env else None
        return OpenAICompatAdapter(
            model,
            base_url=url,
            name=name,
            model_spec=spec,
            api_key=key,
            tokenize_path=tokenize,
        )

    return factory


def _builtin_factories() -> dict[str, AdapterFactory]:
    from eleanity.adapters.fake import FakeAdapter
    from eleanity.adapters.llamacpp_adapter import LlamaCppAdapter
    from eleanity.adapters.ollama_adapter import OllamaAdapter
    from eleanity.adapters.openai_compat import OpenAICompatAdapter
    from eleanity.adapters.sglang_adapter import SGLangAdapter
    from eleanity.adapters.tgi_adapter import TGIAdapter
    from eleanity.adapters.transformers_adapter import TransformersAdapter
    from eleanity.adapters.vllm_adapter import VLLMAdapter

    def openai_factory(model: str, spec: ModelSpec | None):
        url = os.getenv("ELEANITY_OPENAI_URL") or os.getenv("OPENAI_BASE_URL") or ""
        key = os.getenv("ELEANITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        return OpenAICompatAdapter(
            model,
            base_url=url,
            name="openai",
            model_spec=spec,
            api_key=key,
            tokenize_path=os.getenv("ELEANITY_OPENAI_TOKENIZE_PATH"),
        )

    def transformers_factory(model: str, spec: ModelSpec | None):
        return TransformersAdapter(model, model_spec=spec)

    return {
        "transformers": transformers_factory,
        "vllm": lambda m, s: VLLMAdapter(m, model_spec=s),  # base_url applied in create_adapter
        "llamacpp": lambda m, s: LlamaCppAdapter(m, model_spec=s),
        "ollama": lambda m, s: OllamaAdapter(m, model_spec=s),
        "sglang": lambda m, s: SGLangAdapter(m, model_spec=s),
        "tgi": lambda m, s: TGIAdapter(m, model_spec=s),
        "openai": openai_factory,
        "fake": lambda m, s: FakeAdapter(),
    }


def load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    with _LOAD_LOCK:
        if _PLUGINS_LOADED:
            return
        for name, factory in _builtin_factories().items():
            register_adapter(name, factory)
        try:
            eps = metadata.entry_points()
            selected = (
                eps.select(group="eleanity.adapters") if hasattr(eps, "select") else eps.get("eleanity.adapters", [])
            )
            for ep in selected:
                try:
                    loaded = ep.load()
                    register_adapter(ep.name, loaded)
                except Exception as error:  # pragma: no cover
                    log_event(logger, "plugin_load_failed", name=ep.name, error=str(error))
        except Exception as error:  # pragma: no cover
            log_event(logger, "plugin_discovery_failed", error=str(error))
        _PLUGINS_LOADED = True


def available_adapters() -> list[str]:
    load_plugins()
    return sorted(_REGISTRY)


def create_adapter(
    name: str,
    model: str,
    *,
    model_spec: ModelSpec | None = None,
    scenario: Scenario | None = None,
    tokenizer_only: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
):
    load_plugins()
    key = name.lower().strip()
    if model_spec is None and scenario is not None and scenario.model is not None:
        model_spec = scenario.model
        if model_spec.id:
            model = model_spec.id
    if tokenizer_only:
        if model_spec is None:
            model_spec = ModelSpec(id=model, tokenizer_only=True)
        else:
            model_spec = model_spec.model_copy(update={"tokenizer_only": True})
    if key not in _REGISTRY:
        known = ", ".join(available_adapters())
        raise ValueError(f"unknown backend: {name}. known: {known}")
    # Pass base_url into factories that accept it (vLLM)
    if key == "vllm":
        from eleanity.adapters.vllm_adapter import VLLMAdapter

        adapter = VLLMAdapter(model, model_spec=model_spec, base_url=base_url)
    else:
        adapter = _REGISTRY[key](model, model_spec)
    # Apply runtime profile overrides for OpenAI-compat family
    if base_url and hasattr(adapter, "base_url"):
        adapter.base_url = base_url.rstrip("/")
        if hasattr(adapter, "capabilities"):
            adapter.capabilities.generation = True
            adapter.capabilities.stream = True
            adapter.capabilities.streaming = True
            adapter.capabilities.healthcheck = True
            if key == "vllm" and not getattr(adapter, "tokenize_path", None):
                adapter.tokenize_path = "/tokenize"  # type: ignore[attr-defined]
    if api_key and hasattr(adapter, "api_key"):
        adapter.api_key = api_key
    return adapter
