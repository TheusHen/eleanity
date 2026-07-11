from eleanity.adapters.base import BackendAdapter, CapabilitySet, HealthcheckResult
from eleanity.adapters.fake import FakeAdapter
from eleanity.adapters.llamacpp_adapter import LlamaCppAdapter
from eleanity.adapters.ollama_adapter import OllamaAdapter
from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.adapters.registry import available_adapters, create_adapter, register_adapter
from eleanity.adapters.sglang_adapter import SGLangAdapter
from eleanity.adapters.tgi_adapter import TGIAdapter
from eleanity.adapters.transformers_adapter import TransformersAdapter
from eleanity.adapters.vllm_adapter import VLLMAdapter
from eleanity.models.schemas import ModelSpec, Scenario


def adapter_for(
    name: str,
    model: str,
    model_spec: ModelSpec | None = None,
    scenario: Scenario | None = None,
    *,
    tokenizer_only: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Instantiate a backend adapter by short name (builtins + entry-point plugins)."""

    return create_adapter(
        name,
        model,
        model_spec=model_spec,
        scenario=scenario,
        tokenizer_only=tokenizer_only,
        base_url=base_url,
        api_key=api_key,
    )


__all__ = [
    "BackendAdapter",
    "CapabilitySet",
    "HealthcheckResult",
    "FakeAdapter",
    "LlamaCppAdapter",
    "OllamaAdapter",
    "OpenAICompatAdapter",
    "SGLangAdapter",
    "TGIAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
    "adapter_for",
    "available_adapters",
    "create_adapter",
    "register_adapter",
]
