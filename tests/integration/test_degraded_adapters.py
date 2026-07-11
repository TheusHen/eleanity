from eleanity.adapters.llamacpp_adapter import LlamaCppAdapter
from eleanity.adapters.vllm_adapter import VLLMAdapter
from eleanity.models.schemas import LayerState, Scenario


def test_optional_adapters_degrade_without_optional_dependencies():
    scenario = Scenario(
        name="degraded",
        messages=[{"role": "user", "content": "oi"}],
        observe=["template", "tokens", "logits", "generation"],
    )
    for adapter in (VLLMAdapter("Qwen/Qwen2.5-0.5B-Instruct"), LlamaCppAdapter("missing.gguf")):
        assert adapter.forward(adapter.tokenize("oi")).state == LayerState.NOT_OBSERVABLE
        assert adapter.generate(scenario).state in {LayerState.NOT_OBSERVABLE, LayerState.OBSERVED}
