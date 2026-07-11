import pytest

from eleanity.adapters.fake import FakeAdapter
from eleanity.adapters.llamacpp_adapter import LlamaCppAdapter
from eleanity.adapters.vllm_adapter import VLLMAdapter
from eleanity.models.schemas import LayerState, Scenario

SCENARIO = Scenario(
    name="contract",
    messages=[{"role": "user", "content": "hello"}],
    parameters={"max_tokens": 1},
    observe=["template", "tokens", "logits", "generation"],
)


@pytest.mark.parametrize("adapter", [FakeAdapter(), VLLMAdapter("demo"), LlamaCppAdapter("demo.gguf")])
def test_each_adapter_method_returns_an_observation(adapter):
    fingerprint = adapter.fingerprint("demo")
    rendered = adapter.render(SCENARIO)
    tokens = adapter.tokenize(rendered.data.get("text", ""))
    logits = adapter.forward(tokens)
    generated = adapter.generate(SCENARIO)
    assert fingerprint.model_ref == "demo"
    for value in (rendered, tokens, logits, generated):
        assert value.state in set(LayerState)


def test_degraded_adapters_are_explicitly_not_observable():
    for adapter in (VLLMAdapter("demo"), LlamaCppAdapter("demo.gguf")):
        assert adapter.render(SCENARIO).state == LayerState.NOT_OBSERVABLE
        assert adapter.tokenize("hello").state == LayerState.NOT_OBSERVABLE
        assert adapter.forward(adapter.tokenize("hello")).state == LayerState.NOT_OBSERVABLE
        assert adapter.generate(SCENARIO).state == LayerState.NOT_OBSERVABLE
