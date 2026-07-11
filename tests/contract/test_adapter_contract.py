from eleanity.adapters.fake import FakeAdapter
from eleanity.models.schemas import LayerState, Scenario


def test_adapter_contract_returns_explicit_observations():
    adapter = FakeAdapter()
    scenario = Scenario(
        name="contract",
        messages=[{"role": "user", "content": "hello"}],
        observe=["template", "tokens", "logits", "generation"],
    )
    rendered = adapter.render(scenario)
    tokens = adapter.tokenize(rendered.data["text"])
    assert adapter.capabilities.render is True
    assert adapter.capabilities.tokenize is True
    assert adapter.capabilities.logits is False
    assert adapter.capabilities.stream is True
    assert adapter.capabilities.tools is False
    assert adapter.capabilities.special_tokens is True
    assert adapter.capabilities.generation is True
    assert adapter.capabilities.structured_output is True
    assert rendered.state == LayerState.OBSERVED
    assert tokens.state == LayerState.OBSERVED
    assert adapter.forward(tokens).state == LayerState.NOT_OBSERVABLE
    assert adapter.generate(scenario).state == LayerState.OBSERVED
