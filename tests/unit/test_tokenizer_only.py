from eleanity.adapters.transformers_adapter import TransformersAdapter
from eleanity.core.engine import CompareEngine
from eleanity.models.schemas import LayerState, ModelSpec, Scenario


def test_transformers_tokenizer_only_capabilities_without_weights():
    adapter = TransformersAdapter(
        "demo",
        model_spec=ModelSpec(id="demo", tokenizer_only=True),
    )
    # Without transformers installed, capabilities collapse; with it, logits disabled.
    if adapter.capabilities.tokenize:
        assert adapter.capabilities.logits is False
        assert adapter.capabilities.generation is False
        note = adapter.forward(type("T", (), {"state": LayerState.OBSERVED, "data": {"ids": [1]}, "note": None})())
        assert note.state == LayerState.NOT_OBSERVABLE


def test_engine_tokenizer_only_strips_generation_layers(tmp_path):
    scenario = Scenario(
        name="tok",
        messages=[{"role": "user", "content": "hi"}],
        observe=["template", "tokens", "logits", "generation"],
    )
    engine = CompareEngine(runs_dir=tmp_path, parallel=False, tokenizer_only=True)
    result = engine.compare("demo", ["fake", "fake"], scenario=scenario, tokenizer_only=True)
    # fake still can generate, but observe list should exclude logits/generation when tokenizer_only
    for trace in result.traces:
        assert "logits" not in trace.layers
        assert "generation" not in trace.layers
        assert "template" in trace.layers
