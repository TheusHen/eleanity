from eleanity.adapters.fake import FakeAdapter
from eleanity.core.run import observe
from eleanity.models.schemas import Scenario


def test_observation_trace_persists_only_requested_layers_plus_artifact():
    scenario = Scenario(
        name="tokens-only",
        messages=[{"role": "user", "content": "oi"}],
        observe=["tokens"],
    )

    trace = observe(FakeAdapter(), scenario, "demo")

    assert set(trace.layers) == {"artifact", "tokens"}
