from eleanity.adapters.fake import FakeAdapter
from eleanity.core.engine import CompareEngine
from eleanity.core.matrix import consensus_summary
from eleanity.core.observe import observe
from eleanity.models.schemas import ParityProfile, ParityResult, Scenario
from eleanity.policies.engine import PolicyEngine


def test_policy_engine_marks_template_incomparable_under_functional():
    scenario = Scenario(
        name="f",
        messages=[{"role": "user", "content": "x"}],
        observe=["template", "generation"],
        parity_profile=ParityProfile.FUNCTIONAL,
    )
    left = observe(FakeAdapter(), scenario, "demo")
    right = observe(FakeAdapter(), scenario, "demo")
    # Force template text difference
    right.layers["template"].data["text"] = "totally different"
    right.layers["template"].data["rendered_text"] = "totally different"
    engine = PolicyEngine(scenario)
    result = engine.compare_layers(left, right)
    assert result["template"].result == ParityResult.INCOMPARABLE


def test_compare_engine_parallel_fake(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path, parallel=True, max_workers=2)
    result = engine.compare("demo", ["fake", "fake"])
    assert result.run_id
    assert len(result.traces) == 2
    assert (tmp_path / result.run_id / "result.json").exists()
    assert "consensus" in (tmp_path / result.run_id / "result.json").read_text(encoding="utf-8")
    assert result.diagnosis.status in {
        ParityResult.PASS,
        ParityResult.PASS_WITH_TOLERANCE,
        ParityResult.NOT_OBSERVABLE,
    }


def test_consensus_with_two_equal_fakes():
    scenario = Scenario(
        name="c",
        messages=[{"role": "user", "content": "hi"}],
        observe=["template", "tokens", "generation"],
    )
    traces = [
        observe(FakeAdapter(), scenario, "demo"),
        observe(FakeAdapter(), scenario, "demo"),
    ]
    summary = consensus_summary(traces, scenario)
    assert summary["pairs"] == 1
    assert summary["status"] in {ParityResult.PASS.value, ParityResult.PASS_WITH_TOLERANCE.value}
