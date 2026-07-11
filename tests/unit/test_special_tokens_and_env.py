from eleanity.adapters.fake import FakeAdapter
from eleanity.core.run import observe, run_compare
from eleanity.fingerprints import collect_environment_fingerprint
from eleanity.models.schemas import LayerState, Scenario
from eleanity.scenarios import load_scenarios


def test_environment_fingerprint_has_python_and_platform():
    env = collect_environment_fingerprint()
    assert env.python_version
    assert env.platform
    assert "eleanity" in env.packages


def test_fake_adapter_exposes_special_tokens():
    observation = FakeAdapter().special_tokens()
    assert observation.state == LayerState.OBSERVED
    assert observation.data["eos_token_id"] == 2


def test_observe_includes_special_tokens_when_requested():
    scenario = Scenario(
        name="special",
        messages=[{"role": "user", "content": "x"}],
        observe=["special_tokens", "tokens"],
    )
    trace = observe(FakeAdapter(), scenario, "demo")
    assert set(trace.layers) == {"artifact", "special_tokens", "tokens"}
    assert trace.environment is not None
    assert trace.duration_ms is not None


def test_scenario_yaml_supports_large_model_policy():
    scenarios = load_scenarios("fixtures/qwen/scenarios.yaml")
    large = next(item for item in scenarios if item.name == "qwen-large-model-policy")
    assert large.model is not None
    assert large.model.id == "Qwen/Qwen2.5-7B-Instruct"
    assert large.model.device_map == "auto"
    assert large.model.dtype == "auto"
    assert "special_tokens" in large.observe


def test_rendered_prompt_alias_normalizes_to_template():
    scenario = Scenario(
        name="alias",
        messages=[{"role": "user", "content": "x"}],
        observe=["rendered_prompt", "stop_reason"],
    )
    assert "template" in scenario.observe
    assert "generation" in scenario.observe
    assert "rendered_prompt" not in scenario.observe


def test_compare_persists_environment_and_html_panel(tmp_path):
    run_id, _, _ = run_compare("demo", ["fake", "fake"], runs_dir=tmp_path)
    raw = (tmp_path / run_id / "result.json").read_text(encoding="utf-8")
    assert "environment" in raw
    from eleanity.reporters.html import write_html

    html = write_html(tmp_path / run_id / "result.json").read_text(encoding="utf-8")
    assert ("Executive summary" in html) or ("Resumo executivo" in html)
    assert "Causal pipeline" in html or "causal pipeline" in html
    assert "Environment fingerprint" in html or "environment fingerprint" in html
    assert "diag panel" in html
