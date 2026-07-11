from eleanity.comparators.diff import compare_json, compare_prompt, compare_tokens
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ObservationTrace,
    ParityProfile,
    ParityResult,
    Scenario,
)


def scenario():
    return Scenario.model_validate(
        {
            "name": "x",
            "messages": [{"role": "user", "content": "oi"}],
            "parameters": {"max_tokens": 3},
            "observe": ["template", "tokens"],
        }
    )


def test_scenario_sets_profile_tolerance_and_schema():
    value = scenario()
    assert value.parity_profile == ParityProfile.STRICT
    assert value.tolerance == 0.0
    assert "properties" in Scenario.model_json_schema()


def test_trace_serializes_layer_state():
    trace = ObservationTrace(
        trace_id="t",
        scenario_name="x",
        backend="fake",
        artifact_fingerprint=ArtifactFingerprint(model_ref="m"),
        layers={"template": LayerObservation(state=LayerState.OBSERVED, data={"text": "hi"})},
    )
    assert trace.layers["template"].state == LayerState.OBSERVED
    assert trace.model_dump(mode="json")["trace_version"] == "0"


def test_prompt_diff_returns_first_byte():
    result = compare_prompt("abc", "axc")
    assert result.result == ParityResult.DIVERGENT
    assert result.details["first_difference"] == 1


def test_token_diff_reports_downstream_impact():
    result = compare_tokens([1, 2, 3, 4], [1, 9, 8, 4])
    assert result.details["first_difference"] == 1
    assert result.details["downstream_different"] == 2
    assert result.details["downstream_percent"] == 66.67


def test_json_diff_compares_nested_tree():
    assert compare_json({"x": {"y": 1}}, {"x": {"y": 2}}).result == ParityResult.DIVERGENT
