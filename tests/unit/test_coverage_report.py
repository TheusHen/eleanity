"""Tests for coverage, new statuses, observation vs comparison, report fields."""

from __future__ import annotations

import json

from eleanity.core.coverage import (
    apply_coverage_to_status,
    classify_unobserved,
    compute_coverage,
    diagnosis_confidence,
    format_timings,
    policy_required_layers,
)
from eleanity.core.engine import CompareEngine
from eleanity.diagnosers import diagnose
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ObservationTrace,
    ParityResult,
    Scenario,
)
from eleanity.policies.engine import PolicyEngine


def _trace(backend: str, layers: dict[str, LayerObservation]) -> ObservationTrace:
    return ObservationTrace(
        scenario_name="t",
        backend=backend,
        artifact_fingerprint=ArtifactFingerprint(model_ref="demo"),
        layers=layers,
    )


def test_policy_required_layers():
    assert "template" in policy_required_layers("strict")
    assert "generation" in policy_required_layers("functional")


def test_classify_unobserved_states():
    assert classify_unobserved(requested=False, left=None, right=None) == ParityResult.NOT_REQUESTED
    left = LayerObservation(state=LayerState.NOT_SUPPORTED, note="nope")
    right = LayerObservation(state=LayerState.OBSERVED, data={})
    assert classify_unobserved(requested=True, left=left, right=right) == ParityResult.NOT_SUPPORTED
    left = LayerObservation(state=LayerState.NOT_EXPOSED, note="http")
    assert classify_unobserved(requested=True, left=left, right=right) == ParityResult.NOT_OBSERVABLE


def test_pass_with_limited_coverage_when_required_missing():
    left = _trace(
        "a",
        {
            "artifact": LayerObservation(state=LayerState.OBSERVED, data={"model_ref": "x"}),
            "generation": LayerObservation(state=LayerState.OBSERVED, data={"text": "hi", "ids": [1]}),
        },
    )
    right = _trace(
        "b",
        {
            "artifact": LayerObservation(state=LayerState.OBSERVED, data={"model_ref": "x"}),
            "generation": LayerObservation(state=LayerState.OBSERVED, data={"text": "hi", "ids": [1]}),
            "template": LayerObservation(state=LayerState.NOT_EXPOSED, note="no template"),
        },
    )
    sc = Scenario(
        name="t",
        messages=[{"role": "user", "content": "x"}],
        observe=["artifact", "template", "tokens", "generation"],
        parity_profile="strict",
    )
    cov = compute_coverage(left, right, scenario=sc, policy="strict")
    assert cov["required_coverage_percent"] < 100
    status, reasons = apply_coverage_to_status(ParityResult.PASS, cov)
    assert status in {
        ParityResult.PASS_WITH_LIMITED_COVERAGE,
        ParityResult.INCONCLUSIVE,
    }
    assert reasons


def test_diagnosis_confidence_scales_with_coverage():
    cov_hi = {
        "required_coverage_percent": 100.0,
        "meets_min_coverage": True,
    }
    cov_lo = {
        "required_coverage_percent": 25.0,
        "meets_min_coverage": False,
    }
    assert diagnosis_confidence(status=ParityResult.PASS, coverage=cov_hi) > diagnosis_confidence(
        status=ParityResult.PASS, coverage=cov_lo
    )


def test_format_timings_delta():
    info = format_timings({"transformers": 100.0, "vllm": 150.0})
    assert info["total_ms"] == 250.0
    assert info["delta_percent"] == 50.0
    assert len(info["entries"]) == 2


def test_policy_engine_separates_observation_from_compare():
    sc = Scenario(
        name="t",
        messages=[{"role": "user", "content": "hi"}],
        observe=["generation"],
        parity_profile="functional",
    )
    left = _trace(
        "a",
        {
            "template": LayerObservation(state=LayerState.NOT_EXPOSED, note="hidden"),
            "generation": LayerObservation(state=LayerState.OBSERVED, data={"text": "ok", "stop_reason": "stop"}),
        },
    )
    right = _trace(
        "b",
        {
            "template": LayerObservation(state=LayerState.NOT_SUPPORTED, note="n/a"),
            "generation": LayerObservation(state=LayerState.OBSERVED, data={"text": "ok", "stop_reason": "stop"}),
        },
    )
    eng = PolicyEngine(sc)
    # template not requested
    cmp_t = eng.compare_layer("template", left.layers["template"], right.layers["template"])
    assert cmp_t.result == ParityResult.NOT_REQUESTED
    assert cmp_t.baseline_state == LayerState.NOT_EXPOSED
    # generation compared
    cmp_g = eng.compare_layer("generation", left.layers["generation"], right.layers["generation"])
    assert cmp_g.result in {ParityResult.PASS, ParityResult.PASS_WITH_TOLERANCE}


def test_enrich_diagnosis_adds_verified_and_commands(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    d = result.diagnosis
    assert d.coverage is not None
    assert d.confidence is not None
    assert isinstance(d.verified_layers, list)
    assert isinstance(d.practical_commands, list)
    # full fake observe should be high coverage PASS-family
    assert d.status in {
        ParityResult.PASS,
        ParityResult.PASS_WITH_TOLERANCE,
        ParityResult.PASS_WITH_LIMITED_COVERAGE,
    }


def test_json_payload_has_coverage_fields(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    data = json.loads((result.path / "result.json").read_text(encoding="utf-8"))
    assert "coverage" in data
    assert "verified_layers" in data
    assert "confidence" in data
    assert "timings" in data
    assert "reproduction_command" in data
    assert "eleanity compare" in data["reproduction_command"]
    # diagnosis mirrors report fields
    diag = data["diagnosis"]
    assert "coverage" in diag
    assert "practical_commands" in diag


def test_inconclusive_when_nothing_comparable():
    left = _trace(
        "a",
        {"template": LayerObservation(state=LayerState.NOT_EXPOSED, note="x")},
    )
    right = _trace(
        "b",
        {"template": LayerObservation(state=LayerState.NOT_EXPOSED, note="y")},
    )
    d = diagnose([left, right])
    assert d.status in {ParityResult.INCONCLUSIVE, ParityResult.NOT_OBSERVABLE}
