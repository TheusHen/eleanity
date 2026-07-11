"""Tests for formal parity spec, impact, capsule, stabilize, trace v1, capture."""

from __future__ import annotations

import json
from pathlib import Path

from eleanity.core.capture import capture_openai_jsonl
from eleanity.core.engine import CompareEngine
from eleanity.core.stabilize import stabilize_backend
from eleanity.models.schemas import ParityProfile, ParityResult
from eleanity.spec.capsule import build_execution_capsule
from eleanity.spec.impact import FunctionalImpact, assess_impact
from eleanity.spec.layers import expand_observe_layers, to_coarse_observe
from eleanity.spec.observability import (
    ObservationState,
    comparison_outcome_for_states,
    honesty_rule,
)
from eleanity.spec.parity import (
    FormalParityStatus,
    apply_numerical_thresholds,
    apply_prefix_thresholds,
    formal_status_from_parity,
    policy_comparator_set,
    status_definition,
)
from eleanity.spec.trace_v1 import build_trace_document, validate_trace_document


def test_status_definitions_complete():
    for status in FormalParityStatus:
        defn = status_definition(status)
        assert defn["meaning"]
        assert "never" in defn


def test_quantized_policy_has_formal_thresholds():
    spec = policy_comparator_set(ParityProfile.QUANTIZED)
    logits = spec.comparators["prefill_logits"]
    assert logits.mode == "numerical"
    assert logits.atol == 1.0e-4
    assert logits.top_k_agreement == 0.99
    gen = spec.comparators["generated_token_ids"]
    assert gen.mode == "prefix"
    assert gen.exact_prefix_tokens == 16


def test_numerical_thresholds_pass_with_tolerance():
    from eleanity.spec.parity import ComparatorSpec

    spec = ComparatorSpec(mode="numerical", atol=1e-3, rtol=1e-2, top_k_agreement=0.99)
    status = apply_numerical_thresholds(
        max_abs_diff=5e-4, max_rel_diff=1e-3, top_k_agreement=0.995, spec=spec
    )
    assert status == FormalParityStatus.PASS_WITH_TOLERANCE


def test_prefix_thresholds():
    from eleanity.spec.parity import ComparatorSpec

    spec = ComparatorSpec(mode="prefix", exact_prefix_tokens=4)
    assert apply_prefix_thresholds(4, 10, 12, spec) == FormalParityStatus.PASS_WITH_TOLERANCE
    assert apply_prefix_thresholds(2, 10, 12, spec) == FormalParityStatus.DIVERGENT


def test_legacy_status_mapping():
    assert formal_status_from_parity(ParityResult.NOT_OBSERVABLE) == FormalParityStatus.INCONCLUSIVE
    assert formal_status_from_parity(ParityResult.INCOMPARABLE) == FormalParityStatus.INCONCLUSIVE


def test_observability_never_pass_on_missing():
    assert comparison_outcome_for_states(
        ObservationState.NOT_EXPOSED, ObservationState.OBSERVED
    ) == FormalParityStatus.INCONCLUSIVE
    assert comparison_outcome_for_states(
        ObservationState.UNSUPPORTED, ObservationState.OBSERVED
    ) == FormalParityStatus.UNSUPPORTED
    assert comparison_outcome_for_states(
        ObservationState.OBSERVED, ObservationState.OBSERVED
    ) is None
    assert "Never treat" in honesty_rule()


def test_layer_expansion():
    fine = expand_observe_layers(["generation", "tokens"])
    assert "generated_token_ids" in fine
    assert "input_token_ids" in fine
    coarse = to_coarse_observe(["prefill_logits", "chat_template"])
    assert "logits" in coarse
    assert "template" in coarse


def test_impact_none_when_tokens_equal_despite_divergent_parity(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    impact = assess_impact(
        parity_status=ParityResult.PASS,
        first_divergence=None,
        left=result.traces[0],
        right=result.traces[1],
    )
    assert impact.impact == FunctionalImpact.NONE


def test_capsule_sealed():
    from eleanity.models.schemas import Scenario

    sc = Scenario(name="c", messages=[{"role": "user", "content": "hi"}], parameters={"seed": 1})
    cap = build_execution_capsule(backend="fake", model="demo", scenario=sc)
    assert cap.capsule_hash
    assert cap.generation.seed == 1
    assert cap.runtime.name == "fake"


def test_engine_writes_trace_v1_and_impact(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    assert result.diagnosis.formal_status in {"PASS", "PASS_WITH_TOLERANCE"}
    assert result.diagnosis.impact is not None
    assert result.diagnosis.impact["impact"] == "NONE"
    trace_path = result.path / "trace.v1.json"
    assert trace_path.is_file()
    doc = json.loads(trace_path.read_text(encoding="utf-8"))
    assert doc["schema_version"].startswith("1.")
    assert "subjects" in doc
    assert doc["parity"]["status"]
    errors = validate_trace_document(doc)
    assert errors == []


def test_stabilize_fake(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    report = stabilize_backend(engine, "demo", "fake", repetitions=3)
    assert report.rate == 1.0
    assert report.self_consistent


def test_capture_openai_jsonl(tmp_path):
    src = tmp_path / "traffic.jsonl"
    src.write_text(
        json.dumps(
            {
                "request": {
                    "model": "gpt-x",
                    "messages": [{"role": "user", "content": "secret prompt"}],
                    "temperature": 0,
                    "max_tokens": 8,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "suite"
    manifest = capture_openai_jsonl(src, out, redact=True, sample=10)
    assert manifest["scenarios"] == 1
    text = Path(manifest["path"]).read_text(encoding="utf-8")
    assert "secret prompt" not in text
    assert "redacted" in text.lower() or "sha256" in text


def test_build_trace_document_from_result(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    payload = json.loads((result.path / "result.json").read_text(encoding="utf-8"))
    doc = build_trace_document(run_id=result.run_id, result_payload=payload)
    assert doc["run_id"] == result.run_id
    assert "document_hash" in doc
