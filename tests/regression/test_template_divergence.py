from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, ObservationTrace


def test_synthetic_missing_assistant_turn_regression():
    base = dict(scenario_name="synthetic-missing-assistant-turn", artifact_fingerprint=ArtifactFingerprint(model_ref="Qwen/Qwen2.5-0.5B-Instruct"))
    a = ObservationTrace(trace_id="a", backend="transformers", layers={"template": LayerObservation(state=LayerState.OBSERVED, data={"text": "<|im_start|>user\\nOi<|im_end|>\\n<|im_start|>assistant\\n"}), "tokens": LayerObservation(state=LayerState.OBSERVED, data={"ids": [1, 2, 3, 4]})}, **base)
    b = ObservationTrace(trace_id="b", backend="synthetic", layers={"template": LayerObservation(state=LayerState.OBSERVED, data={"text": "<|im_start|>user\\nOi<|im_end|>\\n"}), "tokens": LayerObservation(state=LayerState.OBSERVED, data={"ids": [1, 2]})}, **base)
    assert diagnose([a, b]).first_divergence == "template"
