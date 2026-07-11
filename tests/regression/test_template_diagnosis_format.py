from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, ObservationTrace


def _trace(name, template, ids):
    return ObservationTrace(
        trace_id=name,
        scenario_name="template-fire-test",
        backend=name,
        artifact_fingerprint=ArtifactFingerprint(model_ref="Qwen/Qwen2.5-0.5B-Instruct"),
        layers={
            "artifact": LayerObservation(state=LayerState.OBSERVED, data={"model_ref": "Qwen/Qwen2.5-0.5B-Instruct"}),
            "template": LayerObservation(state=LayerState.OBSERVED, data={"text": template}),
            "tokens": LayerObservation(state=LayerState.OBSERVED, data={"ids": ids}),
        },
    )


def test_template_diagnosis_is_specific_and_numeric():
    report = diagnose([
        _trace("with-system", "<|im_start|>system\nYou help.<|im_end|>\n<|im_start|>user\nOi", [1, 2, 3, 4]),
        _trace("without-system", "<|im_start|>user\nOi", [1, 9, 8, 7]),
    ])
    assert report.first_divergence == "template"
    assert "character 12" in report.summary
    assert "from index 1" in report.summary
    assert "Likely cause:" in report.summary
