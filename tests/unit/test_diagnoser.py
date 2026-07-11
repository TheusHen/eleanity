from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, ObservationTrace


def trace(name, template, tokens):
    return ObservationTrace(
        trace_id=name,
        scenario_name="demo",
        backend=name,
        artifact_fingerprint=ArtifactFingerprint(model_ref="m"),
        layers={
            "artifact": LayerObservation(state=LayerState.OBSERVED, data={"model_ref": "m"}),
            "template": LayerObservation(state=LayerState.OBSERVED, data={"text": template}),
            "tokens": LayerObservation(state=LayerState.OBSERVED, data={"ids": tokens}),
        },
    )


def test_diagnoser_finds_template_before_tokens():
    report = diagnose([trace("a", "abc", [1, 2, 3]), trace("b", "axc", [1, 9, 8])])
    assert report.first_divergence == "template"
    assert "character 1" in report.summary
    assert report.propagation_percent > 0
