from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, ObservationTrace


def _trace(fingerprint):
    return ObservationTrace(
        scenario_name="f",
        backend=fingerprint.backend_flags.get("runtime", "x"),
        artifact_fingerprint=fingerprint,
        layers={"artifact": LayerObservation(state=LayerState.OBSERVED, data=fingerprint.model_dump())},
    )


def test_tokenizer_revision_difference_is_an_artifact_divergence():
    a = ArtifactFingerprint(model_ref="m", revision="a", tokenizer="t")
    b = ArtifactFingerprint(model_ref="m", revision="b", tokenizer="t")
    assert diagnose([_trace(a), _trace(b)]).first_divergence == "artifact"
