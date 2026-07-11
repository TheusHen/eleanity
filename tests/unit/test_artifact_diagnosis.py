from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, ObservationTrace


def test_backend_specific_fingerprint_flags_do_not_create_false_artifact_divergence():
    common = {"model_ref": "same", "revision": "main"}
    a = ObservationTrace(
        scenario_name="x",
        backend="a",
        artifact_fingerprint=ArtifactFingerprint(**common, backend_flags={"runtime": "a"}),
        layers={
            "artifact": LayerObservation(
                state=LayerState.OBSERVED,
                data=ArtifactFingerprint(**common, backend_flags={"runtime": "a"}).model_dump(),
            ),
            "template": LayerObservation(state=LayerState.OBSERVED, data={"text": "x"}),
        },
    )
    b = ObservationTrace(
        scenario_name="x",
        backend="b",
        artifact_fingerprint=ArtifactFingerprint(**common, backend_flags={"runtime": "b"}),
        layers={
            "artifact": LayerObservation(
                state=LayerState.OBSERVED,
                data=ArtifactFingerprint(**common, backend_flags={"runtime": "b"}).model_dump(),
            ),
            "template": LayerObservation(state=LayerState.OBSERVED, data={"text": "x"}),
        },
    )
    assert diagnose([a, b]).first_divergence is None
