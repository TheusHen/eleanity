from eleanity.diagnosers.first_divergence import diagnose
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ObservationTrace,
    ParityResult,
)


def _trace(backend: str, template: str, tokens: list[int], **artifact):
    data = {"model_ref": "m", **artifact}
    return ObservationTrace(
        scenario_name="demo",
        backend=backend,
        artifact_fingerprint=ArtifactFingerprint(model_ref="m"),
        layers={
            "artifact": LayerObservation(state=LayerState.OBSERVED, data=data),
            "template": LayerObservation(
                state=LayerState.OBSERVED,
                data={
                    "text": template,
                    "add_generation_prompt": True,
                    "chat_template_hash": "x",
                },
            ),
            "tokens": LayerObservation(state=LayerState.OBSERVED, data={"ids": tokens}),
        },
    )


def test_missing_assistant_turn_rule():
    a = _trace(
        "transformers",
        "<|im_start|>user\nOi<|im_end|>\n<|im_start|>assistant\n",
        [1, 2, 3, 4],
    )
    b = _trace(
        "llamacpp",
        "<|im_start|>user\nOi<|im_end|>\n",
        [1, 2],
    )
    report = diagnose([a, b])
    assert report.status == ParityResult.DIVERGENT
    assert report.first_divergence == "template"
    assert report.probable_causes
    assert report.probable_causes[0].code == "MISSING_ASSISTANT_TURN_TOKEN"
    assert report.suggested_actions
    assert report.first_divergence_detail is not None
    assert report.first_divergence_detail.location.character is not None


def test_artifact_revision_rule():
    a = _trace("a", "same", [1], revision="main")
    b = _trace("b", "same", [1], revision="other")
    report = diagnose([a, b])
    assert report.first_divergence == "artifact"
    assert any(c.code == "REVISION_DIFFERENT" for c in report.probable_causes)
