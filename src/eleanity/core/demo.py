from __future__ import annotations

from typing import Any

from eleanity.adapters.fake import FakeAdapter
from eleanity.core.observe import observe
from eleanity.diagnosers import diagnose
from eleanity.models.schemas import (
    GenerationConfig,
    LayerObservation,
    LayerState,
    Message,
    ParityProfile,
    PromptObservation,
    Scenario,
)
from eleanity.policies.engine import PolicyEngine
from eleanity.utils.hashing import text_sha256


class MissingAssistantTurnAdapter(FakeAdapter):
    """Deterministic candidate that omits the assistant generation marker."""

    name = "candidate-no-assistant-turn"

    def render(self, scenario: Scenario) -> LayerObservation:
        text = "\n".join(f"{message.role}: {message.content}" for message in scenario.messages)
        prompt = PromptObservation(
            chat_template_source="fake://role-colon",
            chat_template_hash=text_sha256(text),
            add_generation_prompt=False,
            continue_final_message=False,
            rendered_text=text,
            text=text,
            roles=[message.role for message in scenario.messages],
        )
        return LayerObservation(
            state=LayerState.OBSERVED,
            data=prompt.to_layer_data(),
            origin="candidate-no-assistant-turn.render",
            origin_kind="native",
        )


def run_template_divergence_demo() -> dict[str, Any]:
    """Run the offline first-divergence demonstration used for onboarding."""

    scenario = Scenario(
        name="missing-assistant-turn",
        messages=[Message(role="user", content="Hello")],
        parameters={"temperature": 0, "max_tokens": 8, "seed": 42},
        generation=GenerationConfig(add_generation_prompt=True),
        observe=["artifact", "template", "special_tokens", "tokens", "generation"],
        parity_profile=ParityProfile.STRICT,
    )
    baseline = observe(FakeAdapter(), scenario, "org/demo-model")
    candidate = observe(MissingAssistantTurnAdapter(), scenario, "org/demo-model")
    diagnosis = diagnose([baseline, candidate])
    template = PolicyEngine(scenario).compare_layers(baseline, candidate)["template"]
    cause = diagnosis.probable_causes[0] if diagnosis.probable_causes else None
    location = diagnosis.first_divergence_detail.location if diagnosis.first_divergence_detail else None

    return {
        "status": diagnosis.status.value,
        "first_divergence": diagnosis.first_divergence,
        "character": location.character if location else None,
        "byte": location.byte if location else None,
        "baseline_template": baseline.layers["template"].data.get("text"),
        "candidate_template": candidate.layers["template"].data.get("text"),
        "template_result": template.result.value,
        "probable_cause": cause.code if cause else None,
        "confidence": cause.confidence if cause else None,
        "summary": diagnosis.summary,
    }


def render_template_divergence_demo(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Eleanity offline divergence demo",
            "model:             org/demo-model",
            "baseline:          fake (assistant generation marker enabled)",
            "candidate:         fake (assistant generation marker missing)",
            f"status:            {result['status']}",
            f"first_divergence:  {result['first_divergence']}",
            f"character:         {result['character']}",
            f"baseline_template: {result['baseline_template']!r}",
            f"candidate_template:{result['candidate_template']!r}",
            f"probable_cause:    {result['probable_cause']} (confidence={result['confidence']})",
            f"summary:           {result['summary']}",
        ]
    )
