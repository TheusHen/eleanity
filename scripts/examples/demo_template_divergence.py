#!/usr/bin/env python3
"""Reproducible first-divergence demo: missing assistant generation prompt.

Simulates two backends on the same model/scenario where one omits the
assistant-turn suffix (a common Transformers vs server template bug).
"""

from __future__ import annotations

from eleanity.adapters.fake import FakeAdapter
from eleanity.core.observe import observe
from eleanity.diagnosers import diagnose
from eleanity.models.schemas import (
    GenerationConfig,
    LayerObservation,
    LayerState,
    PromptObservation,
    Scenario,
)
from eleanity.policies.engine import PolicyEngine
from eleanity.utils.hashing import text_sha256


class MissingAssistantTurnAdapter(FakeAdapter):
    """Backend that does not append the assistant generation prompt."""

    name = "candidate-no-agp"

    def render(self, scenario: Scenario) -> LayerObservation:
        text = "\n".join(f"{message.role}: {message.content}" for message in scenario.messages)
        # Deliberately omit the trailing "assistant:" turn marker.
        obs = PromptObservation(
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
            data=obs.to_layer_data(),
            origin="candidate-no-agp.render",
            origin_kind="native",
        )


def main() -> None:
    scenario = Scenario(
        name="missing-assistant-turn",
        messages=[{"role": "user", "content": "Hello"}],
        parameters={"temperature": 0, "max_tokens": 8, "seed": 42},
        generation=GenerationConfig(add_generation_prompt=True),
        observe=["artifact", "template", "special_tokens", "tokens", "generation"],
        parity_profile="strict",
    )
    baseline = observe(FakeAdapter(), scenario, "org/demo-model")
    candidate = observe(MissingAssistantTurnAdapter(), scenario, "org/demo-model")
    diagnosis = diagnose([baseline, candidate])
    comparisons = PolicyEngine(scenario).compare_layers(baseline, candidate)
    template = comparisons["template"]

    print("=== Eleanity first-divergence demo ===")
    print("model:     org/demo-model")
    print("baseline:  fake (add_generation_prompt=true)")
    print("candidate: candidate-no-agp (omits assistant turn)")
    print("policy:    strict")
    print()
    print(f"status:            {diagnosis.status.value}")
    print(f"first_divergence:  {diagnosis.first_divergence}")
    if diagnosis.first_divergence_detail:
        loc = diagnosis.first_divergence_detail.location
        print(f"character:         {loc.character}")
        print(f"byte:              {loc.byte}")
        print(f'baseline_snippet:  "{diagnosis.first_divergence_detail.baseline}"')
        print(f'candidate_snippet: "{diagnosis.first_divergence_detail.candidate}"')
    print()
    print("baseline template:")
    print(repr(baseline.layers["template"].data.get("text")))
    print("candidate template:")
    print(repr(candidate.layers["template"].data.get("text")))
    print()
    print(f"template comparison: {template.result.value}")
    print(f"first_character:     {template.details.get('first_character')}")
    print(f"first_byte:          {template.details.get('first_byte')}")
    if diagnosis.probable_causes:
        cause = diagnosis.probable_causes[0]
        print()
        print(f"probable_cause: [{cause.code}] conf={cause.confidence}")
        print(f"  {cause.message}")
    print()
    print(f"summary: {diagnosis.summary}")
    raise SystemExit(1 if diagnosis.status.value == "DIVERGENT" else 0)


if __name__ == "__main__":
    main()
