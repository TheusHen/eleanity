from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eleanity.adapters.sdk import check_adapter_compliance
from eleanity.models.schemas import LayerState, Message, Scenario


@dataclass
class CertificationReport:
    runtime: str
    passed: bool
    level: str  # none | bronze | silver | gold
    checks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "passed": self.passed,
            "level": self.level,
            "checks": self.checks,
            "notes": self.notes,
        }


def certify_runtime(adapter: Any, *, model: str = "cert-demo") -> CertificationReport:
    """Certify an adapter as Eleanity-compatible at bronze/silver/gold.

    Bronze: compliance + honest N/O
    Silver: template + tokens observable
    Gold: generation or structured/api also observable
    """

    checks: list[dict[str, Any]] = []
    notes: list[str] = []
    compliance = check_adapter_compliance(adapter, model=model)
    checks.append({"name": "sdk_compliance", "passed": compliance.passed, "detail": compliance.to_dict()})

    scenario = Scenario(
        name="cert",
        messages=[Message(role="user", content="ping")],
        parameters={"max_tokens": 1, "temperature": 0},
        observe=["template", "tokens", "generation"],
    )
    template_ok = False
    tokens_ok = False
    generation_ok = False
    try:
        rendered = adapter.render(scenario)
        template_ok = getattr(rendered, "state", None) == LayerState.OBSERVED
        checks.append(
            {"name": "template_observable", "passed": template_ok, "state": str(getattr(rendered, "state", None))}
        )
        text = ""
        if template_ok:
            text = str(rendered.data.get("text") or rendered.data.get("rendered_text") or "")
        tokens = adapter.tokenize(text or "ping")
        tokens_ok = getattr(tokens, "state", None) == LayerState.OBSERVED
        checks.append({"name": "tokens_observable", "passed": tokens_ok, "state": str(getattr(tokens, "state", None))})
        gen = adapter.generate(scenario)
        generation_ok = getattr(gen, "state", None) == LayerState.OBSERVED
        checks.append(
            {"name": "generation_observable", "passed": generation_ok, "state": str(getattr(gen, "state", None))}
        )
    except Exception as error:
        notes.append(f"runtime probe error: {error}")
        checks.append({"name": "runtime_probe", "passed": False, "error": str(error)})

    level = "none"
    if compliance.passed:
        level = "bronze"
        if template_ok and tokens_ok:
            level = "silver"
            if generation_ok:
                level = "gold"
    passed = level in {"bronze", "silver", "gold"}
    if not template_ok:
        notes.append("Template not observable — chat parity gates will be limited.")
    if not tokens_ok:
        notes.append("Tokenization not observable — token gates unavailable.")
    return CertificationReport(
        runtime=str(getattr(adapter, "name", type(adapter).__name__)),
        passed=passed,
        level=level,
        checks=checks,
        notes=notes,
    )
