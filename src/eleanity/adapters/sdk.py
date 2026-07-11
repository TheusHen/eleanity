from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eleanity.adapters.base import BackendAdapter, CapabilitySet, HealthcheckResult
from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, Message, Scenario

REQUIRED_METHODS = (
    "fingerprint",
    "render",
    "tokenize",
    "forward",
    "generate",
)

OPTIONAL_METHODS = (
    "special_tokens",
    "stream_generate",
    "structured",
    "api_probe",
    "healthcheck",
)


@dataclass
class ComplianceIssue:
    code: str
    message: str
    severity: str = "error"  # error | warning


@dataclass
class ComplianceReport:
    adapter_name: str
    passed: bool
    issues: list[ComplianceIssue] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "passed": self.passed,
            "issues": [issue.__dict__ for issue in self.issues],
            "capabilities": self.capabilities,
        }


def check_adapter_compliance(
    adapter: Any,
    *,
    scenario: Scenario | None = None,
    model: str = "compliance-demo",
    probe_runtime: bool = True,
) -> ComplianceReport:
    """Adapter SDK compliance checks for plugins and builtins.

    Does not require GPU. Optional methods may return NOT_OBSERVABLE.
    Required methods must exist and must not invent PASS when unavailable —
    they should return LayerObservation with an honest state.
    """

    issues: list[ComplianceIssue] = []
    name = getattr(adapter, "name", type(adapter).__name__)

    for method in REQUIRED_METHODS:
        if not callable(getattr(adapter, method, None)):
            issues.append(ComplianceIssue("MISSING_REQUIRED_METHOD", f"missing {method}()"))

    caps = getattr(adapter, "capabilities", None)
    if caps is None:
        issues.append(ComplianceIssue("MISSING_CAPABILITIES", "adapter.capabilities is required"))
        cap_dict: dict[str, Any] = {}
    else:
        cap_dict = caps.model_dump(mode="json") if hasattr(caps, "model_dump") else dict(caps)

    scenario = scenario or Scenario(
        name="sdk-compliance",
        messages=[Message(role="user", content="hello")],
        parameters={"max_tokens": 1, "temperature": 0},
        observe=["template", "tokens", "generation"],
    )

    if probe_runtime and not any(i.code == "MISSING_REQUIRED_METHOD" for i in issues):
        try:
            fp = adapter.fingerprint(model)
            if not isinstance(fp, ArtifactFingerprint) and not hasattr(fp, "model_ref"):
                issues.append(ComplianceIssue("FINGERPRINT_TYPE", "fingerprint() must return ArtifactFingerprint-like"))
        except Exception as error:
            issues.append(ComplianceIssue("FINGERPRINT_ERROR", str(error), severity="warning"))

        try:
            rendered = adapter.render(scenario)
            _assert_observation(rendered, "render", issues)
            if rendered.state == LayerState.OBSERVED:
                tokens = adapter.tokenize(str(rendered.data.get("text") or rendered.data.get("rendered_text") or ""))
                _assert_observation(tokens, "tokenize", issues)
            else:
                tokens = adapter.tokenize("hello")
                _assert_observation(tokens, "tokenize", issues)
            logits = adapter.forward(
                tokens if isinstance(tokens, LayerObservation) else LayerObservation(state=LayerState.NOT_OBSERVABLE)
            )
            _assert_observation(logits, "forward", issues)
            gen = adapter.generate(scenario)
            _assert_observation(gen, "generate", issues)
            # Honesty: if capability is False, method should not claim OBSERVED without data
            if caps is not None:
                if not getattr(caps, "logits", True) and logits.state == LayerState.OBSERVED and not logits.data:
                    issues.append(
                        ComplianceIssue("FALSE_LOGITS", "logits capability false but OBSERVED empty data", "warning")
                    )
                if not getattr(caps, "generation", True) and gen.state == LayerState.OBSERVED and not gen.data:
                    issues.append(
                        ComplianceIssue("FALSE_GENERATION", "generation capability false but OBSERVED empty", "warning")
                    )
        except Exception as error:
            issues.append(ComplianceIssue("RUNTIME_PROBE_ERROR", str(error), severity="warning"))

        for method in OPTIONAL_METHODS:
            fn = getattr(adapter, method, None)
            if not callable(fn):
                continue
            try:
                if method == "healthcheck":
                    result = fn()
                    if not isinstance(result, HealthcheckResult) and not hasattr(result, "ok"):
                        issues.append(
                            ComplianceIssue(
                                "HEALTHCHECK_TYPE", "healthcheck() should return HealthcheckResult-like", "warning"
                            )
                        )
                elif method == "special_tokens":
                    _assert_observation(fn(), method, issues)
                else:
                    _assert_observation(fn(scenario), method, issues)
            except TypeError:
                # Some optionals take no scenario
                try:
                    _assert_observation(fn(), method, issues)
                except Exception as error:
                    issues.append(ComplianceIssue("OPTIONAL_ERROR", f"{method}: {error}", "warning"))
            except Exception as error:
                issues.append(ComplianceIssue("OPTIONAL_ERROR", f"{method}: {error}", "warning"))

    errors = [i for i in issues if i.severity == "error"]
    return ComplianceReport(adapter_name=str(name), passed=not errors, issues=issues, capabilities=cap_dict)


def _assert_observation(value: Any, method: str, issues: list[ComplianceIssue]) -> None:
    if not isinstance(value, LayerObservation) and not hasattr(value, "state"):
        issues.append(ComplianceIssue("OBSERVATION_TYPE", f"{method}() must return LayerObservation-like"))
        return
    state = value.state if hasattr(value, "state") else None
    if state is None:
        issues.append(ComplianceIssue("OBSERVATION_STATE", f"{method}() missing state"))
        return
    # Accept full observation taxonomy (availability states, not comparison results)
    allowed = set(LayerState)
    if state not in allowed and str(getattr(state, "value", state)) not in {s.value for s in allowed}:
        issues.append(ComplianceIssue("OBSERVATION_STATE_VALUE", f"{method}() invalid state {state}"))


def make_minimal_adapter(
    name: str,
    *,
    render_fn: Callable[[Scenario], LayerObservation] | None = None,
    tokenize_fn: Callable[[str], LayerObservation] | None = None,
) -> type[BackendAdapter]:
    """Factory helper for plugin authors to scaffold a compliant adapter class."""

    class _PluginAdapter(BackendAdapter):
        def __init__(self, model_ref: str, model_spec=None):
            self.model_ref = model_ref
            self.name = name
            self.capabilities = CapabilitySet(
                render=render_fn is not None,
                tokenize=tokenize_fn is not None,
                generation=False,
                artifact=True,
            )

        def fingerprint(self, model_ref: str) -> ArtifactFingerprint:
            return ArtifactFingerprint(model_ref=model_ref, backend_flags={"runtime": name})

        def render(self, scenario: Scenario) -> LayerObservation:
            if render_fn:
                return render_fn(scenario)
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note=f"{name}: render N/O")

        def tokenize(self, rendered: str) -> LayerObservation:
            if tokenize_fn:
                return tokenize_fn(rendered)
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note=f"{name}: tokenize N/O")

        def forward(self, tokens: LayerObservation) -> LayerObservation:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note=f"{name}: logits N/O")

        def generate(self, scenario: Scenario) -> LayerObservation:
            return LayerObservation(state=LayerState.NOT_OBSERVABLE, note=f"{name}: generation N/O")

    _PluginAdapter.name = name
    return _PluginAdapter
