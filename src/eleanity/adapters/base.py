from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic import BaseModel, Field

from eleanity.models.schemas import ArtifactFingerprint, LayerObservation, LayerState, Scenario


class CapabilitySet(BaseModel):
    """Explicit layer capabilities declared by each backend adapter.

    Missing capabilities must surface as NOT_OBSERVABLE, never as a hard failure.
    """

    render: bool = False
    tokenize: bool = False
    logits: bool = False
    stream: bool = False
    tools: bool = False

    artifact: bool = True
    template: bool = False
    rendered_prompt: bool = False
    tokenization: bool = False
    special_tokens: bool = False
    logprobs: bool = False
    generation: bool = False
    structured_output: bool = False
    streaming: bool = False
    usage: bool = False
    errors: bool = False
    cancellation: bool = False
    healthcheck: bool = False

    notes: dict[str, str] = Field(default_factory=dict)


class HealthcheckResult(BaseModel):
    ok: bool
    detail: str
    latency_ms: float | None = None
    endpoint: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BackendAdapter(ABC):
    """Production adapter contract.

    Subclasses must implement fingerprint/render/tokenize/forward/generate.
    Optional hooks (special_tokens, stream, structured, api_probe, healthcheck)
    default to NOT_OBSERVABLE so partial backends stay honest.
    """

    name: str
    version: str = "0"
    capabilities: CapabilitySet

    @abstractmethod
    def fingerprint(self, model_ref: str) -> ArtifactFingerprint: ...

    @abstractmethod
    def render(self, scenario: Scenario) -> LayerObservation: ...

    @abstractmethod
    def tokenize(self, rendered: str) -> LayerObservation: ...

    @abstractmethod
    def forward(self, tokens: LayerObservation) -> LayerObservation: ...

    @abstractmethod
    def generate(self, scenario: Scenario) -> LayerObservation: ...

    def special_tokens(self) -> LayerObservation:
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note=f"{self.name} does not expose special_tokens()",
        )

    def stream_generate(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note=f"{self.name} does not expose streaming observations",
        )

    def structured(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note=f"{self.name} does not expose structured output observations",
        )

    def api_probe(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note=f"{self.name} does not expose API contract observations",
        )

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(ok=False, detail=f"{self.name} healthcheck not implemented")

    def not_observable(self, layer: str, reason: str) -> LayerObservation:
        return LayerObservation(state=LayerState.NOT_OBSERVABLE, note=reason)

    def error(self, layer: str, message: str) -> LayerObservation:
        return LayerObservation(state=LayerState.ERROR, note=message)


class BackendAdapterProtocol(Protocol):
    """Structural protocol kept for duck-typed fakes and plugins."""

    name: str
    capabilities: CapabilitySet

    def fingerprint(self, model_ref: str) -> ArtifactFingerprint: ...

    def render(self, scenario: Scenario) -> LayerObservation: ...

    def tokenize(self, rendered: str) -> LayerObservation: ...

    def forward(self, tokens: LayerObservation) -> LayerObservation: ...

    def generate(self, scenario: Scenario) -> LayerObservation: ...
