from __future__ import annotations

import time
from typing import Any

from eleanity.fingerprints import collect_environment_fingerprint
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ObservationTrace,
    Scenario,
    TraceError,
)
from eleanity.utils.logging import get_logger, log_event

logger = get_logger("eleanity.observe")


def _tag(
    obs: LayerObservation,
    *,
    origin: str,
    origin_kind: str = "native",
) -> LayerObservation:
    return obs.model_copy(update={"origin": origin, "origin_kind": origin_kind})


def _unsupported(layer: str, note: str, *, origin: str) -> LayerObservation:
    return LayerObservation(
        state=LayerState.NOT_SUPPORTED,
        note=note,
        origin=origin,
        origin_kind="unavailable",
    )


def _not_exposed(layer: str, note: str, *, origin: str) -> LayerObservation:
    return LayerObservation(
        state=LayerState.NOT_EXPOSED,
        note=note,
        origin=origin,
        origin_kind="unavailable",
    )


def observe(
    adapter: Any,
    scenario: Scenario,
    model: str,
    baseline_backend: str | None = None,
) -> ObservationTrace:
    """Run the causal observation pipeline against one adapter.

    Order respects dependencies (template before tokens before logits) while
    remaining honest: missing capabilities become NOT_SUPPORTED / NOT_EXPOSED,
    never PASS.
    """

    started = time.perf_counter()
    requested = set(scenario.observe)
    name = getattr(adapter, "name", "adapter")
    environment = collect_environment_fingerprint()
    warnings: list[str] = []
    errors: list[TraceError] = []
    layers: dict[str, LayerObservation] = {}

    fingerprint = ArtifactFingerprint(model_ref=model)
    try:
        fingerprint = adapter.fingerprint(model)
        layers["artifact"] = LayerObservation(
            state=LayerState.OBSERVED,
            data=fingerprint.model_dump(mode="json"),
            origin=f"{name}.fingerprint",
            origin_kind="native",
        )
    except Exception as error:
        errors.append(TraceError(code="ARTIFACT_ERROR", message=str(error), layer="artifact"))
        layers["artifact"] = LayerObservation(
            state=LayerState.FAILED,
            data=fingerprint.model_dump(mode="json"),
            note=str(error),
            origin=f"{name}.fingerprint",
            origin_kind="native",
        )

    def _capture(layer: str, fn, *args, code: str = "LAYER_ERROR", origin: str = ""):
        try:
            result = fn(*args)
            if isinstance(result, LayerObservation):
                # Normalize legacy states
                if result.state == LayerState.ERROR:
                    result = result.model_copy(update={"state": LayerState.FAILED})
                if result.state == LayerState.NOT_OBSERVABLE:
                    # Prefer NOT_EXPOSED / NOT_SUPPORTED when note hints
                    note = (result.note or "").lower()
                    if "does not" in note or "unsupported" in note or "not implement" in note:
                        result = result.model_copy(update={"state": LayerState.NOT_SUPPORTED})
                    else:
                        result = result.model_copy(update={"state": LayerState.NOT_EXPOSED})
                if not result.origin:
                    result = _tag(result, origin=origin or f"{name}.{layer}", origin_kind=result.origin_kind or "native")
                return result
            return result
        except Exception as error:
            errors.append(TraceError(code=code, message=str(error), layer=layer))
            return LayerObservation(
                state=LayerState.FAILED,
                note=str(error),
                origin=origin or f"{name}.{layer}",
                origin_kind="native",
            )

    if "special_tokens" in requested:
        special = getattr(adapter, "special_tokens", None)
        if callable(special):
            layers["special_tokens"] = _capture(
                "special_tokens", special, code="SPECIAL_TOKENS_ERROR", origin=f"{name}.special_tokens"
            )
        else:
            layers["special_tokens"] = _unsupported(
                "special_tokens",
                "adapter does not implement special_tokens()",
                origin=f"{name}.special_tokens",
            )

    rendered = None
    if requested & {"template", "tokens", "logits"}:
        rendered = _capture("template", adapter.render, scenario, code="RENDER_ERROR", origin=f"{name}.render")
        if "template" in requested:
            layers["template"] = rendered
            if rendered.state not in {LayerState.OBSERVED, LayerState.INFERRED}:
                warnings.append(f"{name}: template {rendered.state.value} — {rendered.note}")

    tokens = None
    if requested & {"tokens", "logits"}:
        if rendered and rendered.state == LayerState.OBSERVED:
            text = rendered.data.get("text") or rendered.data.get("rendered_text") or ""
            tokens = _capture("tokens", adapter.tokenize, text, code="TOKENIZE_ERROR", origin=f"{name}.tokenize")
        else:
            tokens = LayerObservation(
                state=LayerState.NOT_EXPOSED,
                note="template unavailable — cannot tokenize",
                origin=f"{name}.tokenize",
                origin_kind="unavailable",
            )
        if "tokens" in requested:
            layers["tokens"] = tokens
            if tokens.state not in {LayerState.OBSERVED, LayerState.INFERRED}:
                warnings.append(f"{name}: tokens {tokens.state.value} — {tokens.note}")

    if "logits" in requested:
        layers["logits"] = _capture(
            "logits",
            adapter.forward,
            tokens or LayerObservation(state=LayerState.NOT_EXPOSED, note="tokens unavailable"),
            code="FORWARD_ERROR",
            origin=f"{name}.forward",
        )
        if layers["logits"].state not in {LayerState.OBSERVED}:
            warnings.append(f"{name}: logits {layers['logits'].state.value} — {layers['logits'].note}")

    if "generation" in requested:
        layers["generation"] = _capture(
            "generation", adapter.generate, scenario, code="GENERATION_ERROR", origin=f"{name}.generate"
        )
        if layers["generation"].state not in {LayerState.OBSERVED}:
            warnings.append(
                f"{name}: generation {layers['generation'].state.value} — {layers['generation'].note}"
            )

    if "structured" in requested:
        structured = getattr(adapter, "structured", None)
        if callable(structured):
            layers["structured"] = _capture(
                "structured", structured, scenario, code="STRUCTURED_ERROR", origin=f"{name}.structured"
            )
        else:
            layers["structured"] = _unsupported(
                "structured", "adapter does not implement structured()", origin=f"{name}.structured"
            )

    if "streaming" in requested:
        stream = getattr(adapter, "stream_generate", None)
        if callable(stream):
            layers["streaming"] = _capture(
                "streaming", stream, scenario, code="STREAM_ERROR", origin=f"{name}.stream_generate"
            )
        else:
            layers["streaming"] = _unsupported(
                "streaming", "adapter does not implement stream_generate()", origin=f"{name}.stream"
            )

    if "api" in requested:
        api = getattr(adapter, "api_probe", None)
        if callable(api):
            layers["api"] = _capture("api", api, scenario, code="API_ERROR", origin=f"{name}.api_probe")
        else:
            layers["api"] = _unsupported(
                "api", "adapter does not implement api_probe()", origin=f"{name}.api_probe"
            )

    duration_ms = (time.perf_counter() - started) * 1000
    log_event(
        logger,
        "observe_done",
        backend=name,
        model=model,
        duration_ms=round(duration_ms, 2),
        layers=",".join(layers),
        errors=len(errors),
    )
    return ObservationTrace(
        scenario_name=scenario.name,
        backend=name,
        baseline_backend=baseline_backend,
        artifact_fingerprint=fingerprint,
        environment=environment,
        layers=layers,
        warnings=warnings,
        errors=errors,
        duration_ms=round(duration_ms, 3),
    )
