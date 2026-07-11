"""Explicit observability model — missing data is never equality."""

from __future__ import annotations

from enum import StrEnum

from eleanity.models.schemas import LayerState, ParityResult
from eleanity.spec.parity import FormalParityStatus


class ObservationState(StrEnum):
    """State of a single layer observation on one subject."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    NOT_EXPOSED = "NOT_EXPOSED"
    UNSUPPORTED = "UNSUPPORTED"
    REDACTED = "REDACTED"
    FAILED = "FAILED"
    # Legacy aliases kept for reading old traces
    NOT_OBSERVABLE = "NOT_OBSERVABLE"
    INCOMPARABLE = "INCOMPARABLE"
    ERROR = "ERROR"


# What formal comparison outcome is allowed when either side is not OBSERVED.
_NON_OBSERVED_OUTCOME: dict[ObservationState, FormalParityStatus] = {
    ObservationState.INFERRED: FormalParityStatus.INCONCLUSIVE,
    ObservationState.NOT_EXPOSED: FormalParityStatus.INCONCLUSIVE,
    ObservationState.NOT_OBSERVABLE: FormalParityStatus.INCONCLUSIVE,
    ObservationState.UNSUPPORTED: FormalParityStatus.UNSUPPORTED,
    ObservationState.REDACTED: FormalParityStatus.INCONCLUSIVE,
    ObservationState.FAILED: FormalParityStatus.ERROR,
    ObservationState.ERROR: FormalParityStatus.ERROR,
    ObservationState.INCOMPARABLE: FormalParityStatus.INCONCLUSIVE,
}


def normalize_observation_state(value: str | LayerState | ObservationState) -> ObservationState:
    if isinstance(value, ObservationState):
        return value
    raw = value.value if isinstance(value, LayerState) else str(value)
    # Map legacy LayerState
    if raw == LayerState.ERROR.value or raw == "ERROR":
        return ObservationState.FAILED
    if raw == LayerState.NOT_OBSERVABLE.value:
        return ObservationState.NOT_EXPOSED
    if raw == LayerState.INCOMPARABLE.value:
        return ObservationState.NOT_EXPOSED
    try:
        return ObservationState(raw)
    except ValueError:
        return ObservationState.FAILED


def to_layer_state(state: ObservationState) -> LayerState:
    """Project v1 observation states onto legacy LayerState for adapters."""

    if state == ObservationState.OBSERVED:
        return LayerState.OBSERVED
    if state in {ObservationState.FAILED, ObservationState.ERROR}:
        return LayerState.ERROR
    if state == ObservationState.INCOMPARABLE:
        return LayerState.INCOMPARABLE
    return LayerState.NOT_OBSERVABLE


def comparison_outcome_for_states(
    left: ObservationState | str | LayerState,
    right: ObservationState | str | LayerState,
    *,
    required: bool = True,
) -> FormalParityStatus | None:
    """Return forced formal status when observability blocks a real compare.

    Returns None when both sides are OBSERVED and a real comparator may run.
    """

    a = normalize_observation_state(left)
    b = normalize_observation_state(right)
    if a == ObservationState.OBSERVED and b == ObservationState.OBSERVED:
        return None
    # Prefer ERROR/FAILED
    for state in (a, b):
        if state in {ObservationState.FAILED, ObservationState.ERROR}:
            return FormalParityStatus.ERROR
    for state in (a, b):
        if state == ObservationState.UNSUPPORTED:
            return FormalParityStatus.UNSUPPORTED if required else FormalParityStatus.INCONCLUSIVE
    # INFERRED only counts as weak evidence → inconclusive for required layers
    return FormalParityStatus.INCONCLUSIVE


def honesty_rule() -> str:
    return (
        "Never treat NOT_EXPOSED, UNSUPPORTED, REDACTED, INFERRED, or FAILED as equality. "
        "Missing observation → INCONCLUSIVE or UNSUPPORTED, never PASS."
    )


def legacy_parity_from_formal(status: FormalParityStatus) -> ParityResult:
    mapping = {
        FormalParityStatus.PASS: ParityResult.PASS,
        FormalParityStatus.PASS_WITH_TOLERANCE: ParityResult.PASS_WITH_TOLERANCE,
        FormalParityStatus.DIVERGENT: ParityResult.DIVERGENT,
        FormalParityStatus.INCONCLUSIVE: ParityResult.NOT_OBSERVABLE,
        FormalParityStatus.UNSUPPORTED: ParityResult.INCOMPARABLE,
        FormalParityStatus.ERROR: ParityResult.ERROR,
    }
    return mapping[status]
