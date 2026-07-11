"""Formal Eleanity specifications: parity, layers, observability, capsule, impact, trace v1."""

from eleanity.spec.capsule import ExecutionCapsule, build_execution_capsule
from eleanity.spec.impact import FunctionalImpact, ImpactAssessment, assess_impact
from eleanity.spec.layers import (
    CANONICAL_LAYERS,
    LAYER_ALIASES,
    LAYER_ORDER_V1,
    canonicalize_layer,
    expand_observe_layers,
)
from eleanity.spec.observability import ObservationState, normalize_observation_state
from eleanity.spec.parity import (
    COMPARATOR_MODES,
    ComparatorSpec,
    FormalParityStatus,
    PolicyComparatorSet,
    formal_status_from_parity,
    policy_comparator_set,
    status_definition,
)
from eleanity.spec.trace_v1 import TRACE_SCHEMA_VERSION, build_trace_document, validate_trace_document

__all__ = [
    "CANONICAL_LAYERS",
    "COMPARATOR_MODES",
    "ComparatorSpec",
    "ExecutionCapsule",
    "FormalParityStatus",
    "FunctionalImpact",
    "ImpactAssessment",
    "LAYER_ALIASES",
    "LAYER_ORDER_V1",
    "ObservationState",
    "PolicyComparatorSet",
    "TRACE_SCHEMA_VERSION",
    "assess_impact",
    "build_execution_capsule",
    "build_trace_document",
    "canonicalize_layer",
    "expand_observe_layers",
    "formal_status_from_parity",
    "normalize_observation_state",
    "policy_comparator_set",
    "status_definition",
    "validate_trace_document",
]
