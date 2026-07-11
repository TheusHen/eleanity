from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ParityProfile(str, Enum):
    STRICT = "strict"
    QUANTIZED = "quantized"
    FUNCTIONAL = "functional"
    API_CONFORMANCE = "api_conformance"


class ParityResult(str, Enum):
    """Comparison outcomes only — never invent PASS when data is missing.

    Observation availability uses LayerState, not these values.
    """

    PASS = "PASS"
    PASS_WITH_TOLERANCE = "PASS_WITH_TOLERANCE"
    PASS_WITH_LIMITED_COVERAGE = "PASS_WITH_LIMITED_COVERAGE"
    DIVERGENT = "DIVERGENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCOMPARABLE = "INCOMPARABLE"  # legacy alias of partial incomparability
    NOT_OBSERVABLE = "NOT_OBSERVABLE"  # legacy comparison placeholder
    NOT_REQUESTED = "NOT_REQUESTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    ERROR = "ERROR"


# Spec alias
ResultStatus = ParityResult


class LayerState(str, Enum):
    """Observation availability of a layer on one backend (not a comparison result)."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    NOT_REQUESTED = "NOT_REQUESTED"  # scenario did not ask for this layer
    NOT_SUPPORTED = "NOT_SUPPORTED"  # adapter cannot expose the layer
    NOT_EXPOSED = "NOT_EXPOSED"  # runtime hides it (HTTP API gap)
    NOT_OBSERVABLE = "NOT_OBSERVABLE"  # legacy umbrella for not-available
    UNSUPPORTED = "UNSUPPORTED"  # alias of NOT_SUPPORTED
    REDACTED = "REDACTED"
    FAILED = "FAILED"
    ERROR = "ERROR"  # legacy alias of FAILED
    INCOMPARABLE = "INCOMPARABLE"


# Coarse + fine-grained layers (fine names accepted; mapped for adapters).
ALLOWED_OBSERVE_LAYERS = frozenset(
    {
        # coarse / legacy
        "artifact",
        "template",
        "rendered_prompt",
        "tokens",
        "special_tokens",
        "logits",
        "forward",
        "generation",
        "stop_reason",
        "structured",
        "streaming",
        "api",
        # fine-grained (Trace Spec v1)
        "model_config",
        "tokenizer_artifact",
        "chat_template",
        "normalization",
        "input_token_ids",
        "generation_config",
        "prefill_logits",
        "decode_logits",
        "logits_processing",
        "sampling",
        "generated_token_ids",
        "stop_decision",
        "detokenization",
        "response_mapping",
        "usage_accounting",
        "structured_output",
        "tool_call_parsing",
        "multimodal_inputs",
        "embeddings",
        "reasoning_content",
        "speculative_decoding",
    }
)

LAYER_ORDER = (
    "artifact",
    "template",
    "special_tokens",
    "tokens",
    "logits",
    "generation",
    "structured",
    "streaming",
    "api",
)

DEFAULT_TOLERANCE = {
    ParityProfile.STRICT: 1e-5,
    ParityProfile.QUANTIZED: 0.02,
    ParityProfile.FUNCTIONAL: 0.1,
    ParityProfile.API_CONFORMANCE: 0.0,
}


class Message(BaseModel):
    role: str
    content: str


class ModelSpec(BaseModel):
    """Model identity and load policy shared across adapters."""

    id: str | None = None
    revision: str | None = "main"
    trust_remote_code: bool = False
    dtype: str | None = "auto"
    device_map: str | None = "auto"
    quantization: str | None = None
    max_memory: dict[str, str] | None = None
    attn_implementation: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    local_path: str | None = None
    lora_adapters: list[str] = Field(default_factory=list)
    tokenizer_only: bool = False


class GenerationConfig(BaseModel):
    add_generation_prompt: bool = True
    continue_final_message: bool = False


class Scenario(BaseModel):
    schema_version: str = "0.1"
    name: str
    description: str | None = None
    model: ModelSpec | None = None
    messages: list[Message]
    parameters: dict[str, Any] = Field(default_factory=dict)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    observe: list[str] = Field(default_factory=lambda: ["template", "tokens", "generation"])
    parity_profile: ParityProfile = ParityProfile.STRICT
    parity_policy: ParityProfile | None = None
    tolerance: float | None = None
    backends: list[str] = Field(default_factory=list)
    redact_prompts: bool = False
    # Optional explicit per-layer comparator overrides (Trace Spec / formal policy)
    comparators: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "Scenario":
        if self.parity_policy is not None:
            self.parity_profile = self.parity_policy
        # Map fine-grained observe names onto coarse adapter pipeline layers
        from eleanity.spec.layers import to_coarse_observe

        try:
            coarse = to_coarse_observe(list(self.observe))
        except Exception:
            coarse = list(self.observe)
        normalized: list[str] = []
        for layer in coarse or list(self.observe):
            if layer == "rendered_prompt":
                layer = "template"
            if layer == "forward":
                layer = "logits"
            if layer == "stop_reason":
                if "generation" not in normalized and "generation" not in self.observe:
                    normalized.append("generation")
                continue
            if layer not in normalized:
                normalized.append(layer)
        # Keep any still-valid fine names that didn't map (for future adapters)
        for layer in self.observe:
            if layer in ALLOWED_OBSERVE_LAYERS and layer not in normalized:
                # only keep coarse for observe pipeline
                pass
        self.observe = normalized
        unknown = set(self.observe) - ALLOWED_OBSERVE_LAYERS
        if unknown:
            raise ValueError(f"unknown observable layers: {sorted(unknown)}")
        if self.tolerance is None:
            # Keep strict numeric default at 0.0 for backward-compatible tests
            # that expect exact zero for ParityProfile.STRICT.
            if self.parity_profile == ParityProfile.STRICT:
                self.tolerance = 0.0
            else:
                self.tolerance = DEFAULT_TOLERANCE[self.parity_profile]
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        return self

    @property
    def model_id(self) -> str | None:
        return self.model.id if self.model else None


class ArtifactFingerprint(BaseModel):
    """Model + environment identity used for artifact-layer comparison."""

    model_ref: str
    revision: str | None = None
    commit_sha: str | None = None
    local_path: str | None = None
    model_hash: str | None = None
    config_hash: str | None = None
    tokenizer: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    model_type: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    dtype: str | None = None
    gguf_metadata: dict[str, Any] = Field(default_factory=dict)
    awq_gptq_metadata: dict[str, Any] = Field(default_factory=dict)
    lora_adapters: list[str] = Field(default_factory=list)
    generation_config: dict[str, Any] = Field(default_factory=dict)
    special_tokens: dict[str, Any] = Field(default_factory=dict)
    runtime_version: str | None = None
    library_versions: dict[str, str | None] = Field(default_factory=dict)
    python_version: str | None = None
    os: str | None = None
    cpu_arch: str | None = None
    gpu: str | None = None
    driver: str | None = None
    cuda_or_rocm: str | None = None
    backend_flags: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def text_hash(value: str | None) -> str | None:
        return sha256(value.encode()).hexdigest() if value is not None else None


class EnvironmentFingerprint(BaseModel):
    python_version: str | None = None
    platform: str | None = None
    machine: str | None = None
    processor: str | None = None
    packages: dict[str, str | None] = Field(default_factory=dict)
    cuda_available: bool | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    torch_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PromptObservation(BaseModel):
    chat_template_source: str | None = None
    chat_template_hash: str | None = None
    add_generation_prompt: bool | None = None
    continue_final_message: bool | None = None
    rendered_text: str | None = None
    rendered_utf8_hex: str | None = None
    rendered_byte_length: int | None = None
    rendered_char_length: int | None = None
    roles: list[str] = Field(default_factory=list)
    tools_included: bool = False
    special_markers: list[str] = Field(default_factory=list)
    # Legacy alias used by existing layers
    text: str | None = None

    def to_layer_data(self, *, redact: bool = False) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        text = self.rendered_text or self.text
        data["text"] = text
        if redact and text is not None:
            data["rendered_text"] = None
            data["text"] = None
            data["rendered_utf8_hex"] = None
            data["content_redacted"] = True
            data["template_hash"] = self.chat_template_hash or ArtifactFingerprint.text_hash(text)
        return data


class TokenObservation(BaseModel):
    token_ids: list[int] = Field(default_factory=list)
    token_strings: list[str] | None = None
    decoded_text: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    added_special_tokens: list[int] = Field(default_factory=list)
    attention_mask: list[int] | None = None
    original_length: int = 0
    final_length: int = 0
    truncated: bool = False
    padding_side: str | None = None
    truncation_side: str | None = None
    special_token_positions: list[int] = Field(default_factory=list)
    special_token_count: int = 0
    add_special_tokens: bool = False

    def to_layer_data(self, *, redact: bool = False) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["ids"] = list(self.token_ids)
        data["count"] = self.final_length or len(self.token_ids)
        if redact:
            data["token_ids"] = []
            data["ids"] = []
            data["token_strings"] = None
            data["decoded_text"] = None
            data["content_redacted"] = True
        return data


class TraceError(BaseModel):
    code: str
    message: str
    layer: str | None = None
    detail: str | None = None


class LayerObservation(BaseModel):
    state: LayerState
    data: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None
    # Provenance: how this observation was obtained
    origin: str | None = None  # e.g. "transformers.apply_chat_template", "vllm.http:/v1/chat/completions"
    origin_kind: str | None = None  # native | http | inferred | embedded | unavailable


class ObservationTrace(BaseModel):
    trace_version: str = "0"
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    scenario_name: str
    backend: str
    baseline_backend: str | None = None
    artifact_fingerprint: ArtifactFingerprint
    environment: EnvironmentFingerprint | None = None
    layers: dict[str, LayerObservation]
    warnings: list[str] = Field(default_factory=list)
    errors: list[TraceError] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = None


class Comparison(BaseModel):
    result: ParityResult
    details: dict[str, Any] = Field(default_factory=dict)
    # Separate observation availability from comparison verdict
    baseline_state: LayerState | None = None
    candidate_state: LayerState | None = None
    tolerance_reason: str | None = None


class ProbableCause(BaseModel):
    code: str
    confidence: float
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    affected_layers: list[str] = Field(default_factory=list)
    suggested_remediation: str | None = None


class DivergenceLocation(BaseModel):
    character: int | None = None
    byte: int | None = None
    line: int | None = None
    column: int | None = None
    token_index: int | None = None


class FirstDivergence(BaseModel):
    layer: str
    location: DivergenceLocation = Field(default_factory=DivergenceLocation)
    baseline: str | None = None
    candidate: str | None = None


class PropagationInfo(BaseModel):
    first_token_difference: int | None = None
    different_tokens_percent: float | None = None
    downstream_different: int | None = None


class Diagnosis(BaseModel):
    status: ParityResult = ParityResult.PASS
    formal_status: str | None = None  # PASS|…|INCONCLUSIVE|UNSUPPORTED (Trace Spec v1)
    first_divergence: str | None = None
    first_divergence_detail: FirstDivergence | None = None
    propagation: PropagationInfo | None = None
    propagation_percent: float = 0.0
    probable_causes: list[ProbableCause] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    next_test: str = ""
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    impact: dict[str, Any] | None = None
    policy_comparators: dict[str, Any] | None = None
    # Coverage / confidence enrichment
    confidence: float | None = None  # 0..1 diagnostic confidence
    coverage: dict[str, Any] | None = None
    verified_layers: list[str] = Field(default_factory=list)
    not_verified_layers: list[dict[str, Any]] = Field(default_factory=list)
    tolerance_reasons: list[str] = Field(default_factory=list)
    artifact_divergent_fields: list[str] = Field(default_factory=list)
    practical_commands: list[str] = Field(default_factory=list)
