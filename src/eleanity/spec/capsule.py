"""Execution Capsule — freeze everything that influences a run for replay."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from typing import Any

from pydantic import BaseModel, Field

from eleanity.models.schemas import ArtifactFingerprint, EnvironmentFingerprint, Scenario


class ArtifactCapsule(BaseModel):
    model_id: str | None = None
    revision: str | None = None
    weight_hash: str | None = None
    tokenizer_hash: str | None = None
    config_hash: str | None = None
    chat_template_hash: str | None = None
    quantization: str | None = None
    dtype: str | None = None
    local_path: str | None = None


class RuntimeCapsule(BaseModel):
    name: str | None = None
    version: str | None = None
    commit: str | None = None
    container_digest: str | None = None
    library_versions: dict[str, str | None] = Field(default_factory=dict)
    python_version: str | None = None


class HardwareCapsule(BaseModel):
    accelerator: str | None = None
    driver: str | None = None
    cuda: str | None = None
    compute_capability: str | None = None
    cpu_arch: str | None = None
    os: str | None = None


class GenerationCapsule(BaseModel):
    seed: int | float | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    stop: list[Any] = Field(default_factory=list)
    logits_processors: list[str] = Field(default_factory=list)
    add_generation_prompt: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ExecutionSettings(BaseModel):
    dtype: str | None = None
    quantization: str | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    batch_size: int | None = None
    attention_backend: str | None = None
    prefix_cache: bool | None = None
    speculative_decoding: bool | None = None
    tokenizer_only: bool | None = None
    offline: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PrivacyCapsule(BaseModel):
    redact_input: bool = False
    redact_output: bool = False
    hash_content: bool = False
    no_store: bool = False
    allow_remote: bool = False
    retention: str | None = None
    content_left_machine: bool = False


class ExecutionCapsule(BaseModel):
    """Frozen execution context stored alongside every trace/run."""

    schema_version: str = "1.0.0"
    artifact: ArtifactCapsule = Field(default_factory=ArtifactCapsule)
    runtime: RuntimeCapsule = Field(default_factory=RuntimeCapsule)
    hardware: HardwareCapsule = Field(default_factory=HardwareCapsule)
    generation: GenerationCapsule = Field(default_factory=GenerationCapsule)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    privacy: PrivacyCapsule = Field(default_factory=PrivacyCapsule)
    scenario_name: str | None = None
    scenario_hash: str | None = None
    observe: list[str] = Field(default_factory=list)
    policy: str | None = None
    capsule_hash: str | None = None

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"capsule_hash"})
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def seal(self) -> ExecutionCapsule:
        self.capsule_hash = self.compute_hash()
        return self


def _scenario_hash(scenario: Scenario | None) -> str | None:
    if scenario is None:
        return None
    payload = scenario.model_dump(mode="json")
    # Redact messages content for hash identity of structure? Include for true freeze.
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_execution_capsule(
    *,
    backend: str,
    model: str,
    scenario: Scenario | None = None,
    fingerprint: ArtifactFingerprint | None = None,
    environment: EnvironmentFingerprint | None = None,
    adapter_version: str | None = None,
    backend_flags: dict[str, Any] | None = None,
    privacy: dict[str, Any] | None = None,
    tokenizer_only: bool = False,
) -> ExecutionCapsule:
    """Assemble a capsule from scenario + fingerprints + env."""

    fp = fingerprint
    env = environment
    params = dict(scenario.parameters) if scenario else {}
    gen_cfg = scenario.generation if scenario else None
    flags = dict(backend_flags or {})
    if fp and fp.backend_flags:
        flags = {**fp.backend_flags, **flags}

    artifact = ArtifactCapsule(
        model_id=(fp.model_ref if fp else None) or model,
        revision=fp.revision if fp else (scenario.model.revision if scenario and scenario.model else None),
        weight_hash=fp.model_hash if fp else None,
        tokenizer_hash=fp.tokenizer_hash if fp else None,
        config_hash=fp.config_hash if fp else None,
        chat_template_hash=fp.chat_template_hash if fp else None,
        quantization=fp.quantization if fp else (scenario.model.quantization if scenario and scenario.model else None),
        dtype=fp.dtype if fp else (scenario.model.dtype if scenario and scenario.model else None),
        local_path=fp.local_path if fp else None,
    )

    libs = dict(fp.library_versions) if fp and fp.library_versions else {}
    if env and env.packages:
        libs = {**env.packages, **libs}

    runtime = RuntimeCapsule(
        name=backend,
        version=adapter_version or (fp.runtime_version if fp else None),
        commit=os.environ.get("ELEANITY_RUNTIME_COMMIT") or (fp.commit_sha if fp else None),
        container_digest=os.environ.get("ELEANITY_CONTAINER_DIGEST"),
        library_versions=libs,
        python_version=(fp.python_version if fp else None)
        or (env.python_version if env else platform.python_version()),
    )

    hardware = HardwareCapsule(
        accelerator=(fp.gpu if fp else None) or (env.gpu_name if env else None),
        driver=fp.driver if fp else None,
        cuda=(fp.cuda_or_rocm if fp else None) or (env.cuda_version if env else None),
        compute_capability=os.environ.get("ELEANITY_COMPUTE_CAPABILITY"),
        cpu_arch=(fp.cpu_arch if fp else None) or (env.machine if env else platform.machine()),
        os=(fp.os if fp else None) or (env.platform if env else platform.platform()),
    )

    generation = GenerationCapsule(
        seed=params.get("seed"),
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        min_p=params.get("min_p"),
        max_tokens=params.get("max_tokens") or params.get("max_new_tokens"),
        repetition_penalty=params.get("repetition_penalty"),
        stop=list(params.get("stop") or params.get("stop_sequences") or []),
        logits_processors=list(params.get("logits_processors") or []),
        add_generation_prompt=gen_cfg.add_generation_prompt if gen_cfg else None,
        extra={
            k: v
            for k, v in params.items()
            if k
            not in {
                "seed",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "max_tokens",
                "max_new_tokens",
                "repetition_penalty",
                "stop",
                "stop_sequences",
                "logits_processors",
            }
        },
    )

    execution = ExecutionSettings(
        dtype=artifact.dtype,
        quantization=artifact.quantization,
        tensor_parallel=flags.get("tensor_parallel") or flags.get("tp"),
        pipeline_parallel=flags.get("pipeline_parallel") or flags.get("pp"),
        batch_size=flags.get("batch_size"),
        attention_backend=flags.get("attention_backend") or flags.get("attn_implementation"),
        prefix_cache=flags.get("prefix_cache"),
        speculative_decoding=flags.get("speculative_decoding"),
        tokenizer_only=tokenizer_only,
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
        extra={
            k: v
            for k, v in flags.items()
            if k
            not in {
                "tensor_parallel",
                "tp",
                "pipeline_parallel",
                "pp",
                "batch_size",
                "attention_backend",
                "attn_implementation",
                "prefix_cache",
                "speculative_decoding",
            }
        },
    )

    priv = PrivacyCapsule(**(privacy or {}))

    capsule = ExecutionCapsule(
        artifact=artifact,
        runtime=runtime,
        hardware=hardware,
        generation=generation,
        execution=execution,
        privacy=priv,
        scenario_name=scenario.name if scenario else None,
        scenario_hash=_scenario_hash(scenario),
        observe=list(scenario.observe) if scenario else [],
        policy=scenario.parity_profile.value if scenario else None,
    )
    return capsule.seal()
