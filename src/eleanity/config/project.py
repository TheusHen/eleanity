from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from eleanity.models.schemas import ParityProfile, ParityResult


class BackendProfile(BaseModel):
    """Named backend connection used by eleanity.yaml."""

    adapter: str
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    tokenize_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GateRule(BaseModel):
    """Production gate: max allowed status per layer set."""

    name: str
    layers: list[str]
    max_status: ParityResult = ParityResult.PASS
    allow: list[ParityResult] = Field(default_factory=list)
    tolerance: float | None = None
    required: bool = True


class SuiteRef(BaseModel):
    name: str
    path: str
    description: str | None = None


class EleanityProject(BaseModel):
    schema_version: str = "0.2"
    name: str = "eleanity"
    model: str | None = None
    baseline: str | None = None
    backends: list[str] = Field(default_factory=lambda: ["transformers", "vllm", "llamacpp"])
    policy: ParityProfile = ParityProfile.STRICT
    observe: list[str] = Field(
        default_factory=lambda: ["artifact", "template", "special_tokens", "tokens", "generation"]
    )
    redact_prompts: bool = False
    tokenizer_only: bool = False
    parallel: bool = True
    workers: int = 4
    runs_dir: str = ".eleanity/runs"
    scenarios_dir: str = "scenarios"
    suites: list[SuiteRef] = Field(default_factory=list)
    backend_profiles: dict[str, BackendProfile] = Field(default_factory=dict)
    gates: list[GateRule] = Field(default_factory=list)
    golden_dir: str = ".eleanity/golden"
    env: dict[str, str] = Field(default_factory=dict)
    default_suite: str | None = None
    check_golden: bool = False
    golden_file: str | None = None
    offline: bool = False
    profiles: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Named compare profiles (backends/policy/observe overrides)",
    )

    def resolved_backends(self) -> list[str]:
        return list(self.backends)

    def profile(self, name: str) -> dict[str, Any]:
        return dict(self.profiles.get(name) or {})


DEFAULT_PROJECT_YAML = """\
schema_version: "0.2"
name: eleanity-project
model: Qwen/Qwen2.5-0.5B-Instruct
baseline: transformers
backends:
  - transformers
  - vllm
  - llamacpp
policy: strict
observe:
  - artifact
  - template
  - special_tokens
  - tokens
  - generation
redact_prompts: false
tokenizer_only: false
parallel: true
workers: 4
runs_dir: .eleanity/runs
scenarios_dir: scenarios
golden_dir: .eleanity/golden
default_suite: null
check_golden: false
offline: false
profiles:
  ci-tokenizer:
    tokenizer_only: true
    backends: [transformers, transformers]
    policy: strict
    observe: [artifact, template, special_tokens, tokens]
  local-fake:
    backends: [fake, fake]
    policy: strict
suites:
  - name: qwen-parity
    path: fixtures/qwen/scenarios.yaml
    description: Core Qwen chat parity suite
  - name: tokenizer-torture
    path: fixtures/qwen/tokenizer_torture.yaml
    description: Unicode and tokenizer edge cases
  - name: tool-calling
    path: fixtures/suites/tool-calling.yaml
    description: Tool / function calling functional checks
backend_profiles:
  transformers:
    adapter: transformers
  vllm:
    adapter: vllm
    base_url: ${ELEANITY_VLLM_URL}
    api_key_env: ELEANITY_VLLM_API_KEY
  llamacpp:
    adapter: llamacpp
    base_url: ${ELEANITY_LLAMACPP_URL}
  ollama:
    adapter: ollama
    base_url: ${ELEANITY_OLLAMA_URL}
  sglang:
    adapter: sglang
    base_url: ${ELEANITY_SGLANG_URL}
  tgi:
    adapter: tgi
    base_url: ${ELEANITY_TGI_URL}
gates:
  # Missing/unobserved layers count as NOT_OBSERVABLE and are allowed here.
  - name: prompt-must-match
    layers: [template, tokens, special_tokens]
    max_status: PASS
    allow: [NOT_OBSERVABLE]
  - name: generation-soft
    layers: [generation]
    max_status: PASS_WITH_TOLERANCE
    allow: [NOT_OBSERVABLE, INCOMPARABLE]
  - name: no-hard-errors
    layers: [artifact, template, tokens, generation]
    max_status: DIVERGENT
    allow: [NOT_OBSERVABLE, INCOMPARABLE, PASS, PASS_WITH_TOLERANCE, DIVERGENT]
"""


def find_project_file(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for path in [cur, *cur.parents]:
        candidate = path / "eleanity.yaml"
        if candidate.is_file():
            return candidate
        candidate = path / "eleanity.yml"
        if candidate.is_file():
            return candidate
    return None


def load_project(path: Path | str | None = None) -> EleanityProject:
    if path is None:
        found = find_project_file()
        if found is None:
            return EleanityProject()
        path = found
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EleanityProject.model_validate(data)


def write_default_project(path: Path | str = "eleanity.yaml", *, force: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists (use force=True to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_PROJECT_YAML, encoding="utf-8")
    return target
