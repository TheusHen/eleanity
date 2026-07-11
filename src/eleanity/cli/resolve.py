from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eleanity.config.project import BackendProfile, EleanityProject, GateRule, find_project_file, load_project
from eleanity.models.schemas import ParityProfile, Scenario


DEFAULT_BACKENDS = "transformers,vllm,llamacpp"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ResolvedCompare:
    model: str
    backends: list[str]
    baseline: str | None
    policy: ParityProfile
    observe: list[str] | None
    tokenizer_only: bool
    redact_prompts: bool
    parallel: bool
    workers: int
    runs_dir: Path
    golden_dir: Path
    apply_gates: bool
    gates: list[GateRule]
    backend_profiles: dict[str, BackendProfile]
    backend_urls: dict[str, str] = field(default_factory=dict)
    offline: bool = False
    project: EleanityProject | None = None
    scenario_path: Path | None = None
    suite: str | None = None
    golden_path: Path | None = None
    check_golden: bool = False


def _parse_backend_urls(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            continue
        name, url = item.split("=", 1)
        result[name.strip().lower()] = url.strip().rstrip("/")
    # Env fallbacks
    env_map = {
        "vllm": "ELEANITY_VLLM_URL",
        "llamacpp": "ELEANITY_LLAMACPP_URL",
        "ollama": "ELEANITY_OLLAMA_URL",
        "sglang": "ELEANITY_SGLANG_URL",
        "tgi": "ELEANITY_TGI_URL",
        "openai": "ELEANITY_OPENAI_URL",
    }
    for backend, env_name in env_map.items():
        if backend not in result and os.getenv(env_name):
            result[backend] = os.environ[env_name].rstrip("/")
    if "openai" not in result and os.getenv("OPENAI_BASE_URL"):
        result["openai"] = os.environ["OPENAI_BASE_URL"].rstrip("/")
    return result


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_project_optional(config: Path | None) -> EleanityProject | None:
    if config is not None:
        return load_project(config)
    found = find_project_file()
    return load_project(found) if found else None


def resolve_compare(
    *,
    model: str | None = None,
    backends: str | None = None,
    baseline: str | None = None,
    policy: str | None = None,
    observe: str | None = None,
    tokenizer_only: bool | None = None,
    redact_prompts: bool | None = None,
    parallel: bool | None = None,
    workers: int | None = None,
    config: Path | None = None,
    backend_url: list[str] | None = None,
    no_gates: bool = False,
    offline: bool = False,
    suite: str | None = None,
    scenario: Path | None = None,
    golden: Path | None = None,
    check_golden: bool = False,
) -> ResolvedCompare:
    """Resolve settings with precedence: CLI > env > eleanity.yaml > defaults."""

    project = load_project_optional(config)

    # Apply project.env into process if not already set
    if project and project.env:
        for key, value in project.env.items():
            os.environ.setdefault(key, value)

    env_model = os.getenv("ELEANITY_MODEL")
    env_backends = os.getenv("ELEANITY_BACKENDS")
    env_baseline = os.getenv("ELEANITY_BASELINE")
    env_policy = os.getenv("ELEANITY_POLICY")
    env_tok = _env_bool("ELEANITY_TOKENIZER_ONLY")
    env_redact = _env_bool("ELEANITY_REDACT_PROMPTS")
    env_offline = _env_bool("ELEANITY_OFFLINE")
    env_parallel = _env_bool("ELEANITY_PARALLEL")

    # Model: CLI (non-null) > env > project > default
    if model is not None:
        resolved_model = model
    elif env_model:
        resolved_model = env_model
    elif project and project.model:
        resolved_model = project.model
    else:
        resolved_model = DEFAULT_MODEL

    # Backends: CLI (non-null) > env > project > default
    if backends is not None:
        backends_str = backends
    elif env_backends:
        backends_str = env_backends
    elif project and project.backends:
        backends_str = ",".join(project.backends)
    else:
        backends_str = DEFAULT_BACKENDS
    backend_list = [b.strip() for b in backends_str.split(",") if b.strip()]

    # Baseline
    if baseline:
        resolved_baseline = baseline
    elif env_baseline:
        resolved_baseline = env_baseline
    elif project and project.baseline and project.baseline in backend_list:
        resolved_baseline = project.baseline
    else:
        resolved_baseline = backend_list[0] if backend_list else None

    # Policy
    if policy:
        resolved_policy = ParityProfile(policy)
    elif env_policy:
        resolved_policy = ParityProfile(env_policy)
    elif project:
        resolved_policy = project.policy
    else:
        resolved_policy = ParityProfile.STRICT

    # Observe
    observe_list: list[str] | None = None
    if observe:
        observe_list = [item.strip() for item in observe.split(",") if item.strip()]
    elif project and project.observe:
        observe_list = list(project.observe)

    resolved_tok = (
        tokenizer_only
        if tokenizer_only is not None
        else (env_tok if env_tok is not None else bool(project.tokenizer_only if project else False))
    )
    resolved_redact = (
        redact_prompts
        if redact_prompts is not None
        else (env_redact if env_redact is not None else bool(project.redact_prompts if project else False))
    )
    resolved_parallel = (
        parallel
        if parallel is not None
        else (env_parallel if env_parallel is not None else bool(project.parallel if project else True))
    )
    if parallel is False:
        resolved_parallel = False

    resolved_workers = workers if workers is not None else (project.workers if project else 4)
    runs_dir = Path(project.runs_dir) if project else Path(".eleanity/runs")
    golden_dir = Path(project.golden_dir) if project else Path(".eleanity/golden")

    urls = _parse_backend_urls(backend_url)
    profiles = dict(project.backend_profiles) if project else {}
    # Inject CLI/env URLs into ephemeral profiles
    for name, url in urls.items():
        if name in profiles:
            profiles[name] = profiles[name].model_copy(update={"base_url": url})
        else:
            profiles[name] = BackendProfile(adapter=name, base_url=url)

    gates = list(project.gates) if project and not no_gates else []
    if no_gates:
        gates = []

    # Default suite from project
    resolved_suite = suite
    if resolved_suite is None and scenario is None and project and project.suites:
        # only auto-suite when compare is zero-flag style (no scenario)
        pass

    golden_path = golden
    if golden_path is None and project and project.golden_file:
        golden_path = Path(project.golden_file)

    check_g = check_golden or bool(project.check_golden if project else False)

    return ResolvedCompare(
        model=resolved_model,
        backends=backend_list,
        baseline=resolved_baseline,
        policy=resolved_policy,
        observe=observe_list,
        tokenizer_only=bool(resolved_tok),
        redact_prompts=bool(resolved_redact),
        parallel=bool(resolved_parallel),
        workers=int(resolved_workers or 4),
        runs_dir=runs_dir,
        golden_dir=golden_dir,
        apply_gates=not no_gates,
        gates=gates,
        backend_profiles=profiles,
        backend_urls=urls,
        offline=bool(offline or env_offline),
        project=project,
        scenario_path=scenario,
        suite=resolved_suite,
        golden_path=golden_path,
        check_golden=check_g,
    )


def apply_scenario_overrides(
    scenario: Scenario,
    resolved: ResolvedCompare,
) -> Scenario:
    """Apply policy/observe/tokenizer_only onto a scenario without clobbering messages."""

    updates: dict[str, Any] = {}
    if resolved.policy:
        updates["parity_profile"] = resolved.policy
        updates["parity_policy"] = resolved.policy
    if resolved.observe:
        updates["observe"] = list(resolved.observe)
    if resolved.redact_prompts:
        updates["redact_prompts"] = True
    if resolved.tokenizer_only:
        model = scenario.model
        if model is None:
            from eleanity.models.schemas import ModelSpec

            updates["model"] = ModelSpec(id=resolved.model, tokenizer_only=True)
        else:
            updates["model"] = model.model_copy(update={"tokenizer_only": True})
    return scenario.model_copy(update=updates) if updates else scenario
