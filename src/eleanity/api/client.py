"""Object-oriented public client (layer B).

Example::

    from eleanity import Eleanity

    client = Eleanity.from_yaml("eleanity.yaml")
    result = client.compare(model="demo", backends=["fake", "fake"])
    if not result.passed:
        raise SystemExit(result.exit_code)
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eleanity.adapters import available_adapters, create_adapter
from eleanity.api.errors import ConfigError, NotFoundError
from eleanity.api.lowlevel import (
    compare_traces,
    diagnose_traces,
    load_scenario_file,
    make_scenario,
    observe_backend,
)
from eleanity.api.types import (
    BackendHealth,
    CompareOutcome,
    DoctorReport,
    ScenarioResult,
    TestReport,
    outcome_from_engine,
)
from eleanity.cli.resolve import (
    ResolvedCompare,
    apply_scenario_overrides,
    load_project_optional,
    resolve_compare,
)
from eleanity.config.project import EleanityProject, find_project_file, load_project
from eleanity.core.engine import CompareEngine
from eleanity.core.golden import golden_gate, save_golden
from eleanity.core.runs_index import diff_runs, list_runs, load_run
from eleanity.models.schemas import ModelSpec, ObservationTrace, Scenario
from eleanity.scenarios.loader import load_scenarios
from eleanity.scenarios.suites import list_builtin_suites, load_suite

__all__ = ["Eleanity"]


class Eleanity:
    """Stateful client for embedding parity diagnostics in Python pipelines.

    Configuration precedence for method kwargs matches the CLI:
    **call kwargs > env > eleanity.yaml > defaults**.
    """

    def __init__(
        self,
        *,
        config: str | Path | None = None,
        project: EleanityProject | None = None,
        model: str | None = None,
        backends: Sequence[str] | None = None,
        baseline: str | None = None,
        policy: str | None = None,
        observe_layers: Sequence[str] | None = None,
        runs_dir: str | Path | None = None,
        tokenizer_only: bool | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        redact_prompts: bool | None = None,
        apply_gates: bool = True,
        backend_urls: dict[str, str] | None = None,
        offline: bool = False,
    ):
        self._config_path = Path(config) if config is not None else None
        if project is not None:
            self._project = project
        elif self._config_path is not None:
            self._project = load_project(self._config_path)
        else:
            self._project = load_project_optional(None)

        self.model = model
        self.backends = list(backends) if backends is not None else None
        self.baseline = baseline
        self.policy = policy
        self.observe_layers = list(observe_layers) if observe_layers is not None else None
        self.runs_dir = Path(runs_dir) if runs_dir is not None else None
        self.tokenizer_only = tokenizer_only
        self.parallel = parallel
        self.workers = workers
        self.redact_prompts = redact_prompts
        self.apply_gates = apply_gates
        self.backend_urls = dict(backend_urls or {})
        self.offline = offline

    # ------------------------------------------------------------------
    # constructors / fluent config
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> Eleanity:
        """Load defaults from an ``eleanity.yaml`` project file."""

        return cls(config=path, **kwargs)

    @classmethod
    def configure(cls, **kwargs: Any) -> Eleanity:
        """Alias for ``Eleanity(**kwargs)`` — explicit factory name."""

        return cls(**kwargs)

    def with_options(self, **kwargs: Any) -> Eleanity:
        """Return a copy with selected fields overridden (immutable-style)."""

        data = {
            "config": self._config_path,
            "project": self._project,
            "model": self.model,
            "backends": self.backends,
            "baseline": self.baseline,
            "policy": self.policy,
            "observe_layers": self.observe_layers,
            "runs_dir": self.runs_dir,
            "tokenizer_only": self.tokenizer_only,
            "parallel": self.parallel,
            "workers": self.workers,
            "redact_prompts": self.redact_prompts,
            "apply_gates": self.apply_gates,
            "backend_urls": self.backend_urls,
            "offline": self.offline,
        }
        data.update(kwargs)
        # Avoid double-loading project when project object provided
        if "project" in kwargs and "config" not in kwargs:
            data["config"] = None
        return Eleanity(**data)

    # ------------------------------------------------------------------
    # resolution helpers
    # ------------------------------------------------------------------

    def _backend_url_flags(self, extra: dict[str, str] | None = None) -> list[str]:
        merged = {**self.backend_urls, **(extra or {})}
        return [f"{k}={v}" for k, v in merged.items() if v]

    def resolve(
        self,
        *,
        model: str | None = None,
        backends: Sequence[str] | str | None = None,
        baseline: str | None = None,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool | None = None,
        redact_prompts: bool | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        no_gates: bool | None = None,
        offline: bool | None = None,
        suite: str | None = None,
        scenario: str | Path | None = None,
        backend_urls: dict[str, str] | None = None,
    ) -> ResolvedCompare:
        """Resolve effective settings (same rules as the CLI)."""

        def _backends_str(value: Sequence[str] | str | None, fallback: Sequence[str] | None) -> str | None:
            if value is None:
                if fallback is None:
                    return None
                return ",".join(fallback)
            if isinstance(value, str):
                return value
            return ",".join(value)

        def _observe_str(value: Sequence[str] | str | None, fallback: Sequence[str] | None) -> str | None:
            if value is None:
                if fallback is None:
                    return None
                return ",".join(fallback)
            if isinstance(value, str):
                return value
            return ",".join(value)

        resolved = resolve_compare(
            model=model if model is not None else self.model,
            backends=_backends_str(backends, self.backends),
            baseline=baseline if baseline is not None else self.baseline,
            policy=policy if policy is not None else self.policy,
            observe=_observe_str(observe, self.observe_layers),
            tokenizer_only=tokenizer_only if tokenizer_only is not None else self.tokenizer_only,
            redact_prompts=redact_prompts if redact_prompts is not None else self.redact_prompts,
            parallel=parallel if parallel is not None else self.parallel,
            workers=workers if workers is not None else self.workers,
            config=self._config_path,
            backend_url=self._backend_url_flags(backend_urls),
            no_gates=(not self.apply_gates) if no_gates is None else no_gates,
            offline=self.offline if offline is None else offline,
            suite=suite,
            scenario=Path(scenario) if isinstance(scenario, (str, Path)) and Path(str(scenario)).exists() else None,
        )
        if self.runs_dir is not None:
            resolved.runs_dir = Path(self.runs_dir)
        if self._project is not None and resolved.project is None:
            resolved.project = self._project
        return resolved

    def engine(self, resolved: ResolvedCompare | None = None, *, parallel: bool | None = None) -> CompareEngine:
        """Build a :class:`~eleanity.core.engine.CompareEngine` from current config."""

        resolved = resolved or self.resolve()
        return CompareEngine(
            project=resolved.project,
            runs_dir=resolved.runs_dir,
            max_workers=resolved.workers,
            parallel=resolved.parallel if parallel is None else parallel,
            tokenizer_only=resolved.tokenizer_only,
            backend_profiles=resolved.backend_profiles,
            gates=resolved.gates if resolved.apply_gates else [],
        )

    def _prepare_scenario(self, scenario: Scenario | None, resolved: ResolvedCompare) -> Scenario:
        if scenario is None:
            scenario = Scenario(
                name="compare",
                messages=[{"role": "user", "content": "Hello"}],
                observe=resolved.observe or ["artifact", "template", "special_tokens", "tokens", "generation"],
                parity_profile=resolved.policy,
            )
        scenario = apply_scenario_overrides(scenario, resolved)
        if scenario.model is None or not scenario.model.id:
            scenario = scenario.model_copy(
                update={
                    "model": ModelSpec(
                        id=resolved.model,
                        tokenizer_only=resolved.tokenizer_only,
                    )
                }
            )
        return scenario

    def _apply_offline(self, offline: bool) -> None:
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ------------------------------------------------------------------
    # high-level operations
    # ------------------------------------------------------------------

    def compare(
        self,
        *,
        model: str | None = None,
        backends: Sequence[str] | str | None = None,
        baseline: str | None = None,
        scenario: Scenario | str | Path | None = None,
        scenario_name: str | None = None,
        suite: str | None = None,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool | None = None,
        parallel: bool | None = None,
        no_gates: bool | None = None,
        redact_prompts: bool | None = None,
        backend_urls: dict[str, str] | None = None,
        offline: bool | None = None,
    ) -> CompareOutcome:
        """Run a parity compare. Returns a :class:`CompareOutcome` (no process exit)."""

        scenario_path: Path | None = None
        scenario_obj: Scenario | None = None
        if isinstance(scenario, Scenario):
            scenario_obj = scenario
        elif scenario is not None:
            scenario_path = Path(scenario)

        resolved = self.resolve(
            model=model,
            backends=backends,
            baseline=baseline,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            parallel=parallel,
            no_gates=no_gates,
            redact_prompts=redact_prompts,
            offline=offline,
            suite=suite,
            scenario=scenario_path,
            backend_urls=backend_urls,
        )
        self._apply_offline(resolved.offline)

        if scenario_obj is None and scenario_path is not None and scenario_path.exists():
            loaded = load_scenarios(scenario_path)
            if scenario_name:
                loaded = [s for s in loaded if s.name == scenario_name]
                if not loaded:
                    raise NotFoundError(f"scenario '{scenario_name}' not found in {scenario_path}")
            scenario_obj = loaded[0] if loaded else None
        elif scenario_obj is None and suite:
            loaded = load_suite(suite, project=resolved.project)
            if scenario_name:
                loaded = [s for s in loaded if s.name == scenario_name]
            scenario_obj = loaded[0] if loaded else None

        selected = self._prepare_scenario(scenario_obj, resolved)
        model_id = selected.model.id if selected.model and selected.model.id else resolved.model
        engine = self.engine(resolved)
        result = engine.compare(
            model_id,
            resolved.backends,
            scenario=selected,
            baseline_backend=resolved.baseline,
            redact_prompts=resolved.redact_prompts,
            tokenizer_only=resolved.tokenizer_only,
        )
        return outcome_from_engine(
            result,
            model=model_id,
            backends=resolved.backends,
            baseline=resolved.baseline,
            policy=resolved.policy.value if hasattr(resolved.policy, "value") else str(resolved.policy),
            scenario=selected,
        )

    def test(
        self,
        path: str | Path,
        *,
        model: str | None = None,
        backends: Sequence[str] | str | None = None,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool | None = None,
        no_gates: bool | None = None,
        fail_fast: bool = False,
    ) -> TestReport:
        """Execute every scenario in a YAML file, directory, or named suite."""

        p = Path(path)
        resolved = self.resolve(
            model=model,
            backends=backends,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            no_gates=no_gates,
        )
        if p.is_file() or p.is_dir():
            scenarios = load_scenario_file(p)
        else:
            scenarios = load_suite(str(path), project=resolved.project)

        engine = self.engine(resolved)
        report = TestReport(fail_fast=fail_fast)
        for scenario in scenarios:
            sc = self._prepare_scenario(scenario, resolved)
            model_id = sc.model.id if sc.model and sc.model.id else resolved.model
            backend_list = sc.backends or resolved.backends
            baseline = resolved.baseline if resolved.baseline in backend_list else backend_list[0]
            result = engine.compare(
                model_id,
                backend_list,
                sc,
                baseline_backend=baseline,
                redact_prompts=resolved.redact_prompts,
                tokenizer_only=resolved.tokenizer_only,
            )
            outcome = outcome_from_engine(
                result,
                model=model_id,
                backends=backend_list,
                baseline=baseline,
                policy=resolved.policy.value if hasattr(resolved.policy, "value") else str(resolved.policy),
                scenario=sc,
            )
            report.results.append(ScenarioResult(name=sc.name, outcome=outcome))
            if fail_fast and not outcome.passed:
                break
        return report

    def doctor(
        self,
        *,
        check_backends: bool = False,
        backends: Sequence[str] | str | None = None,
    ) -> DoctorReport:
        """Environment + optional backend health (structured, no Rich tables)."""

        checks: dict[str, str] = {
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.machine(),
        }
        for module, label in (
            ("transformers", "transformers"),
            ("vllm", "vllm"),
            ("llama_cpp", "llama-cpp-python"),
            ("torch", "torch"),
            ("httpx", "httpx"),
        ):
            checks[label] = "installed" if importlib.util.find_spec(module) is not None else "missing"

        free = None
        try:
            usage = shutil.disk_usage(Path.cwd())
            free = round(usage.free / (1024**3), 2)
        except OSError:
            free = None
        checks["disk_free_gb"] = str(free) if free is not None else "—"

        proj = find_project_file()
        if self._config_path:
            proj = self._config_path
        checks["eleanity.yaml"] = str(proj) if proj else "not found"

        backend_health: list[BackendHealth] = []
        if check_backends:
            from eleanity.cli.backends_check import check_backends as probe

            if isinstance(backends, str):
                names = [b.strip() for b in backends.split(",") if b.strip()]
            elif backends is not None:
                names = list(backends)
            else:
                names = list(self.backends) if self.backends else ["fake", "transformers", "vllm"]
            resolved = self.resolve(backends=names)
            for item in probe(names, model=resolved.model, resolved=resolved, require_healthy=False):
                backend_health.append(
                    BackendHealth(
                        name=item.name,
                        ok=item.ok,
                        detail=item.detail,
                        latency_ms=item.latency_ms,
                    )
                )

        # Avoid circular import with eleanity.__init__
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        try:
            ver = pkg_version("eleanity")
        except PackageNotFoundError:
            ver = "0.4.0"

        return DoctorReport(
            ok=True,
            version=ver,
            python=platform.python_version(),
            adapters=available_adapters(),
            project=str(proj) if proj else None,
            checks=checks,
            backends=backend_health,
        )

    def report(
        self,
        run_id: str,
        *,
        fmt: str = "dict",
        redact: bool = True,
    ) -> dict[str, Any] | str:
        """Load a persisted run. ``fmt``: ``dict`` | ``json`` | ``text`` | ``sarif``."""

        runs_dir = self._runs_dir()
        try:
            data = load_run(run_id, runs_dir)
        except FileNotFoundError as error:
            raise NotFoundError(f"run not found: {run_id}") from error

        if fmt == "dict":
            return data
        if fmt == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        if fmt == "text":
            from eleanity.cli.report_text import render_text_report

            return render_text_report(data, redact=redact)
        if fmt == "sarif":
            from eleanity.reporters.sarif import write_sarif

            result_path = runs_dir / str(data.get("run_id") or run_id) / "result.json"
            return write_sarif(result_path)
        raise ConfigError(f"unsupported report format: {fmt}")

    def replay(self, run_id: str, *, no_gates: bool | None = None) -> CompareOutcome:
        """Re-execute a stored compare from its ``result.json`` metadata."""

        data = self.get_run(run_id)
        scenario_meta = data.get("scenario") or {}
        backends: list[str] = []
        for t in data.get("traces") or []:
            b = t.get("backend")
            if b:
                backends.append(b)
        if len(backends) < 2:
            backends = (backends * 2) if backends else ["fake", "fake"]
        model = None
        traces = data.get("traces") or []
        if traces:
            model = (traces[0].get("artifact_fingerprint") or {}).get("model_ref")
        model = model or "demo"
        policy = scenario_meta.get("parity_profile") or scenario_meta.get("parity_policy") or "strict"
        observe = scenario_meta.get("observe")
        baseline = data.get("baseline_backend") or backends[0]
        sc = Scenario(
            name=scenario_meta.get("name") or "replay",
            messages=[{"role": "user", "content": "Hello"}],
            observe=observe or ["artifact", "template", "tokens", "generation"],
            parity_profile=policy,
            parameters=scenario_meta.get("parameters") or {"temperature": 0, "max_tokens": 32, "seed": 42},
        )
        return self.compare(
            model=model,
            backends=backends[:2],
            baseline=baseline,
            scenario=sc,
            policy=str(policy),
            observe=observe,
            tokenizer_only=bool(data.get("tokenizer_only")),
            no_gates=no_gates,
        )

    def ci(
        self,
        baseline_model: str,
        candidate_model: str,
        *,
        backend: str = "transformers",
        scenario: Scenario | str | Path | None = None,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool = False,
        no_gates: bool | None = None,
        junit_path: str | Path | None = None,
    ) -> CompareOutcome:
        """CI gate: two models on one backend."""

        scenario_obj: Scenario | None = None
        scenario_path = None
        if isinstance(scenario, Scenario):
            scenario_obj = scenario
        elif scenario is not None:
            scenario_path = Path(scenario)

        resolved = self.resolve(
            model=baseline_model,
            backends=backend,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            no_gates=no_gates,
            scenario=scenario_path,
        )
        if scenario_obj is None and scenario_path and scenario_path.exists():
            scenario_obj = load_scenarios(scenario_path)[0]
        selected = self._prepare_scenario(scenario_obj, resolved)
        engine = self.engine(resolved, parallel=False)
        run_id, comparisons, diagnosis, gate_eval = engine.ci(
            baseline_model,
            candidate_model,
            backend,
            scenario=selected,
            junit_path=Path(junit_path) if junit_path else None,
            tokenizer_only=tokenizer_only,
        )
        data = load_run(run_id, resolved.runs_dir)
        traces = [ObservationTrace.model_validate(t) for t in data.get("traces") or []]
        # Synthesize CompareResult-like via outcome_from_engine fields manually
        from eleanity.core.engine import CompareResult

        synthetic = CompareResult(
            run_id=run_id,
            traces=traces,
            diagnosis=diagnosis,
            comparisons={backend: {k: v for k, v in comparisons.items()}},
            consensus={},
            path=resolved.runs_dir / run_id,
            gate_evaluation=gate_eval,
            timings={},
        )
        return outcome_from_engine(
            synthetic,
            model=f"{baseline_model}→{candidate_model}",
            backends=[backend, backend],
            baseline=backend,
            policy=resolved.policy.value if hasattr(resolved.policy, "value") else str(resolved.policy),
            scenario=selected,
        )

    def migrate(
        self,
        *,
        model: str | None = None,
        from_backend: str,
        to_backend: str,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool | None = None,
        no_gates: bool | None = None,
    ) -> CompareOutcome:
        """Migration flow: same model, from_backend → to_backend."""

        return self.compare(
            model=model,
            backends=[from_backend, to_backend],
            baseline=from_backend,
            policy=policy or "quantized",
            observe=observe,
            tokenizer_only=tokenizer_only,
            no_gates=no_gates,
        )

    def promote(
        self,
        *,
        baseline_model: str,
        candidate_model: str,
        backend: str = "transformers",
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool = False,
        no_gates: bool | None = None,
    ) -> CompareOutcome:
        """Promotion flow: baseline model vs candidate on one backend."""

        return self.ci(
            baseline_model,
            candidate_model,
            backend=backend,
            policy=policy or "quantized",
            observe=observe,
            tokenizer_only=tokenizer_only,
            no_gates=no_gates,
        )

    def vendor_check(
        self,
        *,
        model: str | None = None,
        local_backend: str = "transformers",
        remote_backend: str = "openai",
        remote_url: str | None = None,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        no_gates: bool | None = None,
    ) -> CompareOutcome:
        """Local reference backend vs remote OpenAI-compatible endpoint."""

        urls = dict(self.backend_urls)
        if remote_url:
            urls[remote_backend] = remote_url
        return self.compare(
            model=model,
            backends=[local_backend, remote_backend],
            baseline=local_backend,
            policy=policy or "api_conformance",
            observe=observe or ["artifact", "generation", "api"],
            backend_urls=urls,
            no_gates=no_gates,
        )

    def stabilize(
        self,
        *,
        model: str | None = None,
        backend: str = "fake",
        repetitions: int = 3,
        policy: str | None = None,
        observe: Sequence[str] | str | None = None,
        tokenizer_only: bool | None = None,
    ) -> dict[str, Any]:
        """Self-consistency: backend vs itself N times before trusting cross-compare."""

        from eleanity.core.stabilize import stabilize_backend

        resolved = self.resolve(
            model=model,
            backends=backend,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            no_gates=True,
        )
        selected = self._prepare_scenario(None, resolved)
        model_id = selected.model.id if selected.model and selected.model.id else resolved.model
        engine = self.engine(resolved, parallel=False)
        report = stabilize_backend(
            engine,
            model_id,
            backend,
            scenario=selected,
            repetitions=repetitions,
        )
        return report.to_dict() if hasattr(report, "to_dict") else dict(report)

    def inspect(
        self,
        model: str,
        *,
        backend: str = "transformers",
        tokenizer_only: bool = False,
    ) -> dict[str, Any]:
        """Fingerprint / capabilities for a model on one backend."""

        adapter = create_adapter(backend, model, tokenizer_only=tokenizer_only)
        fp = adapter.fingerprint(model)
        caps = adapter.capabilities
        payload: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "fingerprint": fp.model_dump(mode="json") if hasattr(fp, "model_dump") else fp,
            "capabilities": caps.model_dump(mode="json")
            if hasattr(caps, "model_dump")
            else {
                "template": getattr(caps, "template", None),
                "tokenize": getattr(caps, "tokenize", None),
                "generation": getattr(caps, "generation", None),
                "notes": getattr(caps, "notes", {}),
            },
        }
        if hasattr(adapter, "special_tokens"):
            try:
                st = adapter.special_tokens()
                payload["special_tokens"] = st.model_dump(mode="json") if hasattr(st, "model_dump") else st
            except Exception as error:  # noqa: BLE001 — best-effort inspect
                payload["special_tokens_error"] = str(error)
        return payload

    def list_runs(self) -> list[dict[str, Any]]:
        rows = list_runs(self._runs_dir())
        out = []
        for row in rows:
            if hasattr(row, "model_dump"):
                out.append(row.model_dump(mode="json"))
            elif hasattr(row, "__dict__"):
                out.append(dict(row.__dict__))
            else:
                out.append(dict(row))
        return out

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            return load_run(run_id, self._runs_dir())
        except FileNotFoundError as error:
            raise NotFoundError(f"run not found: {run_id}") from error

    def diff_runs(self, left: str, right: str) -> dict[str, Any]:
        return diff_runs(left, right, self._runs_dir())

    def save_golden(
        self,
        run_id: str,
        *,
        backend_index: int = 0,
        directory: str | Path | None = None,
        name: str | None = None,
    ) -> Path:
        data = self.get_run(run_id)
        traces = data.get("traces") or []
        if not traces:
            raise ConfigError(f"run {run_id} has no traces")
        idx = min(backend_index, len(traces) - 1)
        trace = ObservationTrace.model_validate(traces[idx])
        target = Path(directory) if directory else self._runs_dir().parent / "goldens"
        return save_golden(trace, target, name=name or run_id[:8])

    def check_golden(
        self,
        run_id: str,
        golden_path: str | Path,
        *,
        layers: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        data = self.get_run(run_id)
        traces = data.get("traces") or []
        if not traces:
            raise ConfigError(f"run {run_id} has no traces")
        trace = ObservationTrace.model_validate(traces[0])
        sc_meta = data.get("scenario") or {}
        sc = Scenario(
            name=sc_meta.get("name") or "golden-check",
            messages=[{"role": "user", "content": "Hello"}],
            observe=sc_meta.get("observe") or ["template", "tokens"],
            parity_profile=sc_meta.get("parity_profile") or "strict",
        )
        return golden_gate(
            trace,
            Path(golden_path),
            sc,
            layers=list(layers) if layers else ["template", "tokens"],
        )

    def certify(self, backend: str, *, model: str = "demo") -> dict[str, Any]:
        from eleanity.certification import certify_runtime

        adapter = create_adapter(backend, model)
        report = certify_runtime(adapter, model=model)
        if hasattr(report, "to_dict"):
            return report.to_dict()
        if hasattr(report, "model_dump"):
            return report.model_dump(mode="json")
        return {
            "backend": backend,
            "passed": getattr(report, "passed", None),
            "level": getattr(report, "level", None),
            "report": str(report),
        }

    def capture(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        hash_content: bool = True,
    ) -> dict[str, Any]:
        from eleanity.core.capture import capture_openai_jsonl

        return capture_openai_jsonl(
            Path(input_path),
            Path(output_path),
            hash_content=hash_content,
        )

    def policy_spec(self, policy: str = "strict") -> dict[str, Any]:
        from eleanity.spec.parity import policy_comparator_set

        spec = policy_comparator_set(policy)
        if hasattr(spec, "to_dict"):
            return spec.to_dict()
        if hasattr(spec, "model_dump"):
            return spec.model_dump(mode="json")
        return {"policy": policy, "spec": str(spec)}

    def playbook(self, code: str) -> str:
        from eleanity.playbook import get_playbook_entry, render_playbook_markdown

        entry = get_playbook_entry(code)
        if entry is None:
            raise NotFoundError(f"unknown playbook code: {code}")
        if isinstance(entry, str):
            return entry
        return render_playbook_markdown(entry) if callable(render_playbook_markdown) else str(entry)

    def suites(self) -> list[str]:
        """List known suite names (project + built-ins when available)."""

        names: list[str] = []
        project = self._project
        if project and getattr(project, "suites", None):
            suites_val = project.suites
            if isinstance(suites_val, dict):
                names.extend(sorted(suites_val.keys()))
            else:
                for item in suites_val:
                    name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
                    if name:
                        names.append(str(name))
        try:
            for item in list_builtin_suites():
                if isinstance(item, dict) and item.get("name"):
                    names.append(str(item["name"]))
                elif isinstance(item, str):
                    names.append(item)
        except Exception:  # noqa: BLE001 — built-ins optional
            pass
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    # ------------------------------------------------------------------
    # low-level passthroughs (layer C on the client)
    # ------------------------------------------------------------------

    def observe(
        self,
        backend: str,
        model: str | None = None,
        scenario: Scenario | None = None,
        *,
        base_url: str | None = None,
        tokenizer_only: bool | None = None,
    ) -> ObservationTrace:
        """Observe a single backend (low-level)."""

        resolved = self.resolve(model=model, backends=backend, tokenizer_only=tokenizer_only)
        sc = self._prepare_scenario(scenario, resolved)
        model_id = sc.model.id if sc.model and sc.model.id else resolved.model
        url = base_url or self.backend_urls.get(backend) or resolved.backend_urls.get(backend)
        return observe_backend(
            backend,
            model_id,
            sc,
            base_url=url,
            tokenizer_only=resolved.tokenizer_only,
        )

    def compare_traces(
        self,
        left: ObservationTrace,
        right: ObservationTrace,
        scenario: Scenario | None = None,
        *,
        policy: str | None = None,
    ) -> dict[str, Any]:
        return compare_traces(left, right, scenario, policy=policy or self.policy)

    def diagnose(self, traces: Sequence[ObservationTrace]) -> Any:
        return diagnose_traces(list(traces))

    def make_scenario(self, **kwargs: Any) -> Scenario:
        return make_scenario(**kwargs)

    def load_scenarios(self, path: str | Path) -> list[Scenario]:
        return load_scenario_file(path)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _runs_dir(self) -> Path:
        if self.runs_dir is not None:
            return Path(self.runs_dir)
        if self._project is not None:
            return Path(self._project.runs_dir)
        found = find_project_file()
        if found:
            return Path(load_project(found).runs_dir)
        return Path(".eleanity/runs")
