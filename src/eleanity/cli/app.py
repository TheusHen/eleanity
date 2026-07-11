from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eleanity.adapters import adapter_for, available_adapters
from eleanity.cli.backends_check import check_backends, ensure_local_dep
from eleanity.cli.errors import (
    config_error,
    run_not_found,
    unknown_backend,
)
from eleanity.cli.exitcodes import EXIT_CONFIG, EXIT_DIVERGENT, EXIT_OK, exit_from_batch
from eleanity.cli.output import emit_compare_result, emit_error
from eleanity.cli.report_text import render_text_report
from eleanity.cli.resolve import (
    DEFAULT_MODEL,
    apply_scenario_overrides,
    load_project_optional,
    resolve_compare,
)
from eleanity.cli.stdio import configure_cli_stdio
from eleanity.config import find_project_file, load_project, write_default_project
from eleanity.core.demo import render_template_divergence_demo, run_template_divergence_demo
from eleanity.core.engine import CompareEngine
from eleanity.core.golden import golden_gate, save_golden
from eleanity.core.pull import pull_model
from eleanity.core.runs_index import diff_runs, list_runs, load_run
from eleanity.fingerprints.gguf import inspect_gguf
from eleanity.models.schemas import ObservationTrace, ParityProfile, Scenario
from eleanity.playbook import get_playbook_entry, render_playbook_markdown
from eleanity.reporters.sarif import write_sarif
from eleanity.scenarios import load_scenarios
from eleanity.scenarios.suites import list_builtin_suites, load_suite
from eleanity.version import __version__

configure_cli_stdio()

app = typer.Typer(
    help="Same model. Same input. Find the first divergence. (CLI-first parity diagnostics)",
    no_args_is_help=True,
)
runs_app = typer.Typer(help="Inspect local run history")
app.add_typer(runs_app, name="runs")
console = Console()

FormatOpt = typer.Option("text", "--format", help="text | json | quiet")


def _disk_free_gb(path: Path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / (1024**3), 2)
    except OSError:
        return None


def _apply_offline(offline: bool) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _engine_from_resolved(resolved, *, parallel: bool | None = None) -> CompareEngine:
    return CompareEngine(
        project=resolved.project,
        runs_dir=resolved.runs_dir,
        max_workers=resolved.workers,
        parallel=resolved.parallel if parallel is None else parallel,
        tokenizer_only=resolved.tokenizer_only,
        backend_profiles=resolved.backend_profiles,
        gates=resolved.gates if resolved.apply_gates else [],
    )


def _load_scenarios_for_compare(resolved, name: str | None) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if resolved.suite:
        scenarios = load_suite(resolved.suite, project=resolved.project)
    elif resolved.scenario_path is not None:
        scenarios = load_scenarios(resolved.scenario_path)
    elif resolved.project and resolved.project.default_suite:
        scenarios = load_suite(resolved.project.default_suite, project=resolved.project)

    if name and scenarios:
        scenarios = [s for s in scenarios if s.name == name]
        if not scenarios:
            raise config_error(f"scenario '{name}' not found", hint="eleanity suites")
    return scenarios


def _prepare_scenario(scenario: Scenario | None, resolved) -> Scenario:
    if scenario is None:
        scenario = Scenario(
            name="compare",
            messages=[{"role": "user", "content": "Hello"}],
            observe=resolved.observe or ["artifact", "template", "special_tokens", "tokens", "generation"],
            parity_profile=resolved.policy,
        )
    scenario = apply_scenario_overrides(scenario, resolved)
    # Prefer CLI model when scenario has no model id
    if scenario.model is None or not scenario.model.id:
        from eleanity.models.schemas import ModelSpec

        scenario = scenario.model_copy(
            update={
                "model": ModelSpec(
                    id=resolved.model,
                    tokenizer_only=resolved.tokenizer_only,
                )
            }
        )
    return scenario


@app.command()
def demo(format: str = FormatOpt) -> None:
    """Run a deterministic offline first-divergence demonstration."""

    result = run_template_divergence_demo()
    if format == "json":
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    elif format == "quiet":
        typer.echo(
            f"status={result['status']} first_divergence={result['first_divergence']} "
            f"probable_cause={result['probable_cause']}"
        )
    else:
        console.print(render_template_divergence_demo(result))


@app.command()
def doctor(
    probe_backends: bool = typer.Option(False, "--check-backends", help="Probe selected backends"),
    backends: str = typer.Option("", help="Backends to probe (default: project or common HTTP)"),
    format: str = FormatOpt,
) -> None:
    """Verify Python, optional deps, GPU, disk, cache, and optionally backend health."""

    project = load_project_optional(None)
    table = Table(title=f"Eleanity {__version__} · doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_row("Python", platform.python_version())
    table.add_row("OS", platform.platform())
    table.add_row("CPU arch", platform.machine())
    for module, label in (
        ("transformers", "transformers"),
        ("vllm", "vllm"),
        ("llama_cpp", "llama-cpp-python"),
        ("torch", "torch"),
        ("accelerate", "accelerate"),
        ("httpx", "httpx"),
        ("huggingface_hub", "huggingface_hub"),
    ):
        ok = importlib.util.find_spec(module) is not None
        table.add_row(label, "installed" if ok else "not installed — uv sync --extra …")
    table.add_row("adapters", ", ".join(available_adapters()))
    table.add_row("Docker", "available" if shutil.which("docker") else "not found")
    free = _disk_free_gb(Path.cwd())
    table.add_row("disk free (cwd)", f"{free} GB" if free is not None else "—")
    hf_home = (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or str(Path.home() / ".cache" / "huggingface")
    )
    table.add_row("HF cache", hf_home)
    table.add_row("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE") or "0")
    for var in (
        "ELEANITY_VLLM_URL",
        "ELEANITY_LLAMACPP_URL",
        "ELEANITY_OLLAMA_URL",
        "ELEANITY_SGLANG_URL",
        "ELEANITY_TGI_URL",
        "ELEANITY_OPENAI_URL",
        "CUDA_VISIBLE_DEVICES",
        "HF_TOKEN",
    ):
        value = os.environ.get(var)
        if "TOKEN" in var or "KEY" in var:
            table.add_row(var, "set (redacted)" if value else "unset")
        else:
            table.add_row(var, value or "unset")
    try:
        import torch

        table.add_row("GPU", "available" if torch.cuda.is_available() else "not available")
        if torch.cuda.is_available():
            table.add_row("CUDA", str(getattr(torch.version, "cuda", "unknown")))
            table.add_row("GPU name", torch.cuda.get_device_name(0))
    except ImportError:
        table.add_row("GPU", "torch not installed")
    proj = find_project_file()
    table.add_row("eleanity.yaml", str(proj) if proj else "not found (eleanity init)")
    if format != "json":
        console.print(table)

    backend_results = []
    if probe_backends:
        names = [b.strip() for b in backends.split(",") if b.strip()]
        if not names:
            names = (
                list(project.backends)
                if project and project.backends
                else [
                    "transformers",
                    "vllm",
                    "llamacpp",
                    "ollama",
                ]
            )
        resolved = resolve_compare(backends=",".join(names))
        backend_results = check_backends(names, resolved=resolved)
        btable = Table(title="Backend health")
        btable.add_column("Backend")
        btable.add_column("OK")
        btable.add_column("Detail")
        btable.add_column("ms")
        for item in backend_results:
            btable.add_row(
                item.name,
                "yes" if item.ok else "no",
                item.detail,
                f"{item.latency_ms:.1f}" if item.latency_ms is not None else "—",
            )
        if format != "json":
            console.print(btable)

    if format == "json":
        print(
            json.dumps(
                {
                    "ok": True,
                    "version": __version__,
                    "python": platform.python_version(),
                    "adapters": available_adapters(),
                    "project": str(proj) if proj else None,
                    "backends": [
                        {"name": b.name, "ok": b.ok, "detail": b.detail, "latency_ms": b.latency_ms}
                        for b in backend_results
                    ],
                },
                indent=2,
            )
        )


@app.command(name="init")
def init_project(
    path: Path = typer.Option(Path("eleanity.yaml"), help="Output path"),
    force: bool = typer.Option(False, help="Overwrite existing file"),
    scenarios: bool = typer.Option(True, help="Create scenarios/ starter files"),
    format: str = FormatOpt,
) -> None:
    """Scaffold eleanity.yaml and optional starter scenarios."""

    try:
        written = write_default_project(path, force=force)
    except FileExistsError as error:
        code = emit_error(config_error(str(error), hint="use --force"), fmt=format)  # type: ignore[arg-type]
        raise typer.Exit(code) from error
    if scenarios:
        scenario_dir = Path("scenarios")
        scenario_dir.mkdir(exist_ok=True)
        starter = scenario_dir / "basic.yaml"
        if force or not starter.exists():
            starter.write_text(
                """\
schema_version: "0.1"
name: local-basic
description: Starter parity scenario
model:
  id: Qwen/Qwen2.5-0.5B-Instruct
  revision: main
  trust_remote_code: false
messages:
  - role: system
    content: You are a helpful assistant.
  - role: user
    content: Explain recursion in one sentence.
parameters:
  temperature: 0
  max_tokens: 32
  seed: 42
generation:
  add_generation_prompt: true
observe: [artifact, template, special_tokens, tokens, generation]
parity_policy: strict
""",
                encoding="utf-8",
            )
    Path(".eleanity/runs").mkdir(parents=True, exist_ok=True)
    Path(".eleanity/golden").mkdir(parents=True, exist_ok=True)
    if format == "json":
        print(json.dumps({"ok": True, "path": str(written)}))
    else:
        console.print(f"[green]Wrote[/green] {written}")
        console.print("Next: eleanity doctor && eleanity compare --backends fake,fake")


@app.command()
def inspect(
    model: str = typer.Argument(..., help="HF model id, local path, or GGUF file"),
    backend: str = typer.Option("transformers"),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    format: str = FormatOpt,
) -> None:
    """Display fingerprint, special tokens and capabilities."""

    try:
        if model.lower().endswith(".gguf") and Path(model).is_file():
            payload = inspect_gguf(model)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        if backend not in available_adapters():
            raise unknown_backend(backend, available_adapters())
        if backend == "transformers":
            ensure_local_dep("transformers")
        adapter = adapter_for(backend, model, tokenizer_only=tokenizer_only)
        payload = {
            "fingerprint": adapter.fingerprint(model).model_dump(mode="json"),
            "capabilities": adapter.capabilities.model_dump(mode="json"),
        }
        special = getattr(adapter, "special_tokens", None)
        if callable(special):
            payload["special_tokens"] = special().model_dump(mode="json")
        health = getattr(adapter, "healthcheck", None)
        if callable(health):
            payload["healthcheck"] = health().model_dump(mode="json")
        if format == "quiet":
            print(f"model={model} backend={backend} ok=true")
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def pull(
    model: str = typer.Argument(...),
    revision: str = typer.Option("main"),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    trust_remote_code: bool = typer.Option(False),
    offline: bool = typer.Option(False, help="Fail if not already cached"),
    format: str = FormatOpt,
) -> None:
    """Download model/tokenizer into the local Hugging Face cache."""

    try:
        _apply_offline(offline)
        if trust_remote_code and format == "text":
            console.print("[yellow]WARNING: trust_remote_code=True enables remote code execution[/yellow]")
        result = pull_model(
            model,
            revision=revision,
            tokenizer_only=tokenizer_only,
            trust_remote_code=trust_remote_code,
        )
        if format == "quiet":
            print(f"ok=true model={model} path={result.get('local_path')}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def compare(
    model: str | None = typer.Option(None, help=f"Model id (default from yaml or {DEFAULT_MODEL})"),
    backends: str | None = typer.Option(None, help="Comma-separated backends"),
    baseline: str | None = typer.Option(None),
    scenario: Path | None = typer.Option(None),
    name: str | None = typer.Option(None, help="Scenario name inside YAML"),
    suite: str | None = typer.Option(None, help="Named suite"),
    config: Path | None = typer.Option(None, help="eleanity.yaml"),
    profile: str | None = typer.Option(None, help="Named profile from eleanity.yaml"),
    policy: str | None = typer.Option(None, help="strict|quantized|functional|api_conformance"),
    observe: str | None = typer.Option(None, help="Comma-separated layers"),
    redact_prompts: bool | None = typer.Option(None, "--redact-prompts/--no-redact-prompts"),
    parallel: bool | None = typer.Option(None, "--parallel/--no-parallel"),
    workers: int | None = typer.Option(None),
    tokenizer_only: bool | None = typer.Option(None, "--tokenizer-only/--no-tokenizer-only"),
    no_gates: bool = typer.Option(False, "--no-gates"),
    backend_url: list[str] | None = typer.Option(None, help="name=url (repeatable)"),
    offline: bool = typer.Option(False, help="HF offline mode"),
    check_backends: bool = typer.Option(False, "--check-backends"),
    golden: Path | None = typer.Option(None, help="Golden trace to check after compare"),
    repetitions: int = typer.Option(1, help="Self-consistency repetitions before cross-compare"),
    require_self_consistency: bool = typer.Option(
        False, "--require-self-consistency", help="Stabilize each backend before A vs B"
    ),
    no_store: bool = typer.Option(False, "--no-store", help="Do not persist run artifacts"),
    redact_input: bool = typer.Option(False, "--redact-input"),
    redact_output: bool = typer.Option(False, "--redact-output"),
    hash_content: bool = typer.Option(False, "--hash-content"),
    allow_remote: bool = typer.Option(False, "--allow-remote"),
    retention: str | None = typer.Option(None, help="Retention window e.g. 24h, 7d"),
    format: str = FormatOpt,
) -> None:
    """Run the same scenario across backends. Reads eleanity.yaml when flags omitted."""

    try:
        from eleanity.core.privacy import apply_retention, enforce_no_remote, privacy_from_flags
        from eleanity.core.stabilize import compare_with_stability

        privacy = privacy_from_flags(
            no_store=no_store,
            redact_input=redact_input,
            redact_output=redact_output,
            hash_content=hash_content,
            allow_remote=allow_remote,
            retention=retention,
            redact_prompts=bool(redact_prompts),
        )
        enforce_no_remote(privacy)

        # Profile overlay
        project = load_project_optional(config)
        if profile and project:
            p = project.profile(profile)
            backends = backends or (",".join(p["backends"]) if p.get("backends") else None)
            policy = policy or p.get("policy")
            if p.get("observe") and not observe:
                observe = ",".join(p["observe"]) if isinstance(p["observe"], list) else p["observe"]
            if tokenizer_only is None and "tokenizer_only" in p:
                tokenizer_only = bool(p["tokenizer_only"])

        resolved = resolve_compare(
            model=model,
            backends=backends,
            baseline=baseline,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            redact_prompts=True if privacy.redact_input else redact_prompts,
            parallel=parallel,
            workers=workers,
            config=config,
            backend_url=backend_url,
            no_gates=no_gates,
            offline=offline,
            suite=suite,
            scenario=scenario,
            golden=golden,
            check_golden=bool(golden),
        )

        _apply_offline(resolved.offline)
        for b in resolved.backends:
            if b not in available_adapters():
                raise unknown_backend(b, available_adapters())
        if check_backends:
            from eleanity.cli.backends_check import check_backends as probe_backends

            probe_backends(
                resolved.backends,
                model=resolved.model,
                resolved=resolved,
                require_healthy=False,
            )

        if "transformers" in resolved.backends:
            ensure_local_dep("transformers")

        scenarios = _load_scenarios_for_compare(resolved, name)
        engine = _engine_from_resolved(resolved)
        if retention:
            apply_retention(resolved.runs_dir, retention)

        if scenarios and (suite or (resolved.suite and len(scenarios) > 1)):
            # batch suite
            code = 0
            rows = []
            for sc in scenarios:
                sc2 = _prepare_scenario(sc, resolved)
                model_id = sc2.model.id if sc2.model and sc2.model.id else resolved.model
                result = engine.compare(
                    model_id,
                    resolved.backends,
                    scenario=sc2,
                    baseline_backend=resolved.baseline,
                    redact_prompts=resolved.redact_prompts,
                    tokenizer_only=resolved.tokenizer_only,
                )
                exit_c = emit_compare_result(
                    fmt="quiet" if format == "text" else format,  # type: ignore[arg-type]
                    traces=result.traces,
                    diagnosis=result.diagnosis,
                    run_id=result.run_id,
                    scenario=sc2,
                    model=model_id,
                    policy=resolved.policy.value,
                    gate_evaluation=result.gate_evaluation,
                    timings=result.timings,
                    comparisons=result.comparisons,
                    quiet_layers=True,
                )
                rows.append({"scenario": sc2.name, "run_id": result.run_id, "exit": exit_c})
                code = max(code, exit_c)
            if format == "json":
                print(json.dumps({"ok": code == 0, "results": rows}, indent=2))
            elif format == "text":
                table = Table(title="Suite results")
                table.add_column("scenario")
                table.add_column("run_id")
                table.add_column("exit")
                for row in rows:
                    table.add_row(row["scenario"], row["run_id"][:8], str(row["exit"]))
                console.print(table)
            raise typer.Exit(code)

        selected = scenarios[0] if scenarios else None
        selected = _prepare_scenario(selected, resolved)
        model_id = selected.model.id if selected.model and selected.model.id else resolved.model

        stability_report = None
        if require_self_consistency or repetitions > 1:
            result, stability_report = compare_with_stability(
                engine,
                model_id,
                resolved.backends,
                scenario=selected,
                repetitions=max(2, repetitions),
                require_self_consistency=require_self_consistency,
            )
            if result is None:
                if format == "json":
                    print(json.dumps({"ok": False, "stability": stability_report.to_dict()}, indent=2))
                else:
                    console.print(stability_report.conclusion)
                raise typer.Exit(EXIT_CONFIG)
            if format == "text" and stability_report:
                console.print(f"[bold]Stability[/bold]: {stability_report.conclusion}")
        else:
            result = engine.compare(
                model_id,
                resolved.backends,
                scenario=selected,
                baseline_backend=resolved.baseline,
                redact_prompts=resolved.redact_prompts,
                tokenizer_only=resolved.tokenizer_only,
            )

        if privacy.no_store and result.path.exists():
            # Keep only ephemeral summary in memory; remove on-disk run if requested
            import shutil

            shutil.rmtree(result.path, ignore_errors=True)

        if golden or resolved.check_golden:
            gpath = golden or (
                Path(resolved.project.golden_file) if resolved.project and resolved.project.golden_file else None
            )
            if gpath and gpath.is_file():
                report = golden_gate(
                    result.traces[0],
                    gpath,
                    selected,
                    layers=resolved.observe or ["template", "tokens"],
                )
                if format == "text":
                    console.print(f"Golden: {'PASS' if report['passed'] else 'FAIL'} {report.get('divergent_layers')}")
                if not report["passed"]:
                    raise typer.Exit(EXIT_DIVERGENT)

        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=result.traces,
            diagnosis=result.diagnosis,
            run_id=result.run_id,
            scenario=selected,
            model=model_id,
            policy=resolved.policy.value,
            gate_evaluation=result.gate_evaluation,
            timings=result.timings,
            comparisons=result.comparisons,
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="test")
def test_scenario(
    path: Path = typer.Argument(..., help="Scenario YAML, directory, or suite name"),
    model: str | None = typer.Option(None),
    backends: str | None = typer.Option(None),
    config: Path | None = typer.Option(None),
    policy: str | None = typer.Option(None),
    observe: str | None = typer.Option(None),
    tokenizer_only: bool | None = typer.Option(None, "--tokenizer-only/--no-tokenizer-only"),
    no_gates: bool = typer.Option(False, "--no-gates"),
    redact_prompts: bool | None = typer.Option(None, "--redact-prompts/--no-redact-prompts"),
    golden: Path | None = typer.Option(None),
    format: str = FormatOpt,
    fail_fast: bool = typer.Option(False, "--fail-fast"),
) -> None:
    """Execute every scenario in a YAML file, directory, or named suite."""

    try:
        resolved = resolve_compare(
            model=model,
            backends=backends,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            redact_prompts=redact_prompts,
            config=config,
            no_gates=no_gates,
            golden=golden,
            check_golden=bool(golden),
        )

        if path.is_file() or path.is_dir():
            files = [path] if path.is_file() else sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
            scenarios: list[Scenario] = []
            for file_path in files:
                scenarios.extend(load_scenarios(file_path))
        else:
            scenarios = load_suite(str(path), project=resolved.project)

        engine = _engine_from_resolved(resolved)
        code = 0
        rows = []
        for scenario in scenarios:
            sc = _prepare_scenario(scenario, resolved)
            model_id = sc.model.id if sc.model and sc.model.id else resolved.model
            backend_list = sc.backends or resolved.backends
            result = engine.compare(
                model_id,
                backend_list,
                sc,
                baseline_backend=resolved.baseline if resolved.baseline in backend_list else backend_list[0],
                redact_prompts=resolved.redact_prompts,
                tokenizer_only=resolved.tokenizer_only,
            )
            status = getattr(result.diagnosis, "status", None)
            status_v = status.value if hasattr(status, "value") else str(status)
            exit_c = emit_compare_result(
                fmt="quiet",
                traces=result.traces,
                diagnosis=result.diagnosis,
                run_id=result.run_id,
                scenario=sc,
                model=model_id,
                policy=resolved.policy.value,
                gate_evaluation=result.gate_evaluation,
                timings=result.timings,
                quiet_layers=True,
            )
            rows.append({"name": sc.name, "status": status_v, "run_id": result.run_id, "exit": exit_c})
            code = max(code, exit_c)
            if fail_fast and exit_c:
                break

        if format == "json":
            print(json.dumps({"ok": code == 0, "results": rows}, indent=2))
        elif format == "text":
            table = Table(title="Test results")
            table.add_column("scenario")
            table.add_column("status")
            table.add_column("run_id")
            table.add_column("exit")
            for row in rows:
                table.add_row(row["name"], row["status"], row["run_id"][:8], str(row["exit"]))
            console.print(table)
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def report(
    run_id: str = typer.Argument(...),
    format: str = typer.Option("text", help="text | json | sarif"),
    no_redact: bool = typer.Option(False, "--no-redact", help="Show prompt snippets in text mode"),
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
) -> None:
    """Render a persisted run as text (default), json, or SARIF. CLI-only product — no HTML export."""

    try:
        project = find_project_file()
        if project:
            runs_dir = Path(load_project(project).runs_dir)
        try:
            data = load_run(run_id, runs_dir)
        except FileNotFoundError as error:
            raise run_not_found(run_id) from error

        result_path = runs_dir / str(data.get("run_id") or run_id) / "result.json"
        if format == "html":
            raise config_error(
                "HTML export is not part of the CLI product",
                hint="use: eleanity report <run-id> --format text|json|sarif",
            )
        if format == "sarif":
            print(write_sarif(result_path))
        elif format == "json":
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(render_text_report(data, redact=not no_redact))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format if format in {"text", "json", "quiet"} else "text"))  # type: ignore[arg-type]


@app.command()
def ci(
    baseline: str = typer.Option(..., help="Baseline model id"),
    candidate: str = typer.Option(..., help="Candidate model id"),
    backend: str = typer.Option("transformers"),
    scenario: Path | None = typer.Option(None),
    junit: Path | None = typer.Option(None),
    config: Path | None = typer.Option(None),
    policy: str | None = typer.Option(None),
    observe: str | None = typer.Option(None),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    no_gates: bool = typer.Option(False, "--no-gates"),
    golden: Path | None = typer.Option(None),
    format: str = FormatOpt,
) -> None:
    """CI gate between two models on one backend. Exit 0/1/2."""

    try:
        resolved = resolve_compare(
            model=baseline,
            backends=backend,
            policy=policy,
            observe=observe,
            tokenizer_only=tokenizer_only,
            config=config,
            no_gates=no_gates,
            golden=golden,
            check_golden=bool(golden),
        )
        if backend == "transformers":
            ensure_local_dep("transformers")
        selected = load_scenarios(scenario)[0] if scenario else None
        selected = _prepare_scenario(selected, resolved)
        engine = _engine_from_resolved(resolved, parallel=False)
        run_id, comparisons, diagnosis, gate_eval = engine.ci(
            baseline,
            candidate,
            backend,
            scenario=selected,
            junit_path=junit,
            tokenizer_only=tokenizer_only,
        )
        # Build fake dual traces already in result
        data = load_run(run_id, resolved.runs_dir)
        traces = [ObservationTrace.model_validate(t) for t in data.get("traces") or []]
        if golden and golden.is_file() and traces:
            report = golden_gate(traces[0], golden, selected or _prepare_scenario(None, resolved))
            if not report["passed"] and format == "text":
                console.print(f"Golden FAIL: {report['divergent_layers']}")
            if not report["passed"]:
                raise typer.Exit(EXIT_DIVERGENT)

        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=traces,
            diagnosis=diagnosis,
            run_id=run_id,
            scenario=selected,
            model=f"{baseline}→{candidate}",
            policy=resolved.policy.value,
            gate_evaluation=gate_eval,
            comparisons={backend: {k: v.model_dump(mode="json") for k, v in comparisons.items()}},
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="compare-endpoints")
def compare_endpoints(
    endpoint_a: str = typer.Argument(...),
    endpoint_b: str = typer.Argument(...),
    model: str = typer.Option("default"),
    format: str = FormatOpt,
    observe: str = typer.Option("generation,api,streaming"),
) -> None:
    """Compare two OpenAI-compatible HTTP endpoints (api_conformance)."""

    try:
        os.environ["ELEANITY_VLLM_URL"] = endpoint_a.rstrip("/")
        os.environ["ELEANITY_LLAMACPP_URL"] = endpoint_b.rstrip("/")
        scenario = Scenario(
            name="endpoint-compare",
            messages=[{"role": "user", "content": "Hello from Eleanity"}],
            parameters={"temperature": 0, "max_tokens": 16, "seed": 42},
            observe=[x.strip() for x in observe.split(",") if x.strip()],
            parity_policy=ParityProfile.API_CONFORMANCE,
        )
        engine = CompareEngine()
        result = engine.compare(model, ["vllm", "llamacpp"], scenario=scenario)
        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=result.traces,
            diagnosis=result.diagnosis,
            run_id=result.run_id,
            scenario=scenario,
            model=model,
            policy="api_conformance",
            gate_evaluation=result.gate_evaluation,
            timings=result.timings,
            comparisons=result.comparisons,
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def migrate(
    from_backend: str = typer.Option(..., "--from", help="Source backend"),
    to_backend: str = typer.Option(..., "--to", help="Target backend"),
    model: str = typer.Option(DEFAULT_MODEL),
    scenario: Path | None = typer.Option(None),
    suite: str | None = typer.Option(None),
    policy: str = typer.Option("strict"),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    backend_url: list[str] | None = typer.Option(None),
    format: str = FormatOpt,
    config: Path | None = typer.Option(None),
) -> None:
    """Migration flow: compare from_backend → to_backend on the same model."""

    backends = f"{from_backend},{to_backend}"
    raise typer.Exit(
        _dispatch_named_flow(
            model=model,
            backends=backends,
            baseline=from_backend,
            scenario=scenario,
            suite=suite,
            policy=policy,
            tokenizer_only=tokenizer_only,
            backend_url=backend_url,
            format=format,
            config=config,
            flow="migrate",
        )
    )


@app.command()
def promote(
    baseline: str = typer.Option(..., help="Baseline model or checkpoint"),
    candidate: str = typer.Option(..., help="Candidate model or checkpoint"),
    backend: str = typer.Option("transformers"),
    policy: str = typer.Option("quantized"),
    scenario: Path | None = typer.Option(None),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    format: str = FormatOpt,
    config: Path | None = typer.Option(None),
) -> None:
    """Promotion flow: baseline model vs candidate on one backend (quantized-friendly)."""

    try:
        # Reuse ci path
        resolved = resolve_compare(
            model=baseline,
            backends=backend,
            policy=policy,
            tokenizer_only=tokenizer_only,
            config=config,
        )
        selected = load_scenarios(scenario)[0] if scenario else None
        selected = _prepare_scenario(selected, resolved)
        if backend == "transformers":
            ensure_local_dep("transformers")
        engine = _engine_from_resolved(resolved, parallel=False)
        run_id, comparisons, diagnosis, gate_eval = engine.ci(
            baseline,
            candidate,
            backend,
            scenario=selected,
            tokenizer_only=tokenizer_only,
        )
        data = load_run(run_id, resolved.runs_dir)
        traces = [ObservationTrace.model_validate(t) for t in data.get("traces") or []]
        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=traces,
            diagnosis=diagnosis,
            run_id=run_id,
            scenario=selected,
            model=f"{baseline}→{candidate}",
            policy=policy,
            gate_evaluation=gate_eval,
            comparisons={backend: {k: v.model_dump(mode="json") for k, v in comparisons.items()}},
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="vendor-check")
def vendor_check(
    endpoint: str = typer.Option(..., help="OpenAI-compatible base URL"),
    model: str = typer.Option("default", help="Vendor/API model id"),
    reference: str = typer.Option("transformers", help="Local reference backend"),
    reference_model: str | None = typer.Option(None, help="Local model id for reference"),
    format: str = FormatOpt,
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    observe: str | None = typer.Option(None, help="Override observe layers"),
) -> None:
    """Vendor check: local reference backend vs remote OpenAI-compatible endpoint."""

    try:
        from eleanity.config.project import BackendProfile

        os.environ["ELEANITY_OPENAI_URL"] = endpoint.rstrip("/")
        ref_model = reference_model or model
        observe_layers = (
            [x.strip() for x in observe.split(",") if x.strip()]
            if observe
            else (["artifact", "template", "tokens"] if tokenizer_only else ["artifact", "generation", "api"])
        )
        scenario = Scenario(
            name="vendor-check",
            messages=[{"role": "user", "content": "Reply with one short sentence."}],
            parameters={"temperature": 0, "max_tokens": 32, "seed": 42},
            observe=observe_layers,
            parity_policy=ParityProfile.API_CONFORMANCE if not tokenizer_only else ParityProfile.STRICT,
        )
        backends = [reference, "openai"]
        if reference == "transformers":
            ensure_local_dep("transformers")
        profiles = {
            reference: BackendProfile(adapter=reference, model=ref_model),
            "openai": BackendProfile(
                adapter="openai",
                model=model,
                base_url=endpoint.rstrip("/"),
            ),
        }
        engine = CompareEngine(
            tokenizer_only=tokenizer_only,
            backend_profiles=profiles,
        )
        result = engine.compare(
            ref_model,
            backends,
            scenario=scenario,
            baseline_backend=reference,
            tokenizer_only=tokenizer_only,
        )
        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=result.traces,
            diagnosis=result.diagnosis,
            run_id=result.run_id,
            scenario=scenario,
            model=f"{ref_model}→{model}",
            policy=scenario.parity_profile.value,
            gate_evaluation=result.gate_evaluation,
            timings=result.timings,
            comparisons=result.comparisons,
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


def _dispatch_named_flow(
    *,
    model: str,
    backends: str,
    baseline: str,
    scenario: Path | None,
    suite: str | None,
    policy: str,
    tokenizer_only: bool,
    backend_url: list[str] | None,
    format: str,
    config: Path | None,
    flow: str,
) -> int:
    try:
        resolved = resolve_compare(
            model=model,
            backends=backends,
            baseline=baseline,
            policy=policy,
            tokenizer_only=tokenizer_only,
            config=config,
            backend_url=backend_url,
            suite=suite,
            scenario=scenario,
        )
        _apply_offline(resolved.offline)
        if "transformers" in resolved.backends:
            ensure_local_dep("transformers")
        scenarios = _load_scenarios_for_compare(resolved, None)
        selected = _prepare_scenario(scenarios[0] if scenarios else None, resolved)
        engine = _engine_from_resolved(resolved)
        result = engine.compare(
            model,
            resolved.backends,
            scenario=selected,
            baseline_backend=baseline,
            tokenizer_only=tokenizer_only,
            redact_prompts=resolved.redact_prompts,
        )
        if format == "text":
            console.print(f"[bold]flow[/bold]={flow}")
        return emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=result.traces,
            diagnosis=result.diagnosis,
            run_id=result.run_id,
            scenario=selected,
            model=model,
            policy=policy,
            gate_evaluation=result.gate_evaluation,
            timings=result.timings,
            comparisons=result.comparisons,
        )
    except Exception as error:
        return emit_error(error, fmt=format)  # type: ignore[arg-type]


@app.command()
def batch(
    models: str = typer.Option(..., help="Comma-separated model ids"),
    backends: str = typer.Option("fake,fake"),
    suite: str = typer.Option("generic-chat"),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    config: Path | None = typer.Option(None),
    policy: str | None = typer.Option(None),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    format: str = FormatOpt,
) -> None:
    """Batch multi-model report with table summary."""

    try:
        from eleanity.core.batch_report import run_multi_model_batch

        resolved = resolve_compare(
            backends=backends,
            policy=policy,
            tokenizer_only=tokenizer_only,
            config=config,
            suite=suite,
        )
        scenarios = load_suite(suite, project=resolved.project)
        model_list = [item.strip() for item in models.split(",") if item.strip()]
        backend_list = resolved.backends
        jobs = []
        for m in model_list:
            for sc in scenarios:
                jobs.append((m, backend_list, apply_scenario_overrides(sc, resolved)))

        # fail-fast: run sequentially via engine if needed
        if fail_fast:
            engine = _engine_from_resolved(resolved)
            failed = 0
            had_error = False
            rows = []
            for m, b, sc in jobs:
                result = engine.compare(m, b, scenario=sc, tokenizer_only=tokenizer_only)
                status = getattr(result.diagnosis, "status", None)
                value = status.value if hasattr(status, "value") else str(status)
                if value == "ERROR":
                    had_error = True
                    failed += 1
                    rows.append({"model": m, "scenario": sc.name, "status": value, "run_id": result.run_id})
                    break
                if value == "DIVERGENT" or (result.gate_evaluation and not result.gate_evaluation.passed):
                    failed += 1
                rows.append({"model": m, "scenario": sc.name, "status": value, "run_id": result.run_id})
            code = exit_from_batch(failed, had_error=had_error)
        else:
            engine = _engine_from_resolved(resolved)
            report = run_multi_model_batch(jobs, engine=engine, tokenizer_only=tokenizer_only)
            rows = report.summary.get("rows") or []
            code = exit_from_batch(int(report.summary.get("failed") or 0))
            if format == "text":
                console.print(f"[bold]batch[/bold] {report.batch_id}")
                console.print(f"report: {report.path / 'batch.md'}")

        if format == "json":
            print(json.dumps({"ok": code == 0, "results": rows}, indent=2))
        elif format == "quiet":
            print(f"failed={sum(1 for r in rows if r.get('status') == 'DIVERGENT')} jobs={len(rows)}")
        else:
            table = Table(title="Batch results")
            table.add_column("model")
            table.add_column("scenario")
            table.add_column("status")
            table.add_column("run_id")
            for row in rows:
                table.add_row(
                    str(row.get("model")),
                    str(row.get("scenario")),
                    str(row.get("status")),
                    str(row.get("run_id", ""))[:8],
                )
            console.print(table)
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="suites")
def suites_cmd(format: str = FormatOpt) -> None:
    """List built-in and project scenario suites."""

    items = list_builtin_suites()
    project = find_project_file()
    if project:
        for suite in load_project(project).suites:
            items.append({"name": suite.name, "path": suite.path})
    if format == "json":
        print(json.dumps(items, indent=2))
    else:
        table = Table(title="Suites")
        table.add_column("Name")
        table.add_column("Path")
        for item in items:
            table.add_row(item["name"], item["path"])
        console.print(table)


@app.command()
def playbook(code: str = typer.Argument(...), format: str = FormatOpt) -> None:
    """Show remediation playbook for a diagnosis code."""

    entry = get_playbook_entry(code)
    if not entry:
        raise typer.Exit(emit_error(config_error(f"no playbook for {code}"), fmt=format))  # type: ignore[arg-type]
    if format == "json":
        print(json.dumps({"code": code, **entry}, indent=2))
    else:
        console.print(render_playbook_markdown(code))


@app.command(name="gguf")
def gguf_cmd(
    path: Path = typer.Argument(...),
    deep: bool = typer.Option(True),
    format: str = FormatOpt,
) -> None:
    """Inspect GGUF metadata."""

    data = inspect_gguf(path, deep=deep)
    if format == "quiet":
        print(f"ok={data.get('ok')} arch={data.get('architecture')} hash={data.get('parity_fingerprint')}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command(name="snapshot")
@app.command(name="save-golden")
def save_golden_cmd(
    run_id: str = typer.Argument(...),
    backend: str = typer.Option(...),
    name: str | None = typer.Option(None),
    golden_dir: Path = typer.Option(Path(".eleanity/golden")),
    format: str = FormatOpt,
) -> None:
    """Pin a trace from a run as a golden baseline (alias: snapshot)."""

    try:
        data = load_run(run_id)
        traces = data.get("traces") or []
        match = next((t for t in traces if t.get("backend") == backend), None)
        if not match:
            raise config_error(f"backend {backend} not in run {run_id}")
        path = save_golden(ObservationTrace.model_validate(match), golden_dir, name=name)
        if format == "json":
            print(json.dumps({"ok": True, "path": str(path)}))
        else:
            console.print(f"[green]Saved golden[/green] {path}")
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="check-golden")
def check_golden_cmd(
    run_id: str = typer.Argument(...),
    golden: Path = typer.Option(..., exists=True),
    backend: str | None = typer.Option(None),
    layers: str = typer.Option("template,tokens,special_tokens"),
    format: str = FormatOpt,
) -> None:
    """Fail if a live run diverges from a golden baseline on selected layers."""

    try:
        data = load_run(run_id)
        traces = data.get("traces") or []
        match = (
            next((t for t in traces if t.get("backend") == backend), None)
            if backend
            else (traces[0] if traces else None)
        )
        if not match:
            raise config_error("trace not found in run")
        scenario_meta = data.get("scenario") or {}
        scenario = Scenario(
            name=scenario_meta.get("name") or "golden-check",
            messages=[{"role": "user", "content": "x"}],
            observe=[item.strip() for item in layers.split(",") if item.strip()],
            parity_profile=scenario_meta.get("parity_profile") or "strict",
            tolerance=scenario_meta.get("tolerance"),
        )
        report = golden_gate(
            ObservationTrace.model_validate(match),
            golden,
            scenario,
            layers=[item.strip() for item in layers.split(",") if item.strip()],
        )
        if format == "json":
            print(json.dumps(report, indent=2))
        elif format == "quiet":
            print(f"passed={report['passed']} divergent={report.get('divergent_layers')}")
        else:
            print(json.dumps(report, indent=2))
        raise typer.Exit(EXIT_OK if report["passed"] else EXIT_DIVERGENT)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def certify(
    backend: str = typer.Option("fake"),
    model: str = typer.Option("demo"),
    format: str = FormatOpt,
) -> None:
    """Certify a runtime adapter (bronze/silver/gold)."""

    try:
        from eleanity.certification import certify_runtime

        report = certify_runtime(adapter_for(backend, model), model=model)
        if format == "quiet":
            print(f"passed={report.passed} level={report.level}")
        else:
            print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(EXIT_OK if report.passed else EXIT_DIVERGENT)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="check-adapter")
def check_adapter_cmd(
    backend: str = typer.Option("fake"),
    model: str = typer.Option("demo"),
    format: str = FormatOpt,
) -> None:
    """Run Adapter SDK compliance tests."""

    try:
        from eleanity.adapters.sdk import check_adapter_compliance

        report = check_adapter_compliance(adapter_for(backend, model), model=model)
        if format == "quiet":
            print(f"passed={report.passed}")
        else:
            print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(EXIT_OK if report.passed else EXIT_DIVERGENT)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def export(
    run_id: str = typer.Argument(...),
    sink: str = typer.Option("local", help="local|mlflow|wandb"),
    destination: Path = typer.Option(Path(".eleanity/exports")),
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
    format: str = FormatOpt,
) -> None:
    """Export redacted run artifacts to a sink."""

    try:
        from eleanity.integrations.artifacts import export_run_artifacts

        data = load_run(run_id, runs_dir)
        run_dir = runs_dir / str(data.get("run_id") or run_id)
        result = export_run_artifacts(run_dir, sink=sink, destination=destination)
        if format == "quiet":
            print(f"ok=true sink={sink}")
        else:
            print(json.dumps(result, indent=2))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@runs_app.command("ls")
def runs_ls(
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
    limit: int = typer.Option(20),
    format: str = FormatOpt,
) -> None:
    """List recent local runs."""

    items = list_runs(runs_dir)[:limit]
    if format == "json":
        print(
            json.dumps(
                [
                    {
                        "run_id": i.run_id,
                        "status": i.status,
                        "scenario": i.scenario,
                        "model": i.model,
                        "backends": i.backends,
                        "duration_ms": i.duration_ms,
                    }
                    for i in items
                ],
                indent=2,
            )
        )
        return
    if format == "quiet":
        for i in items:
            print(f"{i.run_id}\t{i.status}\t{i.scenario}")
        return
    table = Table(title="Eleanity runs")
    table.add_column("run_id")
    table.add_column("status")
    table.add_column("scenario")
    table.add_column("model")
    table.add_column("backends")
    table.add_column("ms")
    for item in items:
        table.add_row(
            item.run_id[:8],
            item.status,
            item.scenario,
            (item.model or "")[:32],
            ",".join(item.backends),
            f"{item.duration_ms:.0f}" if item.duration_ms is not None else "—",
        )
    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(...),
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
    format: str = FormatOpt,
) -> None:
    """Show diagnosis summary for a run."""

    try:
        data = load_run(run_id, runs_dir)
    except FileNotFoundError as error:
        raise typer.Exit(emit_error(run_not_found(run_id), fmt=format)) from error  # type: ignore[arg-type]
    if format == "json":
        print(json.dumps(data.get("diagnosis") or {}, indent=2))
        return
    if format == "text":
        print(render_text_report(data, redact=True))
        return
    diagnosis = data.get("diagnosis") or {}
    print(
        f"status={diagnosis.get('status')} first_divergence={diagnosis.get('first_divergence')} "
        f"run_id={data.get('run_id')}"
    )


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Existing run id to re-execute"),
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
    format: str = FormatOpt,
    no_gates: bool = typer.Option(False, "--no-gates"),
) -> None:
    """Re-run a stored compare from its reproduction metadata / result.json."""

    try:
        project = find_project_file()
        if project:
            runs_dir = Path(load_project(project).runs_dir)
        data = load_run(run_id, runs_dir)
        repro = data.get("reproduction_command")
        scenario_meta = data.get("scenario") or {}
        backends = []
        for t in data.get("traces") or []:
            b = t.get("backend")
            if b:
                backends.append(b)
        if len(backends) < 2:
            backends = backends * 2 if backends else ["fake", "fake"]
        model = None
        traces = data.get("traces") or []
        if traces:
            model = (traces[0].get("artifact_fingerprint") or {}).get("model_ref")
        model = model or "demo"
        policy = scenario_meta.get("parity_profile") or scenario_meta.get("parity_policy") or "strict"
        observe = scenario_meta.get("observe")
        observe_s = ",".join(observe) if isinstance(observe, list) else None
        baseline = data.get("baseline_backend") or backends[0]
        if format == "text":
            console.print(f"[bold]replay[/bold] of {run_id}")
            if repro:
                console.print(f"[dim]original:[/dim] {repro}")
        # Reconstruct scenario
        sc = Scenario(
            name=scenario_meta.get("name") or "replay",
            messages=[{"role": "user", "content": "Hello"}],
            observe=observe or ["artifact", "template", "tokens", "generation"],
            parity_profile=policy,
            parameters=scenario_meta.get("parameters") or {"temperature": 0, "max_tokens": 32, "seed": 42},
        )
        resolved = resolve_compare(
            model=model,
            backends=",".join(backends[:2]),
            baseline=baseline,
            policy=policy,
            observe=observe_s,
            no_gates=no_gates,
            tokenizer_only=bool(data.get("tokenizer_only")),
        )
        engine = _engine_from_resolved(resolved)
        result = engine.compare(
            model,
            resolved.backends,
            scenario=sc,
            baseline_backend=baseline,
            tokenizer_only=resolved.tokenizer_only,
            redact_prompts=resolved.redact_prompts,
        )
        code = emit_compare_result(
            fmt=format,  # type: ignore[arg-type]
            traces=result.traces,
            diagnosis=result.diagnosis,
            run_id=result.run_id,
            scenario=sc,
            model=model,
            policy=str(policy),
            gate_evaluation=result.gate_evaluation,
            timings=result.timings,
            comparisons=result.comparisons,
        )
        raise typer.Exit(code)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@runs_app.command("diff")
def runs_diff(
    left: str = typer.Argument(...),
    right: str = typer.Argument(...),
    runs_dir: Path = typer.Option(Path(".eleanity/runs")),
    format: str = FormatOpt,
) -> None:
    """Diff two runs (status + per-layer worst outcome)."""

    try:
        result = diff_runs(left, right, runs_dir)
        if format == "quiet":
            print(
                f"status_changed={result.get('status_changed')} "
                f"first_divergence_changed={result.get('first_divergence_changed')}"
            )
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def stabilize(
    backend: str = typer.Option(..., help="Backend to self-test"),
    model: str = typer.Option(DEFAULT_MODEL),
    repetitions: int = typer.Option(5, "--repetitions", "-n"),
    threshold: float = typer.Option(1.0, help="Minimum agreement rate (0-1)"),
    scenario: Path | None = typer.Option(None),
    tokenizer_only: bool = typer.Option(False, "--tokenizer-only"),
    format: str = FormatOpt,
) -> None:
    """Self-consistency protocol: backend vs itself N times before cross-compare."""

    try:
        from eleanity.core.stabilize import stabilize_backend

        resolved = resolve_compare(model=model, backends=backend, tokenizer_only=tokenizer_only)
        selected = load_scenarios(scenario)[0] if scenario else None
        selected = _prepare_scenario(selected, resolved)
        engine = _engine_from_resolved(resolved, parallel=False)
        report = stabilize_backend(
            engine,
            resolved.model,
            backend,
            scenario=selected,
            repetitions=repetitions,
            threshold=threshold,
        )
        if format == "quiet":
            print(f"backend={backend} rate={report.rate} self_consistent={report.self_consistent}")
        else:
            print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(EXIT_OK if report.self_consistent else EXIT_DIVERGENT)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def bisect(
    backend: str = typer.Option("fake"),
    model: str = typer.Option("demo"),
    good: str = typer.Option(..., help="Known-good version or revision"),
    bad: str = typer.Option(..., help="Known-bad version or revision"),
    versions: str | None = typer.Option(None, help="Ordered comma list good…bad (optional full path)"),
    scenario: Path | None = typer.Option(None),
    format: str = FormatOpt,
) -> None:
    """Binary-search first bad runtime/model revision (model@rev encoding for fake/CI)."""

    try:
        from eleanity.core.bisect import bisect_model_revisions, parse_version_list

        revs = parse_version_list(versions) if versions else [good, bad]
        if revs[0] != good:
            revs = [good, *[r for r in revs if r not in {good, bad}], bad]
        engine = CompareEngine(parallel=False)
        selected = load_scenarios(scenario)[0] if scenario else None
        report = bisect_model_revisions(
            engine,
            model,
            backend,
            revs,
            scenario=selected,
            baseline_revision=good,
        )
        if format == "quiet":
            print(f"first_bad={report.first_bad} good={report.good}")
        else:
            print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(EXIT_OK)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="bisect-model")
def bisect_model_cmd(
    good_revision: str = typer.Option(..., "--good-revision"),
    bad_revision: str = typer.Option(..., "--bad-revision"),
    model: str = typer.Option(...),
    backend: str = typer.Option("transformers"),
    revisions: str | None = typer.Option(None, help="Ordered revision list"),
    format: str = FormatOpt,
) -> None:
    """Bisect model revisions (wrapper around bisect)."""

    try:
        from eleanity.core.bisect import bisect_model_revisions, parse_version_list

        revs = parse_version_list(revisions) if revisions else [good_revision, bad_revision]
        if revs[0] != good_revision:
            revs = [good_revision, *[r for r in revs if r not in {good_revision, bad_revision}], bad_revision]
        engine = CompareEngine(parallel=False)
        report = bisect_model_revisions(
            engine,
            model,
            backend,
            revs,
            baseline_revision=good_revision,
        )
        if format == "quiet":
            print(f"first_bad={report.first_bad} good={report.good}")
        else:
            print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(EXIT_OK)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command()
def capture(
    source: Path = typer.Argument(..., help="OpenAI-style JSONL traffic file"),
    output: Path = typer.Option(Path("production-suite"), "--output", "-o"),
    redact: bool = typer.Option(True, "--redact/--no-redact"),
    sample: int | None = typer.Option(None, help="Max records to capture"),
    hash_content: bool = typer.Option(True, "--hash-content/--no-hash-content"),
    format: str = FormatOpt,
) -> None:
    """Turn production OpenAI traces into a redacted scenario suite."""

    try:
        from eleanity.core.capture import capture_openai_jsonl

        manifest = capture_openai_jsonl(
            source,
            output,
            redact=redact,
            hash_content=hash_content,
            sample=sample,
        )
        if format == "quiet":
            print(f"scenarios={manifest['scenarios']} path={manifest['path']}")
        else:
            print(json.dumps(manifest, indent=2))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="policy-spec")
def policy_spec_cmd(
    policy: str = typer.Option("strict", help="strict|quantized|functional|api_conformance"),
    format: str = FormatOpt,
) -> None:
    """Print formal comparator specification for a parity policy."""

    from eleanity.spec.parity import STATUS_DEFINITIONS, policy_comparator_set, status_definition

    payload = {
        "policy": policy_comparator_set(policy).to_dict(),
        "status_definitions": STATUS_DEFINITIONS,
        "example_status": status_definition("PASS_WITH_TOLERANCE"),
    }
    if format == "quiet":
        comps = payload["policy"]["comparators"]
        print(" ".join(f"{k}={v['mode']}" for k, v in comps.items()))
    else:
        print(json.dumps(payload, indent=2))


@app.command(name="compat-matrix")
def compat_matrix_cmd(
    model: str = typer.Option("demo"),
    backends: str = typer.Option("fake,transformers"),
    output: Path = typer.Option(Path(".eleanity/matrix/compat.json")),
    format: str = FormatOpt,
) -> None:
    """Build a public-style model×runtime compatibility matrix."""

    try:
        from eleanity.core.compat_matrix import certify_row, render_matrix_markdown, write_matrix

        rows = []
        for backend in [b.strip() for b in backends.split(",") if b.strip()]:
            if backend not in available_adapters():
                continue
            if backend == "transformers":
                try:
                    ensure_local_dep("transformers")
                except Exception:
                    rows.append(
                        {
                            "model": model,
                            "runtime": backend,
                            "status": "unavailable",
                            "features": {},
                            "notes": ["optional dep missing"],
                        }
                    )
                    continue
            rows.append(certify_row(model, backend))
        path = write_matrix(output, rows)
        if format == "quiet":
            print(f"rows={len(rows)} path={path}")
        elif format == "text":
            console.print(render_matrix_markdown(rows))
            console.print(f"Wrote {path}")
        else:
            print(json.dumps({"path": str(path), "rows": rows}, indent=2))
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]


@app.command(name="trace-validate")
def trace_validate_cmd(
    path: Path = typer.Argument(..., help="trace.v1.json or result.json"),
    format: str = FormatOpt,
) -> None:
    """Validate an Eleanity Trace Spec document (v1 or migrate from result.json)."""

    try:
        from eleanity.spec.trace_v1 import migrate_result_to_v1, validate_trace_document

        data = json.loads(path.read_text(encoding="utf-8"))
        if "subjects" not in data and "traces" in data:
            data = migrate_result_to_v1(data)
        errors = validate_trace_document(data)
        ok = not errors
        if format == "quiet":
            print(f"ok={ok} errors={len(errors)}")
        else:
            print(json.dumps({"ok": ok, "errors": errors, "schema_version": data.get("schema_version")}, indent=2))
        raise typer.Exit(EXIT_OK if ok else EXIT_CONFIG)
    except typer.Exit:
        raise
    except Exception as error:
        raise typer.Exit(emit_error(error, fmt=format))  # type: ignore[arg-type]
