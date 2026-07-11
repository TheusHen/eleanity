from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from eleanity.adapters import adapter_for
from eleanity.config.project import BackendProfile, EleanityProject, GateRule
from eleanity.core.matrix import build_pairwise_matrix, consensus_summary
from eleanity.core.observe import observe
from eleanity.core.store import write_github_annotations, write_junit, write_result_json
from eleanity.diagnosers import diagnose
from eleanity.fingerprints import collect_environment_fingerprint
from eleanity.gates.engine import GateEvaluation, evaluate_gates
from eleanity.models.schemas import ModelSpec, ObservationTrace, Scenario
from eleanity.policies.engine import PolicyEngine
from eleanity.reporters.sarif import write_sarif
from eleanity.utils.logging import get_logger, log_event

logger = get_logger("eleanity.engine")

RESULT_SCHEMA_VERSION = "1"


@dataclass
class CompareResult:
    run_id: str
    traces: list[ObservationTrace]
    diagnosis: Any
    comparisons: dict[str, dict]
    consensus: dict[str, Any]
    path: Path
    gate_evaluation: GateEvaluation | None = None
    timings: dict[str, float] = field(default_factory=dict)


class CompareEngine:
    """Production orchestration for multi-backend parity runs."""

    def __init__(
        self,
        *,
        runs_dir: Path = Path(".eleanity/runs"),
        max_workers: int = 4,
        parallel: bool = True,
        tokenizer_only: bool = False,
        project: EleanityProject | None = None,
        backend_profiles: dict[str, BackendProfile] | None = None,
        gates: list[GateRule] | None = None,
    ):
        self.project = project
        # Explicit runs_dir wins (tests / CLI); otherwise fall back to project config.
        if runs_dir != Path(".eleanity/runs"):
            self.runs_dir = Path(runs_dir)
        elif project is not None:
            self.runs_dir = Path(project.runs_dir)
        else:
            self.runs_dir = Path(runs_dir)
        self.max_workers = max_workers if project is None else project.workers
        if max_workers != 4:
            self.max_workers = max_workers
        self.parallel = parallel if project is None else project.parallel
        # Allow callers to force sequential/parallel regardless of project defaults.
        if parallel is False:
            self.parallel = False
        self.tokenizer_only = tokenizer_only or (project.tokenizer_only if project else False)
        self.backend_profiles = backend_profiles or (
            project.backend_profiles if project else {}
        )
        self.gates = gates if gates is not None else (project.gates if project else [])

    def compare(
        self,
        model: str,
        backends: list[str],
        scenario: Scenario | None = None,
        baseline_backend: str | None = None,
        *,
        redact_prompts: bool = False,
        junit_path: Path | None = None,
        annotations_path: Path | None = None,
        sarif_path: Path | None = None,
        tokenizer_only: bool | None = None,
    ) -> CompareResult:
        scenario = scenario or Scenario(
            name="compare",
            messages=[{"role": "user", "content": "Hello"}],
            observe=["template", "tokens", "logits", "generation"],
        )
        tok_only = self.tokenizer_only if tokenizer_only is None else tokenizer_only
        if tok_only:
            # Cheap CI path: drop weight-heavy layers from observe if present
            light = [
                layer
                for layer in scenario.observe
                if layer not in {"logits", "generation", "structured", "streaming"}
            ]
            if "template" not in light:
                light.append("template")
            if "tokens" not in light:
                light.append("tokens")
            scenario = scenario.model_copy(update={"observe": light})
            if scenario.model:
                scenario = scenario.model_copy(
                    update={
                        "model": scenario.model.model_copy(update={"tokenizer_only": True})
                    }
                )
            else:
                scenario = scenario.model_copy(
                    update={"model": ModelSpec(id=model, tokenizer_only=True)}
                )

        if redact_prompts or (self.project and self.project.redact_prompts):
            scenario = scenario.model_copy(update={"redact_prompts": True})
        if scenario.model and scenario.model.id:
            model = scenario.model.id

        baseline_backend = baseline_backend or backends[0]
        if baseline_backend not in backends:
            raise ValueError(f"baseline backend '{baseline_backend}' is not in backends")

        remaining = list(backends)
        remaining.pop(remaining.index(baseline_backend))
        ordered = [baseline_backend, *remaining]

        log_event(
            logger,
            "compare_start",
            model=model,
            backends=",".join(ordered),
            parallel=self.parallel,
            policy=scenario.parity_profile.value,
            tokenizer_only=tok_only,
        )

        traces = self._collect_traces(ordered, model, scenario, baseline_backend, tok_only)
        diagnosis = diagnose(traces)
        comparisons = build_pairwise_matrix(traces, scenario, baseline_index=0)
        consensus = consensus_summary(traces, scenario)

        # Coverage + enrichment (limited coverage, confidence, verified sets)
        from eleanity.core.coverage import build_reproduction_command, format_timings
        from eleanity.core.enrich import enrich_diagnosis
        from eleanity.spec.capsule import build_execution_capsule
        from eleanity.spec.impact import assess_impact
        from eleanity.spec.parity import formal_status_from_parity, policy_comparator_set
        from eleanity.spec.trace_v1 import build_trace_document, write_trace_v1

        if len(traces) >= 2:
            diagnosis = enrich_diagnosis(
                diagnosis,
                left=traces[0],
                right=traces[1],
                scenario=scenario,
                comparisons=comparisons,
                model=model,
                backends=ordered,
            )

        gate_eval = evaluate_gates(
            self.gates,
            comparisons,
            diagnosis_status=getattr(diagnosis, "status", None),
            coverage=getattr(diagnosis, "coverage", None),
            policy=scenario.parity_profile.value,
        )

        formal = formal_status_from_parity(diagnosis.status)
        impact = assess_impact(
            parity_status=diagnosis.status,
            first_divergence=diagnosis.first_divergence,
            left=traces[0] if traces else None,
            right=traces[1] if len(traces) > 1 else None,
            comparisons=(
                next(iter(comparisons.values()), {}) if comparisons else {}
            ),
        )
        diagnosis = diagnosis.model_copy(
            update={
                "formal_status": (
                    diagnosis.status.value
                    if hasattr(diagnosis.status, "value")
                    else formal.value
                ),
                "impact": impact.to_dict(),
                "policy_comparators": policy_comparator_set(scenario.parity_profile).to_dict(),
            }
        )

        env_fp = collect_environment_fingerprint()
        capsules = {}
        for trace in traces:
            capsules[trace.backend] = build_execution_capsule(
                backend=trace.backend,
                model=model,
                scenario=scenario,
                fingerprint=trace.artifact_fingerprint,
                environment=trace.environment or env_fp,
                tokenizer_only=tok_only,
            )

        timings: dict[str, float] = {}
        seen_timing: dict[str, int] = {}
        for t in traces:
            count = seen_timing.get(t.backend, 0) + 1
            seen_timing[t.backend] = count
            key = t.backend if count == 1 else f"{t.backend}#{count}"
            timings[key] = float(t.duration_ms or 0.0)
        total_ms = sum(timings.values())
        timing_report = format_timings(timings)

        run_id = str(uuid4())
        # Attach run_id to practical commands
        if len(traces) >= 2:
            diagnosis = enrich_diagnosis(
                diagnosis,
                left=traces[0],
                right=traces[1],
                scenario=scenario,
                comparisons=comparisons,
                model=model,
                backends=ordered,
                run_id=run_id,
            )
            # re-apply impact after second enrich
            diagnosis = diagnosis.model_copy(
                update={
                    "impact": impact.to_dict(),
                    "formal_status": diagnosis.status.value
                    if hasattr(diagnosis.status, "value")
                    else str(diagnosis.status),
                }
            )

        target = self.runs_dir / run_id
        backend_urls = {
            name: (self.backend_profiles[name].base_url or "")
            for name in ordered
            if name in self.backend_profiles and self.backend_profiles[name].base_url
        }
        reproduction = build_reproduction_command(
            model=model,
            backends=ordered,
            baseline=baseline_backend,
            policy=scenario.parity_profile.value,
            observe=list(scenario.observe),
            tokenizer_only=tok_only,
            backend_urls={k: v for k, v in backend_urls.items() if v and not str(v).startswith("${")},
            no_gates=not bool(self.gates),
            scenario_name=scenario.name,
        )
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_type": "compare",
            "run_id": run_id,
            "baseline_backend": baseline_backend,
            "scenario": scenario.model_dump(mode="json", exclude={"messages"}),
            "environment": env_fp.model_dump(mode="json"),
            "traces": [trace.model_dump(mode="json") for trace in traces],
            "execution_capsules": {
                name: cap.model_dump(mode="json") for name, cap in capsules.items()
            },
            "comparisons": comparisons,
            "consensus": consensus,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "impact": impact.to_dict(),
            "coverage": diagnosis.coverage,
            "verified_layers": diagnosis.verified_layers,
            "not_verified_layers": diagnosis.not_verified_layers,
            "tolerance_reasons": diagnosis.tolerance_reasons,
            "artifact_divergent_fields": diagnosis.artifact_divergent_fields,
            "confidence": diagnosis.confidence,
            "timings": timing_report,
            "parity": {
                "status": diagnosis.status.value
                if hasattr(diagnosis.status, "value")
                else str(diagnosis.status),
                "legacy_status": diagnosis.status.value
                if hasattr(diagnosis.status, "value")
                else str(diagnosis.status),
            },
            "gates": {
                "passed": gate_eval.passed,
                "summary": gate_eval.summary,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "layer": r.layer,
                        "status": r.status.value if r.status else None,
                        "message": r.message,
                        "backend": r.backend,
                    }
                    for r in gate_eval.results
                ],
            },
            "timings_ms": timings,
            "total_duration_ms": round(total_ms, 3),
            "tokenizer_only": tok_only,
            "reproduction_command": reproduction,
            "practical_commands": diagnosis.practical_commands,
            "capabilities": self._capability_map(ordered, model, scenario, tok_only),
        }

        path = write_result_json(
            target, payload, redact_prompts=scenario.redact_prompts
        )
        # Trace Spec v1 product document alongside result.json
        try:
            trace_doc = build_trace_document(
                run_id=run_id,
                result_payload=payload,
                capsules=capsules,
                impact=impact,
            )
            write_trace_v1(target / "trace.v1.json", trace_doc)
            payload["trace_v1_path"] = str(target / "trace.v1.json")
        except Exception as error:  # never fail compare on export
            log_event(logger, "trace_v1_export_failed", error=str(error))

        write_junit(junit_path or target / "junit.xml", run_id, diagnosis)
        write_github_annotations(annotations_path or target / "github-annotations.txt", diagnosis)
        write_sarif(path, sarif_path or target / "results.sarif")

        # index line for quick listing
        (target / "summary.txt").write_text(
            f"{run_id}\t{getattr(diagnosis, 'status', None)}\t"
            f"{scenario.name}\t{model}\t{','.join(ordered)}\n",
            encoding="utf-8",
        )

        log_event(
            logger,
            "compare_done",
            run_id=run_id,
            status=getattr(diagnosis, "status", None),
            formal_status=formal.value,
            impact=impact.impact.value,
            first_divergence=getattr(diagnosis, "first_divergence", None),
            gates_passed=gate_eval.passed,
            total_ms=round(total_ms, 2),
        )
        return CompareResult(
            run_id=run_id,
            traces=traces,
            diagnosis=diagnosis,
            comparisons=comparisons,
            consensus=consensus,
            path=path.parent,
            gate_evaluation=gate_eval,
            timings=timings,
        )

    def batch(
        self,
        model: str,
        backends: list[str],
        scenarios: list[Scenario],
        *,
        baseline_backend: str | None = None,
        redact_prompts: bool = False,
        tokenizer_only: bool | None = None,
    ) -> list[CompareResult]:
        results = []
        for scenario in scenarios:
            results.append(
                self.compare(
                    model,
                    backends,
                    scenario=scenario,
                    baseline_backend=baseline_backend,
                    redact_prompts=redact_prompts,
                    tokenizer_only=tokenizer_only,
                )
            )
        return results

    def ci(
        self,
        baseline_model: str,
        candidate_model: str,
        backend: str,
        scenario: Scenario | None = None,
        *,
        junit_path: Path | None = None,
        tokenizer_only: bool = False,
    ) -> tuple[str, dict, Any, GateEvaluation]:
        scenario = scenario or Scenario(
            name="ci",
            messages=[{"role": "user", "content": "Hello"}],
            observe=["template", "tokens", "logits", "generation"],
        )
        if tokenizer_only:
            scenario = scenario.model_copy(
                update={
                    "observe": [layer for layer in scenario.observe if layer not in {"logits", "generation"}],
                    "model": ModelSpec(id=baseline_model, tokenizer_only=True),
                }
            )
        baseline = observe(
            adapter_for(backend, baseline_model, scenario=scenario, tokenizer_only=tokenizer_only),
            scenario,
            baseline_model,
            backend,
        )
        candidate = observe(
            adapter_for(backend, candidate_model, scenario=scenario, tokenizer_only=tokenizer_only),
            scenario,
            candidate_model,
            backend,
        )
        engine = PolicyEngine(scenario)
        comparisons = engine.compare_layers(baseline, candidate)
        diagnosis = diagnose([baseline, candidate])
        cmp_map = {
            backend: {
                layer: comparison.model_dump(mode="json")
                for layer, comparison in comparisons.items()
            }
        }
        gate_eval = evaluate_gates(
            self.gates, cmp_map, diagnosis_status=getattr(diagnosis, "status", None)
        )
        run_id = str(uuid4())
        target = self.runs_dir / run_id
        path = write_result_json(
            target,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_type": "ci",
                "run_id": run_id,
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "baseline_backend": backend,
                "scenario": scenario.model_dump(mode="json", exclude={"messages"}),
                "environment": collect_environment_fingerprint().model_dump(mode="json"),
                "traces": [baseline.model_dump(mode="json"), candidate.model_dump(mode="json")],
                "comparisons": cmp_map,
                "consensus": consensus_summary([baseline, candidate], scenario),
                "diagnosis": diagnosis.model_dump(mode="json"),
                "gates": {
                    "passed": gate_eval.passed,
                    "summary": gate_eval.summary,
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "message": r.message,
                            "status": r.status.value if r.status else None,
                        }
                        for r in gate_eval.results
                    ],
                },
                "reproduction_command": (
                    f"eleanity ci --baseline {baseline_model} --candidate {candidate_model} "
                    f"--backend {backend}"
                ),
            },
        )
        write_junit(junit_path or target / "junit.xml", run_id, diagnosis)
        write_github_annotations(target / "github-annotations.txt", diagnosis)
        write_sarif(path, target / "results.sarif")
        return run_id, comparisons, diagnosis, gate_eval

    def _profile_kwargs(self, name: str) -> dict[str, Any]:
        profile = self.backend_profiles.get(name)
        if not profile:
            return {}
        kwargs: dict[str, Any] = {}
        if profile.base_url:
            url = profile.base_url
            if url.startswith("${") and url.endswith("}"):
                env_name = url[2:-1]
                url = os.getenv(env_name, "")
            if url:
                kwargs["base_url"] = url
        if profile.api_key_env:
            key = os.getenv(profile.api_key_env)
            if key:
                kwargs["api_key"] = key
        return kwargs

    def _make_adapter(self, name: str, model: str, scenario: Scenario, tokenizer_only: bool):
        profile = self.backend_profiles.get(name)
        adapter_name = profile.adapter if profile else name
        model_id = (profile.model if profile and profile.model else model)
        kwargs = self._profile_kwargs(name)
        return adapter_for(
            adapter_name,
            model_id,
            scenario=scenario,
            tokenizer_only=tokenizer_only,
            **kwargs,
        )

    def _collect_traces(
        self,
        ordered: list[str],
        model: str,
        scenario: Scenario,
        baseline_backend: str,
        tokenizer_only: bool,
    ) -> list[ObservationTrace]:
        def _one(name: str) -> ObservationTrace:
            adapter = self._make_adapter(name, model, scenario, tokenizer_only)
            # Prefer per-backend model from profile (e.g. HF id vs LM Studio id)
            profile = self.backend_profiles.get(name)
            model_id = profile.model if profile and profile.model else model
            return observe(adapter, scenario, model_id, baseline_backend)

        if not self.parallel or len(ordered) == 1:
            return [_one(name) for name in ordered]

        traces_by_name: dict[int, ObservationTrace] = {}

        def _run(index: int, name: str) -> tuple[int, ObservationTrace]:
            return index, _one(name)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ordered))) as pool:
            futures = [pool.submit(_run, index, name) for index, name in enumerate(ordered)]
            for future in as_completed(futures):
                index, trace = future.result()
                traces_by_name[index] = trace
        return [traces_by_name[i] for i in range(len(ordered))]

    def _capability_map(
        self,
        backends: list[str],
        model: str,
        scenario: Scenario,
        tokenizer_only: bool,
    ) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for name in backends:
            adapter = self._make_adapter(name, model, scenario, tokenizer_only)
            caps = getattr(adapter, "capabilities", None)
            if caps is not None:
                result[name] = caps.model_dump(mode="json")
        return result
