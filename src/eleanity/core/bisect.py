"""Automatic bisect for runtime versions or model revisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eleanity.core.engine import CompareEngine
from eleanity.models.schemas import ParityResult, Scenario


@dataclass
class BisectStep:
    mid: str
    status: str
    run_id: str | None = None
    first_divergence: str | None = None


@dataclass
class BisectReport:
    kind: str
    good: str
    bad: str
    steps: list[BisectStep] = field(default_factory=list)
    first_bad: str | None = None
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "good": self.good,
            "bad": self.bad,
            "first_bad": self.first_bad,
            "conclusion": self.conclusion,
            "steps": [
                {
                    "mid": s.mid,
                    "status": s.status,
                    "run_id": s.run_id,
                    "first_divergence": s.first_divergence,
                }
                for s in self.steps
            ],
        }


def _is_good(status: str) -> bool:
    return status in {
        ParityResult.PASS.value,
        ParityResult.PASS_WITH_TOLERANCE.value,
    }


def bisect_versions(
    versions: list[str],
    *,
    is_good_fn: Callable[[str], tuple[bool, str, str | None, str | None]],
) -> BisectReport:
    """Generic binary search over an ordered list (good … bad).

    is_good_fn(version) -> (is_good, status, run_id, first_divergence)
    versions must be ordered from oldest/good side toward newest/bad side.
    """

    if len(versions) < 2:
        return BisectReport(
            kind="versions",
            good=versions[0] if versions else "",
            bad=versions[-1] if versions else "",
            conclusion="Need at least two versions to bisect.",
        )
    lo = 0
    hi = len(versions) - 1
    versions[lo]
    versions[hi]
    steps: list[BisectStep] = []
    # Verify endpoints lightly via caller
    while hi - lo > 1:
        mid = (lo + hi) // 2
        version = versions[mid]
        ok, status, run_id, first = is_good_fn(version)
        steps.append(BisectStep(mid=version, status=status, run_id=run_id, first_divergence=first))
        if ok:
            lo = mid
        else:
            hi = mid
    first_bad = versions[hi]
    return BisectReport(
        kind="versions",
        good=versions[lo],
        bad=versions[hi],
        steps=steps,
        first_bad=first_bad,
        conclusion=f"First bad revision/version: {first_bad} (last good: {versions[lo]}).",
    )


def bisect_model_revisions(
    engine: CompareEngine,
    model: str,
    backend: str,
    revisions: list[str],
    *,
    scenario: Scenario | None = None,
    baseline_revision: str | None = None,
) -> BisectReport:
    """Bisect model revisions against a fixed good baseline revision."""

    if not revisions:
        return BisectReport(kind="model", good="", bad="", conclusion="No revisions provided.")
    baseline = baseline_revision or revisions[0]

    def probe(revision: str) -> tuple[bool, str, str | None, str | None]:
        # Compare baseline model@rev0 vs model@revision on same backend via ci
        run_id, _, diagnosis, _ = engine.ci(
            f"{model}@{baseline}" if "@" not in model else model,
            f"{model}@{revision}" if "@" not in model else revision,
            backend,
            scenario=scenario,
        )
        # Prefer using model ids as revision-suffixed when adapters support it
        status = diagnosis.status.value if hasattr(diagnosis.status, "value") else str(diagnosis.status)
        return _is_good(status), status, run_id, diagnosis.first_divergence

    # For real HF revisions, adapters use model id + revision via scenario.model
    def probe_scenario(revision: str) -> tuple[bool, str, str | None, str | None]:
        from eleanity.models.schemas import ModelSpec

        base_sc = scenario or Scenario(
            name="bisect",
            messages=[{"role": "user", "content": "Hello"}],
            observe=["artifact", "template", "tokens"],
            parameters={"temperature": 0, "max_tokens": 8, "seed": 42},
        )
        left = base_sc.model_copy(update={"model": ModelSpec(id=model, revision=baseline, tokenizer_only=True)})
        # Use compare of same model string but different fingerprints via two runs:
        # engine.ci with same model id — revision is scenario-level; for fake, always pass.
        engine.compare(model, [backend, backend], scenario=left, baseline_backend=backend)
        # Secondary: force candidate scenario revision into second observe by sequential ci
        run_id, _, diagnosis, _ = engine.ci(
            model,
            model,
            backend,
            scenario=base_sc.model_copy(update={"model": ModelSpec(id=model, revision=revision, tokenizer_only=True)}),
        )
        status = diagnosis.status.value if hasattr(diagnosis.status, "value") else str(diagnosis.status)
        # Mark mid as good if self-ci passes (same model); real divergence needs two revisions
        # as distinct model refs: model + revision encoded
        good, st, rid, first = True, status, run_id, diagnosis.first_divergence
        # Re-run as promote-style when revisions differ from baseline using encoded ids
        if revision != baseline:
            run_id2, _, diag2, _ = engine.ci(
                f"{model}@{baseline}",
                f"{model}@{revision}",
                backend,
                scenario=base_sc,
            )
            st = diag2.status.value if hasattr(diag2.status, "value") else str(diag2.status)
            good = _is_good(st)
            return good, st, run_id2, diag2.first_divergence
        return good, st, rid, first

    return bisect_versions(revisions, is_good_fn=probe_scenario)


def parse_version_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]
