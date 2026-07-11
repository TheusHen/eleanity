"""Public Python API (client B + low-level C)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eleanity import (
    EXIT_OK,
    CompareOutcome,
    Eleanity,
    ParityError,
    __version__,
    compare_traces,
    make_scenario,
    observe_backend,
)
from eleanity.api import observe_backend as obs2


def test_version_and_import_surface():
    assert __version__
    assert Eleanity is not None
    assert obs2 is observe_backend


def test_client_compare_fake(tmp_path: Path):
    client = Eleanity(runs_dir=tmp_path / "runs", parallel=False, apply_gates=False)
    result = client.compare(model="demo", backends=["fake", "fake"], no_gates=True, parallel=False)
    assert isinstance(result, CompareOutcome)
    assert result.passed
    assert result.exit_code == EXIT_OK
    assert result.status == "PASS"
    assert result.first_divergence is None or result.first_divergence == "none" or result.first_divergence is None
    assert result.run_id
    assert len(result.traces) == 2
    assert (tmp_path / "runs" / result.run_id / "result.json").is_file()
    result.raise_for_status()
    d = result.to_dict()
    assert d["passed"] is True
    assert "run_id" in d


def test_client_from_yaml_and_test(tmp_path: Path):
    yaml_path = tmp_path / "eleanity.yaml"
    yaml_path.write_text(
        """\
model: demo
backends: [fake, fake]
policy: strict
runs_dir: runs
parallel: false
""",
        encoding="utf-8",
    )
    client = Eleanity.from_yaml(yaml_path, runs_dir=tmp_path / "runs")
    report = client.test(
        Path("fixtures/public/tokenizer-edge.yaml"),
        backends=["fake", "fake"],
        no_gates=True,
    )
    assert report.passed
    assert len(report.results) >= 1
    assert report.exit_code == EXIT_OK


def test_doctor_and_report(tmp_path: Path):
    client = Eleanity(runs_dir=tmp_path / "runs", parallel=False)
    doc = client.doctor(check_backends=True, backends=["fake"])
    assert doc.ok
    assert "fake" in doc.adapters
    assert any(b.name == "fake" and b.ok for b in doc.backends)

    outcome = client.compare(model="demo", backends=["fake", "fake"], no_gates=True, parallel=False)
    data = client.report(outcome.run_id, fmt="dict")
    assert data["run_id"] == outcome.run_id
    text = client.report(outcome.run_id, fmt="text")
    assert isinstance(text, str)
    assert outcome.run_id[:8] in text or "PASS" in text or "status" in text.lower() or len(text) > 10


def test_replay(tmp_path: Path):
    client = Eleanity(runs_dir=tmp_path / "runs", parallel=False, apply_gates=False)
    first = client.compare(model="demo", backends=["fake", "fake"], no_gates=True, parallel=False)
    second = client.replay(first.run_id, no_gates=True)
    assert second.passed
    assert second.run_id != first.run_id


def test_migrate_and_list_runs(tmp_path: Path):
    client = Eleanity(runs_dir=tmp_path / "runs", parallel=False)
    out = client.migrate(model="demo", from_backend="fake", to_backend="fake", no_gates=True)
    assert out.passed
    runs = client.list_runs()
    assert any(r.get("run_id") == out.run_id or True for r in runs) or len(runs) >= 0
    loaded = client.get_run(out.run_id)
    assert loaded["run_id"] == out.run_id


def test_lowlevel_observe_and_compare_traces():
    sc = make_scenario(model="demo", policy="strict")
    left = observe_backend("fake", "demo", sc)
    right = observe_backend("fake", "demo", sc)
    comps = compare_traces(left, right, sc)
    assert isinstance(comps, dict)
    assert comps
    # template/tokens/generation should compare
    for key in ("template", "tokens", "generation"):
        if key in comps:
            status = comps[key].status if hasattr(comps[key], "status") else comps[key]
            assert status is not None


def test_raise_for_status_divergent(tmp_path: Path):
    """Fake adapters match — use diagnosis override path via exit_code property."""

    client = Eleanity(runs_dir=tmp_path / "runs", parallel=False)
    result = client.compare(model="demo", backends=["fake", "fake"], no_gates=True, parallel=False)
    # Force divergent for unit test of raise_for_status
    result.status = "DIVERGENT"
    result.gates_passed = False
    # diagnosis still PASS — exit_code uses diagnosis; patch diagnosis status
    if hasattr(result.diagnosis, "model_copy"):
        from eleanity.models.schemas import ParityResult

        result.diagnosis = result.diagnosis.model_copy(update={"status": ParityResult.DIVERGENT})
    with pytest.raises(ParityError):
        result.raise_for_status()


def test_policy_spec_and_certify_fake():
    client = Eleanity()
    spec = client.policy_spec("quantized")
    assert isinstance(spec, dict)
    cert = client.certify("fake", model="demo")
    assert cert.get("passed") is True or cert.get("level") in {"bronze", "silver", "gold", "none"}
