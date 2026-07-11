import json
from pathlib import Path

from typer.testing import CliRunner

from eleanity.cli.app import app
from eleanity.cli.exitcodes import EXIT_OK
from eleanity.cli.resolve import resolve_compare
from eleanity.core.engine import CompareEngine
from eleanity.core.golden import save_golden
from eleanity.models.schemas import ObservationTrace

runner = CliRunner()


def test_compare_json_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "--model", "demo", "--backends", "fake,fake", "--format", "json", "--no-parallel"],
    )
    assert result.exit_code == EXIT_OK
    payload = json.loads(result.stdout)
    assert "summary" in payload
    assert payload["summary"]["status"] in {"PASS", "PASS_WITH_TOLERANCE", "NOT_OBSERVABLE"}
    assert payload["summary"]["run_id"]


def test_compare_quiet_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "--model", "demo", "--backends", "fake,fake", "--format", "quiet", "--no-parallel"],
    )
    assert result.exit_code == EXIT_OK
    assert "status=" in result.stdout
    assert "run_id=" in result.stdout


def test_report_text_contains_layers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    compare = runner.invoke(
        app,
        ["compare", "--model", "demo", "--backends", "fake,fake", "--format", "quiet", "--no-parallel"],
    )
    run_id = [part.split("=", 1)[1] for part in compare.stdout.split() if part.startswith("run_id=")][0]
    report = runner.invoke(app, ["report", run_id, "--format", "text"])
    assert report.exit_code == 0
    assert "Eleanity" in report.stdout
    assert "Diagnosis" in report.stdout or "status" in report.stdout.lower()


def test_resolve_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("ELEANITY_MODEL", "from-env")
    monkeypatch.setenv("ELEANITY_BACKENDS", "fake,fake")
    resolved = resolve_compare(model="from-cli", backends="fake,fake")
    assert resolved.model == "from-cli"
    assert resolved.backends == ["fake", "fake"]


def test_doctor_json():
    result = runner.invoke(app, ["doctor", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "adapters" in payload


def test_migrate_fake(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["migrate", "--from", "fake", "--to", "fake", "--model", "demo", "--format", "quiet"],
    )
    assert result.exit_code == EXIT_OK
    assert "status=" in result.stdout


def test_batch_fail_fast_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ensure suite path exists relative to repo — chdir breaks fixtures path
    # run from repo root instead
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    result = runner.invoke(
        app,
        [
            "batch",
            "--models",
            "demo",
            "--backends",
            "fake,fake",
            "--suite",
            "generic-chat",
            "--tokenizer-only",
            "--format",
            "quiet",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "jobs=" in result.stdout


def test_snapshot_and_check_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    golden_path = save_golden(
        ObservationTrace.model_validate(
            json.loads((tmp_path / "runs" / result.run_id / "result.json").read_text(encoding="utf-8"))["traces"][0]
        ),
        tmp_path / "golden",
    )
    check = runner.invoke(
        app,
        [
            "check-golden",
            result.run_id,
            "--golden",
            str(golden_path),
            "--format",
            "quiet",
        ],
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PWD": str(tmp_path)},
    )
    # load_run looks at .eleanity/runs by default — point via chdir + copy
    # Re-run using explicit path by copying run into .eleanity/runs
    runs = tmp_path / ".eleanity" / "runs" / result.run_id
    runs.mkdir(parents=True)
    (runs / "result.json").write_text(
        (tmp_path / "runs" / result.run_id / "result.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    check = runner.invoke(
        app,
        ["check-golden", result.run_id, "--golden", str(golden_path), "--format", "quiet"],
    )
    assert check.exit_code == EXIT_OK
    assert "passed=True" in check.stdout


def test_runs_ls_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = CompareEngine(runs_dir=tmp_path / ".eleanity" / "runs", parallel=False)
    engine.compare("demo", ["fake", "fake"])
    result = runner.invoke(app, ["runs", "ls", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload


def test_doctor_check_backends_fake():
    result = runner.invoke(
        app,
        ["doctor", "--check-backends", "--backends", "fake", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["backends"]
    assert payload["backends"][0]["name"] == "fake"
    assert payload["backends"][0]["ok"] is True


def test_unknown_backend_error_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "--model", "demo", "--backends", "nope", "--format", "json"],
    )
    assert result.exit_code == 2
    err = result.stderr or result.stdout
    assert "ELEANITY_E001" in err or "unknown backend" in err.lower()


def test_compare_no_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "compare",
            "--model",
            "demo",
            "--backends",
            "fake,fake",
            "--no-gates",
            "--format",
            "json",
            "--no-parallel",
        ],
    )
    assert result.exit_code == EXIT_OK
    payload = json.loads(result.stdout)
    # Empty gate list: evaluation may still be present but with no named rules
    gates = payload.get("gates")
    if gates is not None:
        assert gates.get("results") == [] or gates.get("passed") is True


def test_ci_quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "ci",
            "--baseline",
            "demo",
            "--candidate",
            "demo",
            "--backend",
            "fake",
            "--format",
            "quiet",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "status=" in result.stdout
    assert "run_id=" in result.stdout
