from typer.testing import CliRunner
from eleanity.cli.app import app


def test_doctor_runs():
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.stdout


def test_compare_with_fake_backends_writes_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["compare", "--model", "demo", "--backends", "fake,fake"])
    assert result.exit_code == 0
    assert "Run:" in result.stdout
