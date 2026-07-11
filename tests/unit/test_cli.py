import os
import subprocess
import sys

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


def test_help_survives_legacy_cp1252_encoding():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252:strict"
    completed = subprocess.run(
        [sys.executable, "-c", "from eleanity.cli.app import app; app()", "--help"],
        env=env,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("cp1252", errors="replace")
    assert b"Usage:" in completed.stdout
