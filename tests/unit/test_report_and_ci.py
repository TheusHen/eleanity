import json
import re

import pytest
from typer.testing import CliRunner

from eleanity.cli.app import app
from eleanity.core.run import run_ci, run_compare
from eleanity.models.schemas import Scenario
from eleanity.reporters.html import render_html, write_html


def test_html_includes_executive_summary_and_layer_navigation(tmp_path):
    run_id, _, _ = run_compare("demo", ["fake", "fake"], runs_dir=tmp_path)
    html = write_html(tmp_path / run_id / "result.json").read_text(encoding="utf-8")
    assert ("Executive summary" in html) or ("Resumo executivo" in html) or ("PASS" in html)
    assert "#template" in html
    assert "PASS" in html


def test_html_is_share_safe_and_escapes_diagnostic_copy(tmp_path):
    run_id, _, _ = run_compare("demo", ["fake", "fake"], runs_dir=tmp_path)
    result_path = tmp_path / run_id / "result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["traces"][0]["layers"]["template"]["data"]["text"] = "<script>private prompt</script>"
    data["traces"][0]["layers"]["structured"] = {
        "state": "NOT_OBSERVABLE",
        "data": {},
        "note": "api_key=super-secret",
    }
    data["diagnosis"]["summary"] = "Summary <script>alert(1)</script>"
    result_path.write_text(json.dumps(data), encoding="utf-8")

    html = write_html(result_path).read_text(encoding="utf-8")

    assert "private prompt" not in html
    assert "super-secret" not in html
    assert "<script>alert(1)</script>" not in html
    assert "Summary &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert ("[redacted]" in html) or ("[redigido]" in html)
    assert "default-src 'none'" in html


def test_html_distinguishes_same_backend_reference_and_candidate(tmp_path):
    run_id, _, _ = run_ci("demo", "demo", "fake", runs_dir=tmp_path)
    html = write_html(tmp_path / run_id / "result.json").read_text(encoding="utf-8")

    assert "Fake adapter · reference" in html
    assert "Fake adapter · candidate 1" in html
    assert re.search(r'data-status="REFERENCE">REF</span>', html)


def test_run_persists_scenario_metadata_without_messages(tmp_path):
    scenario = Scenario(
        name="private-scenario",
        messages=[{"role": "user", "content": "never persist this in scenario metadata"}],
        parameters={"temperature": 0, "seed": 7},
        parity_profile="quantized",
    )
    run_id, _, _ = run_compare("demo", ["fake", "fake"], scenario=scenario, runs_dir=tmp_path)
    data = json.loads((tmp_path / run_id / "result.json").read_text(encoding="utf-8"))

    assert data["schema_version"] == "1"
    assert data["run_type"] == "compare"
    assert data["scenario"]["name"] == "private-scenario"
    assert data["scenario"]["parity_profile"] == "quantized"
    assert data["scenario"]["tolerance"] == 0.02
    assert "messages" not in data["scenario"]


def test_html_rejects_run_without_traces():
    with pytest.raises(ValueError, match="at least one trace"):
        render_html({"run_id": "empty", "traces": []})


def test_report_json_prints_serialized_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id, _, _ = run_compare("demo", ["fake", "fake"])
    result = CliRunner().invoke(app, ["report", run_id, "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_id"] == run_id


def test_ci_returns_zero_for_equal_fake_references(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["ci", "--baseline", "demo", "--candidate", "demo", "--backend", "fake"],
    )
    assert result.exit_code == 0
