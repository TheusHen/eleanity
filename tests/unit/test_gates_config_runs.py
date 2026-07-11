from pathlib import Path

from eleanity.config.project import EleanityProject, GateRule, write_default_project, load_project
from eleanity.core.engine import CompareEngine
from eleanity.core.runs_index import diff_runs, list_runs
from eleanity.gates.engine import evaluate_gates
from eleanity.models.schemas import ParityResult
from eleanity.playbook import get_playbook_entry
from eleanity.reporters.sarif import build_sarif
from eleanity.scenarios.suites import list_builtin_suites


def test_write_and_load_project(tmp_path):
    path = write_default_project(tmp_path / "eleanity.yaml")
    project = load_project(path)
    assert project.schema_version == "0.2"
    assert project.gates
    assert "transformers" in project.backend_profiles


def test_gates_fail_on_divergent_template():
    gates = [
        GateRule(name="prompt", layers=["template"], max_status=ParityResult.PASS),
    ]
    comparisons = {
        "vllm": {"template": {"result": "DIVERGENT", "details": {}}},
    }
    evaluation = evaluate_gates(gates, comparisons)
    assert evaluation.passed is False
    assert evaluation.exit_code == 1


def test_gates_allow_not_observable():
    gates = [
        GateRule(
            name="gen",
            layers=["generation"],
            max_status=ParityResult.PASS,
            allow=[ParityResult.NOT_OBSERVABLE],
        ),
    ]
    comparisons = {
        "vllm": {"generation": {"result": "NOT_OBSERVABLE", "details": {}}},
    }
    assert evaluate_gates(gates, comparisons).passed is True


def test_engine_writes_sarif_and_gates(tmp_path):
    project = EleanityProject(
        backends=["fake", "fake"],
        gates=[
            GateRule(name="tokens", layers=["tokens"], max_status=ParityResult.PASS),
        ],
    )
    engine = CompareEngine(project=project, runs_dir=tmp_path, parallel=False)
    result = engine.compare("demo", ["fake", "fake"])
    assert (tmp_path / result.run_id / "results.sarif").exists()
    assert result.gate_evaluation is not None
    assert result.gate_evaluation.passed is True
    data = (tmp_path / result.run_id / "result.json").read_text(encoding="utf-8")
    assert "gates" in data
    assert "timings_ms" in data


def test_runs_index_and_diff(tmp_path):
    engine = CompareEngine(runs_dir=tmp_path, parallel=False)
    a = engine.compare("demo", ["fake", "fake"])
    b = engine.compare("demo", ["fake", "fake"])
    runs = list_runs(tmp_path)
    assert len(runs) >= 2
    delta = diff_runs(a.run_id, b.run_id, tmp_path)
    assert delta["left_run_id"] == a.run_id
    assert "layer_delta" in delta


def test_playbook_and_suites_exist():
    assert get_playbook_entry("MISSING_ASSISTANT_TURN_TOKEN")
    names = {item["name"] for item in list_builtin_suites()}
    assert "qwen-parity" in names
    assert "tool-calling" in names


def test_sarif_build_from_divergent_diagnosis():
    payload = {
        "run_id": "abc",
        "diagnosis": {
            "status": "DIVERGENT",
            "summary": "template diverged",
            "first_divergence": "template",
            "probable_causes": [
                {"code": "MISSING_ASSISTANT_TURN_TOKEN", "confidence": 0.9, "message": "missing"}
            ],
        },
    }
    sarif = build_sarif(payload)
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"]
