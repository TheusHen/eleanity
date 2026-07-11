import struct
from pathlib import Path

from eleanity.adapters.fake import FakeAdapter
from eleanity.adapters.sdk import check_adapter_compliance
from eleanity.certification import certify_runtime
from eleanity.comparators.api import compare_streaming
from eleanity.comparators.structured import build_structured_observation, compare_structured, validate_json_schema
from eleanity.core.batch_report import run_multi_model_batch
from eleanity.core.engine import CompareEngine
from eleanity.core.golden import golden_gate, save_golden
from eleanity.core.observe import observe
from eleanity.fingerprints.gguf import inspect_gguf
from eleanity.integrations.artifacts import LocalArtifactSink
from eleanity.models.schemas import ParityResult, Scenario


def test_json_schema_validation():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    ok, err = validate_json_schema({"ok": True}, schema)
    assert ok is True
    ok, err = validate_json_schema({"ok": "no"}, schema)
    assert ok is False


def test_structured_tool_argument_diff():
    left = {
        "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"SP"}'}}],
        "stop_reason": "tool_calls",
    }
    right = {
        "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"RJ"}'}}],
        "stop_reason": "tool_calls",
    }
    result = compare_structured(left, right)
    assert result.result == ParityResult.DIVERGENT
    assert "arguments" in result.details.get("reason", "")


def test_build_structured_observation_with_schema():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    data = build_structured_observation(text='{"ok": true}', json_schema=schema)
    assert data["is_json"] is True
    assert data["schema_valid"] is True


def test_streaming_detects_missing_terminal():
    left = {
        "chunk_count": 2,
        "event_types": ["chunk", "chunk", "done"],
        "ordered": True,
        "finish_reason": "stop",
        "text": "hi",
    }
    right = {
        "chunk_count": 1,
        "event_types": ["chunk"],
        "ordered": True,
        "finish_reason": None,
        "text": "hi",
    }
    result = compare_streaming(left, right)
    assert result.result == ParityResult.DIVERGENT
    assert "candidate_missing_terminal" in result.details["issues"] or "finish_reason" in result.details["issues"]


def test_gguf_deep_parse_roundtrip(tmp_path):
    # Minimal synthetic GGUF-like header with one string KV is complex;
    # write invalid magic and ensure honest failure, plus empty valid-ish path via real header only.
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"NOTG" + b"\x00" * 20)
    info = inspect_gguf(bad)
    assert info["ok"] is False

    # Valid magic + version + counts + one string key/value
    path = tmp_path / "ok.gguf"
    key = b"general.architecture"
    value = b"llama"
    body = bytearray()
    body += b"GGUF"
    body += struct.pack("<I", 3)  # version
    body += struct.pack("<Q", 0)  # tensor count
    body += struct.pack("<Q", 1)  # kv count
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 8)  # string type
    body += struct.pack("<Q", len(value)) + value
    path.write_bytes(bytes(body))
    info = inspect_gguf(path, deep=True)
    assert info["ok"] is True
    assert info["architecture"] == "llama"
    assert "parity_fingerprint" in info


def test_sdk_compliance_and_certification():
    adapter = FakeAdapter()
    report = check_adapter_compliance(adapter, model="demo")
    assert report.passed is True
    cert = certify_runtime(adapter, model="demo")
    assert cert.passed is True
    assert cert.level in {"bronze", "silver", "gold"}


def test_batch_multi_model_and_local_export(tmp_path):
    scenario = Scenario(
        name="batch-sc",
        messages=[{"role": "user", "content": "hi"}],
        observe=["template", "tokens"],
    )
    engine = CompareEngine(runs_dir=tmp_path / "runs", parallel=False)
    report = run_multi_model_batch(
        [
            ("demo", ["fake", "fake"], scenario),
            ("demo-2", ["fake", "fake"], scenario),
        ],
        engine=engine,
        output_dir=tmp_path / "batches",
    )
    assert report.summary["job_count"] == 2
    assert (report.path / "batch.md").exists()
    # export first run
    run_dir = tmp_path / "runs" / report.results[0].run_id
    out = LocalArtifactSink(tmp_path / "exports").publish(run_dir)
    assert Path(out["path"]).exists()


def test_golden_gate_pass(tmp_path):
    scenario = Scenario(
        name="g",
        messages=[{"role": "user", "content": "hi"}],
        observe=["template", "tokens"],
    )
    live = observe(FakeAdapter(), scenario, "demo")
    golden_path = save_golden(live, tmp_path)
    result = golden_gate(live, golden_path, scenario, layers=["template", "tokens"])
    assert result["passed"] is True
