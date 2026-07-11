import pytest

from eleanity.adapters import adapter_for, available_adapters
from eleanity.adapters.sdk import check_adapter_compliance


@pytest.mark.parametrize("name", ["fake", "vllm", "llamacpp", "ollama", "sglang", "tgi", "openai"])
def test_builtin_http_family_and_fake_are_compliant(name):
    assert name in available_adapters()
    adapter = adapter_for(name, "demo")
    report = check_adapter_compliance(adapter, model="demo", probe_runtime=True)
    # Degraded HTTP without URL may warn but should not fail required methods
    assert report.passed is True
    assert not any(i.severity == "error" for i in report.issues)
