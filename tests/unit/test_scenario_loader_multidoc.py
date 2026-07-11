from eleanity.scenarios.loader import load_scenarios


def test_load_public_multidoc_fixtures():
    scenarios = load_scenarios("fixtures/public/chat-templates.yaml")
    assert len(scenarios) >= 5
    names = {s.name for s in scenarios}
    assert "system-user" in names
    assert "assistant-prefill" in names


def test_load_single_document_list_style():
    # qwen fixtures are multi-scenario single-file or multi-doc — still loadable
    scenarios = load_scenarios("fixtures/public/tokenizer-edge.yaml")
    assert scenarios
    assert all(s.messages for s in scenarios)
