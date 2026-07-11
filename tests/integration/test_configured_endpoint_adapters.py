import json

import httpx
import pytest

from eleanity.adapters.llamacpp_adapter import LlamaCppAdapter
from eleanity.adapters.vllm_adapter import VLLMAdapter
from eleanity.models.schemas import LayerState, Scenario

SCENARIO = Scenario(
    name="endpoint",
    messages=[{"role": "user", "content": "Olá"}],
    parameters={"max_tokens": 4, "temperature": 0},
)


def test_vllm_endpoint_mode_declares_observable_token_and_generation_layers(monkeypatch):
    monkeypatch.setenv("ELEANITY_VLLM_URL", "http://127.0.0.1:8000")

    adapter = VLLMAdapter("Qwen/Qwen2.5-0.5B-Instruct")

    assert adapter.capabilities.tokenize is True
    assert adapter.capabilities.stream is True


def test_llamacpp_endpoint_mode_declares_observable_token_and_generation_layers(monkeypatch):
    monkeypatch.setenv("ELEANITY_LLAMACPP_URL", "http://127.0.0.1:8080")

    adapter = LlamaCppAdapter("qwen.gguf")

    assert adapter.capabilities.tokenize is True
    assert adapter.capabilities.stream is True


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None):
        self.calls.append((url, json))
        for suffix, payload in self.routes.items():
            if url.endswith(suffix):
                return _FakeResponse(payload)
        return _FakeResponse({"error": "not found"}, status_code=404)

    def get(self, url):
        return _FakeResponse({"data": []})


@pytest.mark.parametrize(
    ("adapter_class", "url_variable", "base_url", "model"),
    [
        (VLLMAdapter, "ELEANITY_VLLM_URL", "http://vllm", "Qwen"),
        (LlamaCppAdapter, "ELEANITY_LLAMACPP_URL", "http://llama", "qwen.gguf"),
    ],
)
def test_configured_runtime_endpoint_observes_tokens_and_greedy_generation(
    monkeypatch, adapter_class, url_variable, base_url, model
):
    monkeypatch.setenv(url_variable, base_url)
    client = _FakeClient(
        {
            "/tokenize": {"tokens": [1, 2]},
            "/v1/chat/completions": {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        }
    )
    monkeypatch.setattr(
        "eleanity.adapters.openai_compat.httpx.Client",
        lambda *a, **k: client,
    )
    adapter = adapter_class(model)

    tokens = adapter.tokenize("prompt")
    generation = adapter.generate(SCENARIO)

    assert tokens.state == LayerState.OBSERVED
    assert tokens.data["ids"] == [1, 2]
    assert generation.state == LayerState.OBSERVED
    assert generation.data["text"] == "ok"
    assert generation.data["ids"] == []
    assert generation.data["stop_reason"] == "stop"
    assert any(url.endswith("/tokenize") for url, _ in client.calls)
    assert any(url.endswith("/v1/chat/completions") for url, _ in client.calls)
