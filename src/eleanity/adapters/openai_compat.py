from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from eleanity.adapters.base import BackendAdapter, CapabilitySet, HealthcheckResult
from eleanity.fingerprints import collect_environment_fingerprint
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    ModelSpec,
    Scenario,
)
from eleanity.utils.logging import get_logger, log_event

logger = get_logger("eleanity.adapters.openai")


class OpenAICompatAdapter(BackendAdapter):
    """First-class OpenAI-compatible HTTP adapter.

    Works with vLLM, llama.cpp server, SGLang, Ollama (/v1), LiteLLM, and any
    endpoint exposing chat completions (+ optional tokenize).
    """

    def __init__(
        self,
        model_ref: str,
        *,
        base_url: str,
        name: str = "openai",
        model_spec: ModelSpec | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        tokenize_path: str | None = None,
        models_path: str = "/v1/models",
        chat_path: str = "/v1/chat/completions",
    ):
        self.model_ref = model_ref
        self.model_spec = model_spec or ModelSpec(id=model_ref)
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.api_key = api_key
        self.timeout = timeout
        self.tokenize_path = tokenize_path
        self.models_path = models_path
        self.chat_path = chat_path
        self.version = "0.2"
        configured = bool(self.base_url)
        self.capabilities = CapabilitySet(
            render=False,
            tokenize=configured and tokenize_path is not None,
            logits=False,
            stream=configured,
            tools=configured,
            artifact=True,
            template=False,
            tokenization=configured and tokenize_path is not None,
            special_tokens=False,
            generation=configured,
            structured_output=configured,
            streaming=configured,
            usage=configured,
            errors=True,
            healthcheck=configured,
            notes={
                "render": "OpenAI-compatible servers do not expose apply_chat_template text",
                "logits": "Logits are not part of the OpenAI chat surface",
            },
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, headers=self._headers())

    def healthcheck(self) -> HealthcheckResult:
        if not self.base_url:
            return HealthcheckResult(ok=False, detail="base_url empty")
        started = time.perf_counter()
        try:
            with self._client() as client:
                response = client.get(self._url(self.models_path))
                latency = (time.perf_counter() - started) * 1000
                if response.status_code >= 400:
                    return HealthcheckResult(
                        ok=False,
                        detail=f"HTTP {response.status_code}",
                        latency_ms=latency,
                        endpoint=self.base_url,
                    )
                return HealthcheckResult(
                    ok=True,
                    detail="reachable",
                    latency_ms=round(latency, 2),
                    endpoint=self.base_url,
                )
        except httpx.HTTPError as error:
            return HealthcheckResult(ok=False, detail=str(error), endpoint=self.base_url)

    def fingerprint(self, model_ref: str) -> ArtifactFingerprint:
        env = collect_environment_fingerprint()
        return ArtifactFingerprint(
            model_ref=model_ref,
            revision=self.model_spec.revision,
            quantization=self.model_spec.quantization,
            dtype=self.model_spec.dtype,
            runtime_version=self.version,
            library_versions=env.packages,
            python_version=env.python_version,
            os=env.platform,
            cpu_arch=env.machine,
            gpu=env.gpu_name,
            cuda_or_rocm=env.cuda_version,
            backend_flags={
                "runtime": self.name,
                "base_url": self.base_url,
                "chat_path": self.chat_path,
                "tokenize_path": self.tokenize_path,
            },
        )

    def render(self, scenario: Scenario) -> LayerObservation:
        return self.not_observable(
            "template",
            f"{self.name} does not expose a canonical rendered chat template",
        )

    def tokenize(self, rendered: str) -> LayerObservation:
        if not self.tokenize_path:
            return self.not_observable("tokens", f"{self.name}: no tokenize endpoint configured")
        payload: dict[str, Any] = {"model": self.model_ref, "prompt": rendered}
        # llama.cpp often wants "content"
        alt_payload = {"content": rendered, "model": self.model_ref}
        try:
            with self._client() as client:
                response = client.post(self._url(self.tokenize_path), json=payload)
                if response.status_code >= 400:
                    response = client.post(self._url(self.tokenize_path), json=alt_payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as error:
            return self.not_observable("tokens", f"{self.name} tokenize failed: {error}")

        ids = data.get("tokens") or data.get("input_ids") or data.get("tokens_list") or []
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            return self.not_observable("tokens", f"{self.name} tokenize returned non-integer ids")
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "ids": ids,
                "token_ids": ids,
                "count": len(ids),
                "source": f"{self.name}/tokenize",
            },
        )

    def forward(self, tokens: LayerObservation) -> LayerObservation:
        return self.not_observable("logits", f"{self.name} does not expose logits")

    def _chat_payload(self, scenario: Scenario, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_ref,
            "messages": [message.model_dump() for message in scenario.messages],
            "max_tokens": int(scenario.parameters.get("max_tokens", 64)),
            "temperature": float(scenario.parameters.get("temperature", 0)),
            "stream": stream,
        }
        if scenario.parameters.get("top_p") is not None:
            payload["top_p"] = float(scenario.parameters["top_p"])
        if scenario.parameters.get("top_k") is not None:
            payload["top_k"] = int(scenario.parameters["top_k"])
        if "seed" in scenario.parameters:
            payload["seed"] = int(scenario.parameters["seed"])
        if scenario.parameters.get("stop"):
            payload["stop"] = scenario.parameters["stop"]
        if scenario.parameters.get("response_format"):
            payload["response_format"] = scenario.parameters["response_format"]
        if scenario.parameters.get("tools"):
            payload["tools"] = scenario.parameters["tools"]
        return payload

    def generate(self, scenario: Scenario) -> LayerObservation:
        if not self.base_url:
            return self.not_observable("generation", f"{self.name}: base_url empty")
        payload = self._chat_payload(scenario, stream=False)
        started = time.perf_counter()
        try:
            with self._client() as client:
                response = client.post(self._url(self.chat_path), json=payload)
                latency = (time.perf_counter() - started) * 1000
                status = response.status_code
                body = response.json() if response.content else {}
        except (httpx.HTTPError, ValueError) as error:
            log_event(logger, "generate_error", backend=self.name, error=str(error))
            return self.error("generation", str(error))

        if status >= 400:
            return LayerObservation(
                state=LayerState.ERROR,
                note=f"HTTP {status}",
                data={
                    "http_status": status,
                    "error": body,
                    "latency_ms": round(latency, 2),
                },
            )

        try:
            choice = (body.get("choices") or [])[0]
        except (IndexError, TypeError) as error:
            return self.error("generation", f"invalid choices: {error}")

        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        tool_calls = message.get("tool_calls")
        ids = choice.get("tokens") if isinstance(choice.get("tokens"), list) else []
        usage = body.get("usage") or {}
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "text": text,
                "ids": ids,
                "token_ids": ids,
                "stop_reason": choice.get("finish_reason") or "unknown",
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "tool_calls": tool_calls,
                "http_status": status,
                "latency_ms": round(latency, 2),
                "source": f"{self.name}{self.chat_path}",
                "seed": scenario.parameters.get("seed"),
            },
        )

    def stream_generate(self, scenario: Scenario) -> LayerObservation:
        if not self.base_url:
            return self.not_observable("streaming", f"{self.name}: base_url empty")
        payload = self._chat_payload(scenario, stream=True)
        chunks: list[dict[str, Any]] = []
        event_types: list[str] = []
        text_parts: list[str] = []
        finish_reason = None
        ordered = True
        last_index = -1
        ttft_ms = None
        started = time.perf_counter()
        try:
            with self._client() as client:
                with client.stream("POST", self._url(self.chat_path), json=payload) as response:
                    status = response.status_code
                    if status >= 400:
                        return LayerObservation(
                            state=LayerState.ERROR,
                            note=f"HTTP {status}",
                            data={"http_status": status},
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data_str = line[5:].strip() if line.startswith("data:") else line.strip()
                        if data_str == "[DONE]":
                            event_types.append("done")
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            event_types.append("non_json")
                            ordered = False
                            continue
                        chunks.append(event)
                        event_types.append("chunk")
                        try:
                            choice = (event.get("choices") or [{}])[0]
                            idx = choice.get("index")
                            if isinstance(idx, int):
                                if idx < last_index:
                                    ordered = False
                                last_index = idx
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                if ttft_ms is None:
                                    ttft_ms = (time.perf_counter() - started) * 1000
                                text_parts.append(str(delta["content"]))
                            fr = choice.get("finish_reason")
                            if fr:
                                finish_reason = fr
                        except (IndexError, AttributeError, TypeError):
                            pass
        except httpx.HTTPError as error:
            return self.error("streaming", str(error))

        latency = (time.perf_counter() - started) * 1000
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "chunk_count": len(chunks),
                "event_types": event_types,
                "text": "".join(text_parts),
                "finish_reason": finish_reason,
                "latency_ms": round(latency, 2),
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "ordered": ordered,
                "has_done": "done" in event_types,
                "source": f"{self.name}{self.chat_path}?stream=1",
            },
        )

    def structured(self, scenario: Scenario) -> LayerObservation:
        from eleanity.comparators.structured import build_structured_observation

        generation = self.generate(scenario)
        if generation.state != LayerState.OBSERVED:
            return generation
        schema = None
        if isinstance(scenario.parameters.get("response_format"), dict):
            schema = scenario.parameters["response_format"].get("json_schema") or scenario.parameters[
                "response_format"
            ].get("schema")
        schema = schema or scenario.parameters.get("json_schema")
        data = build_structured_observation(
            text=str(generation.data.get("text") or ""),
            tool_calls=generation.data.get("tool_calls"),
            stop_reason=generation.data.get("stop_reason"),
            json_schema=schema if isinstance(schema, dict) else None,
            required_keys=scenario.parameters.get("required_keys"),
            expected_tool_names=scenario.parameters.get("expected_tool_names"),
        )
        return LayerObservation(state=LayerState.OBSERVED, data=data)

    def api_probe(self, scenario: Scenario) -> LayerObservation:
        health = self.healthcheck()
        generation = self.generate(scenario)
        return LayerObservation(
            state=LayerState.OBSERVED if generation.state != LayerState.ERROR else LayerState.ERROR,
            data={
                "health_ok": health.ok,
                "health_detail": health.detail,
                "http_status": generation.data.get("http_status"),
                "has_usage": bool((generation.data or {}).get("usage")),
                "finish_reason": (generation.data or {}).get("finish_reason")
                or (generation.data or {}).get("stop_reason"),
                "openai_shape": bool(
                    isinstance(generation.data, dict)
                    and ("text" in generation.data or generation.state != LayerState.OBSERVED)
                ),
                "latency_ms": generation.data.get("latency_ms") or health.latency_ms,
                "error": generation.note if generation.state == LayerState.ERROR else None,
            },
            note=generation.note,
        )
