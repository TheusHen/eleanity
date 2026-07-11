from __future__ import annotations

import os
from typing import Any

from eleanity.adapters.base import CapabilitySet, HealthcheckResult
from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.models.schemas import LayerObservation, LayerState, ModelSpec, Scenario


class VLLMAdapter(OpenAICompatAdapter):
    """vLLM adapter — HTTP OpenAI-compat (default) or embedded Engine mode.

    Modes:
      - http (default): ELEANITY_VLLM_URL or base_url kwarg → OpenAI-compatible server
        (vLLM serve, LM Studio, etc.)
      - embedded: ELEANITY_VLLM_MODE=embedded and ``vllm`` package installed —
        uses vLLM Engine/LLM in-process when available.
    """

    def __init__(
        self,
        model_ref: str,
        model_spec: ModelSpec | None = None,
        *,
        base_url: str | None = None,
        mode: str | None = None,
    ):
        env_url = os.getenv("ELEANITY_VLLM_URL", "").rstrip("/")
        url = (base_url or env_url or "").rstrip("/")
        self.mode = (mode or os.getenv("ELEANITY_VLLM_MODE") or ("http" if url else "auto")).lower()
        self._embedded_llm = None
        self._embedded_tokenizer = None

        # Prefer HTTP when URL present
        if self.mode == "auto":
            self.mode = "http" if url else "embedded"

        super().__init__(
            model_ref,
            base_url=url,
            name="vllm",
            model_spec=model_spec,
            tokenize_path=os.getenv("ELEANITY_VLLM_TOKENIZE_PATH")
            or ("/tokenize" if url else None),
            models_path="/v1/models",
            chat_path="/v1/chat/completions",
        )

        try:
            import vllm  # noqa: F401

            self.version = getattr(vllm, "__version__", self.version)
        except ImportError:
            vllm = None  # type: ignore

        if self.mode == "embedded" and not url:
            self._init_embedded()
        elif not url:
            self.capabilities.notes["runtime"] = (
                "set ELEANITY_VLLM_URL for HTTP observations, "
                "or ELEANITY_VLLM_MODE=embedded with vllm installed"
            )
            if vllm is None:
                self.capabilities.generation = False
                self.capabilities.tokenize = False
                self.capabilities.tokenization = False
                self.capabilities.streaming = False
        else:
            # HTTP mode: try richer observation paths
            self.capabilities.notes["mode"] = "http"
            self.capabilities.notes["tokenize"] = (
                f"POST {self.tokenize_path}" if self.tokenize_path else "no tokenize path"
            )

    def _init_embedded(self) -> None:
        """Best-effort embedded vLLM / transformers-fallback path."""

        try:
            from vllm import LLM  # type: ignore

            # Lightweight deferred note — full load happens on first generate if needed
            self.capabilities.generation = True
            self.capabilities.tokenize = True
            self.capabilities.tokenization = True
            self.capabilities.template = True
            self.capabilities.rendered_prompt = True
            self.capabilities.notes["mode"] = "embedded"
            self.capabilities.notes["runtime"] = f"embedded vLLM {self.version}"
            self._embedded_cls = LLM
            self.mode = "embedded"
        except Exception as error:
            self.capabilities.notes["embedded_error"] = str(error)
            self.capabilities.generation = False
            self.mode = "http"

    def _ensure_embedded(self) -> None:
        if self._embedded_llm is not None:
            return
        if not hasattr(self, "_embedded_cls"):
            raise RuntimeError("embedded vLLM not available")
        self._embedded_llm = self._embedded_cls(
            model=self.model_ref,
            trust_remote_code=bool(self.model_spec.trust_remote_code),
        )
        self._embedded_tokenizer = getattr(self._embedded_llm, "get_tokenizer", lambda: None)()

    def render(self, scenario: Scenario) -> LayerObservation:
        if self.mode == "embedded":
            try:
                self._ensure_embedded()
                tok = self._embedded_tokenizer
                if tok is not None and hasattr(tok, "apply_chat_template"):
                    messages = [m.model_dump() for m in scenario.messages]
                    text = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=scenario.generation.add_generation_prompt,
                    )
                    return LayerObservation(
                        state=LayerState.OBSERVED,
                        data={
                            "text": text,
                            "rendered_text": text,
                            "add_generation_prompt": scenario.generation.add_generation_prompt,
                            "source": "vllm.embedded.apply_chat_template",
                        },
                        origin="vllm.embedded.apply_chat_template",
                        origin_kind="embedded",
                    )
            except Exception as error:
                return LayerObservation(
                    state=LayerState.FAILED,
                    note=str(error),
                    origin="vllm.embedded.render",
                    origin_kind="embedded",
                )
        # HTTP: try optional template debug endpoints used by some servers
        if self.base_url:
            for path in ("/v1/chat/template", "/template", "/v1/apply_chat_template"):
                try:
                    with self._client() as client:
                        payload = {
                            "model": self.model_ref,
                            "messages": [m.model_dump() for m in scenario.messages],
                            "add_generation_prompt": scenario.generation.add_generation_prompt,
                        }
                        response = client.post(self._url(path), json=payload)
                        if response.status_code < 400:
                            data = response.json()
                            text = data.get("prompt") or data.get("text") or data.get("rendered")
                            if text:
                                return LayerObservation(
                                    state=LayerState.OBSERVED,
                                    data={
                                        "text": text,
                                        "rendered_text": text,
                                        "source": f"vllm.http:{path}",
                                    },
                                    origin=f"vllm.http:{path}",
                                    origin_kind="http",
                                )
                except Exception:
                    continue
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note="vllm HTTP server does not expose a canonical rendered chat template",
            origin="vllm.http:template",
            origin_kind="unavailable",
        )

    def tokenize(self, rendered: str) -> LayerObservation:
        if self.mode == "embedded":
            try:
                self._ensure_embedded()
                tok = self._embedded_tokenizer
                if tok is not None:
                    ids = tok.encode(rendered)
                    if hasattr(ids, "ids"):
                        ids = list(ids.ids)
                    else:
                        ids = list(ids)
                    return LayerObservation(
                        state=LayerState.OBSERVED,
                        data={"ids": ids, "token_ids": ids, "count": len(ids), "source": "vllm.embedded"},
                        origin="vllm.embedded.tokenize",
                        origin_kind="embedded",
                    )
            except Exception as error:
                return LayerObservation(
                    state=LayerState.FAILED,
                    note=str(error),
                    origin="vllm.embedded.tokenize",
                    origin_kind="embedded",
                )
        obs = super().tokenize(rendered)
        if obs.state == LayerState.OBSERVED:
            return obs.model_copy(
                update={"origin": f"vllm.http:{self.tokenize_path}", "origin_kind": "http"}
            )
        # Try alternate tokenize paths used by some OpenAI-compat servers
        if self.base_url:
            for path in ("/tokenize", "/v1/tokenize", "/completion/tokenize"):
                try:
                    with self._client() as client:
                        response = client.post(
                            self._url(path),
                            json={"model": self.model_ref, "prompt": rendered},
                        )
                        if response.status_code >= 400:
                            continue
                        data = response.json()
                        ids = data.get("tokens") or data.get("input_ids") or data.get("tokens_list") or []
                        if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
                            return LayerObservation(
                                state=LayerState.OBSERVED,
                                data={
                                    "ids": ids,
                                    "token_ids": ids,
                                    "count": len(ids),
                                    "source": f"vllm.http:{path}",
                                },
                                origin=f"vllm.http:{path}",
                                origin_kind="http",
                            )
                except Exception:
                    continue
        return LayerObservation(
            state=LayerState.NOT_OBSERVABLE,
            note=obs.note or "vllm tokenize endpoint unavailable",
            origin="vllm.http:tokenize",
            origin_kind="unavailable",
        )

    def generate(self, scenario: Scenario) -> LayerObservation:
        if self.mode == "embedded" and not self.base_url:
            if not hasattr(self, "_embedded_cls"):
                return LayerObservation(
                    state=LayerState.NOT_OBSERVABLE,
                    note="embedded vLLM unavailable; set ELEANITY_VLLM_URL for HTTP mode",
                    origin="vllm.embedded.generate",
                    origin_kind="unavailable",
                )
            try:
                self._ensure_embedded()
                from vllm import SamplingParams  # type: ignore

                params = scenario.parameters or {}
                sp = SamplingParams(
                    temperature=float(params.get("temperature", 0)),
                    max_tokens=int(params.get("max_tokens", 64)),
                    seed=int(params["seed"]) if params.get("seed") is not None else None,
                )
                messages = [m.model_dump() for m in scenario.messages]
                # Prefer chat API if present
                outputs = self._embedded_llm.chat(messages, sp)  # type: ignore[union-attr]
                out0 = outputs[0]
                text = getattr(out0.outputs[0], "text", "") if out0.outputs else ""
                token_ids = list(getattr(out0.outputs[0], "token_ids", []) or [])
                return LayerObservation(
                    state=LayerState.OBSERVED,
                    data={
                        "text": text,
                        "ids": token_ids,
                        "token_ids": token_ids,
                        "stop_reason": "stop",
                        "source": "vllm.embedded.chat",
                        "seed": params.get("seed"),
                    },
                    origin="vllm.embedded.chat",
                    origin_kind="embedded",
                )
            except Exception as error:
                return LayerObservation(
                    state=LayerState.NOT_OBSERVABLE,
                    note=f"embedded generate failed: {error}",
                    origin="vllm.embedded.generate",
                    origin_kind="unavailable",
                )
        obs = super().generate(scenario)
        # Capture prompt tokens from usage when ids absent
        if obs.state == LayerState.OBSERVED and obs.data is not None:
            data = dict(obs.data)
            if not data.get("ids") and data.get("usage"):
                data["ids_source"] = "unavailable"
            return obs.model_copy(
                update={
                    "data": data,
                    "origin": f"vllm.http:{self.chat_path}",
                    "origin_kind": "http",
                }
            )
        return obs.model_copy(
            update={"origin": f"vllm.http:{self.chat_path}", "origin_kind": "http"}
        ) if isinstance(obs, LayerObservation) else obs

    def healthcheck(self) -> HealthcheckResult:
        if self.mode == "embedded" and not self.base_url:
            try:
                import vllm  # noqa: F401

                return HealthcheckResult(ok=True, detail=f"embedded vllm {self.version}")
            except ImportError:
                return HealthcheckResult(ok=False, detail="vllm package not installed")
        return super().healthcheck()
