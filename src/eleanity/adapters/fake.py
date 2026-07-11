from eleanity.adapters.base import BackendAdapter, CapabilitySet, HealthcheckResult
from eleanity.models.schemas import (
    ArtifactFingerprint,
    LayerObservation,
    LayerState,
    PromptObservation,
    Scenario,
    TokenObservation,
)
from eleanity.utils.hashing import text_sha256


class FakeAdapter(BackendAdapter):
    """Deterministic in-process adapter used by unit tests and doctor checks."""

    name = "fake"
    version = "0.2"
    capabilities = CapabilitySet(
        render=True,
        tokenize=True,
        logits=False,
        stream=True,
        tools=False,
        artifact=True,
        template=True,
        rendered_prompt=True,
        tokenization=True,
        special_tokens=True,
        generation=True,
        structured_output=True,
        streaming=True,
        usage=True,
        healthcheck=True,
    )

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(ok=True, detail="fake always healthy", latency_ms=0.1)

    def fingerprint(self, model_ref: str) -> ArtifactFingerprint:
        return ArtifactFingerprint(
            model_ref=model_ref,
            tokenizer="fake",
            chat_template_hash=ArtifactFingerprint.text_hash("role: content"),
            dtype="n/a",
            model_type="fake",
            architecture="FakeForCausalLM",
            runtime_version=self.version,
            backend_flags={"runtime": "fake"},
            special_tokens={
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "unk_token_id": 3,
            },
        )

    def special_tokens(self) -> LayerObservation:
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "unk_token_id": 3,
                "additional_special_tokens": [],
                "added_special_tokens": [],
                "vocab_size": 256,
            },
        )

    def render(self, scenario: Scenario) -> LayerObservation:
        text = "\n".join(f"{message.role}: {message.content}" for message in scenario.messages)
        if scenario.generation.add_generation_prompt:
            text += "\nassistant:"
        roles = [message.role for message in scenario.messages]
        obs = PromptObservation(
            chat_template_source="fake://role-colon",
            chat_template_hash=text_sha256(text),
            add_generation_prompt=scenario.generation.add_generation_prompt,
            continue_final_message=scenario.generation.continue_final_message,
            rendered_text=text,
            rendered_utf8_hex=text.encode("utf-8").hex(),
            rendered_byte_length=len(text.encode("utf-8")),
            rendered_char_length=len(text),
            roles=roles,
            tools_included=False,
            special_markers=["assistant:"] if scenario.generation.add_generation_prompt else [],
            text=text,
        )
        return LayerObservation(state=LayerState.OBSERVED, data=obs.to_layer_data())

    def tokenize(self, rendered: str) -> LayerObservation:
        ids = list(rendered.encode("utf-8"))
        strings = [chr(i) if 32 <= i < 127 else f"<{i}>" for i in ids]
        obs = TokenObservation(
            token_ids=ids,
            token_strings=strings,
            decoded_text=rendered,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            unk_token_id=3,
            added_special_tokens=[],
            original_length=len(ids),
            final_length=len(ids),
            truncated=False,
            padding_side="right",
            truncation_side="right",
            special_token_positions=[],
            special_token_count=0,
            add_special_tokens=False,
        )
        return LayerObservation(state=LayerState.OBSERVED, data=obs.to_layer_data())

    def forward(self, tokens: LayerObservation) -> LayerObservation:
        return LayerObservation(state=LayerState.NOT_OBSERVABLE, note="fake has no logits")

    def generate(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "text": "fake generation",
                "ids": [102, 97, 107, 101],
                "token_ids": [102, 97, 107, 101],
                "stop_reason": "length",
                "finish_reason": "length",
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
                "seed": int(scenario.parameters.get("seed", 42)),
            },
        )

    def stream_generate(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "chunk_count": 2,
                "event_types": ["chunk", "chunk", "done"],
                "text": "fake generation",
                "finish_reason": "length",
                "ordered": True,
            },
        )

    def structured(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "raw_text": '{"ok": true}',
                "parsed": {"ok": True},
                "is_json": True,
                "parse_error": None,
                "tool_calls": None,
                "stop_reason": "stop",
                "schema_valid": True,
            },
        )

    def api_probe(self, scenario: Scenario) -> LayerObservation:
        return LayerObservation(
            state=LayerState.OBSERVED,
            data={
                "health_ok": True,
                "http_status": 200,
                "has_usage": True,
                "finish_reason": "length",
                "openai_shape": True,
                "latency_ms": 0.1,
            },
        )
