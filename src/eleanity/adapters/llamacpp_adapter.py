from __future__ import annotations

import os

from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.models.schemas import ArtifactFingerprint, ModelSpec


class LlamaCppAdapter(OpenAICompatAdapter):
    """llama.cpp server adapter — OpenAI-compatible HTTP (ELEANITY_LLAMACPP_URL)."""

    def __init__(self, model_ref: str, model_spec: ModelSpec | None = None):
        base = os.getenv("ELEANITY_LLAMACPP_URL", "").rstrip("/")
        super().__init__(
            model_ref,
            base_url=base,
            name="llamacpp",
            model_spec=model_spec,
            tokenize_path="/tokenize" if base else None,
            models_path="/v1/models",
            chat_path="/v1/chat/completions",
        )
        try:
            import llama_cpp  # noqa: F401

            self.version = getattr(llama_cpp, "__version__", self.version)
            if not base:
                self.capabilities.notes["runtime"] = (
                    f"llama-cpp-python {self.version} installed; set ELEANITY_LLAMACPP_URL"
                )
        except ImportError:
            if not base:
                self.capabilities.generation = False
                self.capabilities.tokenize = False
                self.capabilities.tokenization = False
                self.capabilities.streaming = False

    def fingerprint(self, model_ref: str) -> ArtifactFingerprint:
        fp = super().fingerprint(model_ref)
        updates: dict = {
            "quantization": self.model_spec.quantization or "GGUF",
            "gguf_metadata": {"format": "gguf", "source": self.base_url or "unset"},
        }
        local = self.model_spec.local_path or (model_ref if str(model_ref).lower().endswith(".gguf") else None)
        if local:
            try:
                from eleanity.fingerprints.gguf import gguf_to_artifact_fields

                updates.update(gguf_to_artifact_fields(local))
            except Exception:
                pass
        return fp.model_copy(update=updates)
