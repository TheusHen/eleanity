from __future__ import annotations

import os

from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.models.schemas import ModelSpec


class OllamaAdapter(OpenAICompatAdapter):
    """Ollama via its OpenAI-compatible /v1 surface."""

    def __init__(self, model_ref: str, model_spec: ModelSpec | None = None):
        base = os.getenv("ELEANITY_OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        super().__init__(
            model_ref,
            base_url=base.rstrip("/"),
            name="ollama",
            model_spec=model_spec,
            tokenize_path=None,
            models_path="/api/tags",
            chat_path="/v1/chat/completions",
        )
        # Ollama may only speak native API if /v1 is disabled — keep honest notes.
        self.capabilities.notes["api"] = "Uses Ollama OpenAI-compatible /v1 when available"
