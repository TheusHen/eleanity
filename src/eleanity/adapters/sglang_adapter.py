from __future__ import annotations

import os

from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.models.schemas import ModelSpec


class SGLangAdapter(OpenAICompatAdapter):
    """SGLang OpenAI-compatible server (ELEANITY_SGLANG_URL)."""

    def __init__(self, model_ref: str, model_spec: ModelSpec | None = None):
        base = os.getenv("ELEANITY_SGLANG_URL", "").rstrip("/")
        key = os.getenv("ELEANITY_SGLANG_API_KEY")
        super().__init__(
            model_ref,
            base_url=base,
            name="sglang",
            model_spec=model_spec,
            api_key=key,
            tokenize_path=os.getenv("ELEANITY_SGLANG_TOKENIZE_PATH"),
            models_path="/v1/models",
            chat_path="/v1/chat/completions",
        )
        self.capabilities.notes["runtime"] = "SGLang via OpenAI-compatible HTTP"
