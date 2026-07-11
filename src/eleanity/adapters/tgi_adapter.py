from __future__ import annotations

import os

from eleanity.adapters.openai_compat import OpenAICompatAdapter
from eleanity.models.schemas import ModelSpec


class TGIAdapter(OpenAICompatAdapter):
    """Text Generation Inference (TGI) OpenAI-compatible endpoint (ELEANITY_TGI_URL)."""

    def __init__(self, model_ref: str, model_spec: ModelSpec | None = None):
        base = os.getenv("ELEANITY_TGI_URL", "").rstrip("/")
        key = os.getenv("ELEANITY_TGI_API_KEY") or os.getenv("HF_TOKEN")
        super().__init__(
            model_ref,
            base_url=base,
            name="tgi",
            model_spec=model_spec,
            api_key=key,
            tokenize_path=None,
            models_path="/v1/models",
            chat_path="/v1/chat/completions",
        )
        self.capabilities.notes["runtime"] = "Hugging Face TGI via OpenAI-compatible HTTP"
