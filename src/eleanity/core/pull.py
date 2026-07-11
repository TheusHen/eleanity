from __future__ import annotations

from pathlib import Path
from typing import Any

from eleanity.utils.logging import get_logger, log_event
from eleanity.utils.security import sanitize_path

logger = get_logger("eleanity.pull")


def pull_model(
    model_id: str,
    *,
    revision: str | None = "main",
    trust_remote_code: bool = False,
    tokenizer_only: bool = False,
) -> dict[str, Any]:
    """Download model/tokenizer artifacts into the local Hugging Face cache.

    Requires optional transformers/huggingface_hub. Never enables trust_remote_code
    unless the caller passes True explicitly.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "pull requires huggingface_hub (install eleanity[transformers])"
        ) from error

    log_event(logger, "pull_start", model=model_id, revision=revision, tokenizer_only=tokenizer_only)
    allow_patterns = None
    if tokenizer_only:
        allow_patterns = [
            "tokenizer*",
            "vocab*",
            "merges*",
            "special_tokens*",
            "added_tokens*",
            "chat_template*",
            "config.json",
            "generation_config.json",
            "*.jinja",
            "tokenizer_config.json",
        ]
    local_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    # Optional: validate tokenizer loads
    tokenizer_ok = False
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            local_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_ok = True
    except Exception as error:
        log_event(logger, "pull_tokenizer_validate_failed", error=str(error))

    result = {
        "model_id": model_id,
        "revision": revision,
        "local_path": sanitize_path(local_dir),
        "local_path_raw": str(Path(local_dir)),
        "tokenizer_only": tokenizer_only,
        "tokenizer_loadable": tokenizer_ok,
        "trust_remote_code": trust_remote_code,
    }
    log_event(logger, "pull_done", model=model_id, path=result["local_path"])
    return result
