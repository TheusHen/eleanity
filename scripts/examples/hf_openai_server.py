#!/usr/bin/env python3
"""Real OpenAI-compatible HTTP server backed by Hugging Face Transformers.

Loads actual model weights and serves:
  GET  /v1/models
  POST /v1/chat/completions

Used for honest cross-runtime demos:
  transformers (in-process adapter) × vllm/openai adapter → this server

Does NOT expose /tokenize or raw chat-template endpoints (same honesty
profile as many production OpenAI-compat gateways).

Optional --omit-generation-prompt simulates a common production bug where
the server applies the chat template without the assistant turn marker.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class ModelRuntime:
    """Thread-safe lazy load of tokenizer + causal LM."""

    def __init__(self, model_id: str, *, omit_generation_prompt: bool = False, max_new_tokens: int = 64):
        self.model_id = model_id
        self.omit_generation_prompt = omit_generation_prompt
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        import torch

        tok = self._tokenizer
        add_gen = not self.omit_generation_prompt
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_gen,
            )
        except Exception:
            # Fallback: plain role: content formatting
            parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            if add_gen:
                parts.append("assistant:")
            prompt = "\n".join(parts)

        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        n_new = int(max_tokens or self.max_new_tokens)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": n_new,
            "pad_token_id": tok.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[-1] :]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()


def make_handler(runtime: ModelRuntime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/v1/models"):
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": runtime.model_id,
                                "object": "model",
                                "owned_by": "eleanity-hf-server",
                            }
                        ],
                    },
                )
                return
            if self.path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok", "model": runtime.model_id})
                return
            self._json(404, {"error": {"message": f"not found: {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid json"}})
                return

            if not self.path.startswith("/v1/chat/completions"):
                self._json(404, {"error": {"message": f"not found: {self.path}"}})
                return

            messages = payload.get("messages") or []
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": {"message": "messages required"}})
                return
            norm = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in messages
                if isinstance(m, dict)
            ]
            max_tokens = payload.get("max_tokens")
            temperature = float(payload.get("temperature") or 0)
            try:
                text = runtime.chat(norm, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # surface load/runtime errors
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
                return

            self._json(
                200,
                {
                    "id": "chatcmpl-eleanity-hf",
                    "object": "chat.completion",
                    "model": runtime.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "eleanity_server": {
                        "backend": "transformers",
                        "omit_generation_prompt": runtime.omit_generation_prompt,
                    },
                },
            )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("ELEANITY_DEMO_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ELEANITY_DEMO_PORT", "8765")))
    parser.add_argument(
        "--omit-generation-prompt",
        action="store_true",
        help="Apply chat template without assistant generation prompt (forces template-class divergence).",
    )
    parser.add_argument("--preload", action="store_true", help="Load weights before accepting traffic.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    runtime = ModelRuntime(
        args.model,
        omit_generation_prompt=args.omit_generation_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    if args.preload:
        print(f"loading {args.model}…", flush=True)
        runtime.ensure_loaded()
        print("ready", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        json.dumps(
            {
                "event": "hf_openai_server_start",
                "url": f"http://{args.host}:{args.port}",
                "model": args.model,
                "omit_generation_prompt": args.omit_generation_prompt,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
