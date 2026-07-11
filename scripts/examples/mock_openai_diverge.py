#!/usr/bin/env python3
"""Minimal OpenAI-compatible server that diverges on chat templates.

Starts on 127.0.0.1:8765 by default. Used to demo:
  transformers (real) × vllm adapter (HTTP → this server)

The server:
  - answers /v1/models
  - answers /v1/chat/completions with a fixed string
  - does NOT expose /tokenize or template endpoints (honest NOT_EXPOSED)
  - can omit assistant-turn semantics by using a different default reply style
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    model_id = "mock-divergent-model"
    mode = "no_agp"  # no_agp | match

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
                    "data": [{"id": self.model_id, "object": "model", "owned_by": "eleanity-mock"}],
                },
            )
            return
        self._json(404, {"error": {"message": f"not found: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            req = {}

        if self.path.startswith("/v1/chat/completions"):
            # Divergent content vs typical SmolLM "Hello! How can I help you today?"
            if self.mode == "no_agp":
                content = "hi from divergent mock (no matching chat template path)"
            else:
                content = "Hello! How can I help you today?"
            self._json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": req.get("model") or self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16},
                },
            )
            return

        # No tokenize / template endpoints → NOT_EXPOSED on those layers
        self._json(404, {"error": {"message": f"not found: {self.path}"}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default="mock-divergent-model")
    parser.add_argument("--mode", choices=["no_agp", "match"], default="no_agp")
    args = parser.parse_args()
    Handler.model_id = args.model_id
    Handler.mode = args.mode
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock openai server on http://{args.host}:{args.port} model={args.model_id} mode={args.mode}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
