#!/usr/bin/env python3
"""End-to-end demo: Transformers (in-process) × OpenAI-compat HTTP (real HF weights).

1) Starts scripts/examples/hf_openai_server.py (loads the same HF model)
2) Runs eleanity doctor + compare transformers,vllm
3) Prints quiet line + diagnosis extract
4) Stops the server

Default mode uses --omit-generation-prompt on the HTTP server to reproduce a
real production bug class (chat template without assistant generation prompt)
while both sides still run **real** model weights — not a fixed-string mock.

Env:
  ELEANITY_DEMO_MODEL   default HuggingFaceTB/SmolLM2-135M-Instruct
  ELEANITY_DEMO_PORT    default 8765
  ELEANITY_DEMO_MATCH   if set to 1, server uses full AGP (parity attempt)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("ELEANITY_DEMO_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
PORT = int(os.environ.get("ELEANITY_DEMO_PORT", "8765"))
URL = f"http://127.0.0.1:{PORT}"
MATCH = os.environ.get("ELEANITY_DEMO_MATCH", "").strip() in {"1", "true", "yes"}


def wait_healthy(timeout: float = 300.0) -> None:
    """Wait for /v1/models (model load can take minutes on cold CPU)."""
    deadline = time.time() + timeout
    last_err = "timeout"
    while time.time() < deadline:
        try:
            r = httpx.get(f"{URL}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            last_err = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"HF OpenAI server not healthy at {URL}: {last_err}")


def main() -> int:
    env = os.environ.copy()
    env["ELEANITY_VLLM_URL"] = URL
    server_cmd = [
        sys.executable,
        str(ROOT / "scripts/examples/hf_openai_server.py"),
        "--port",
        str(PORT),
        "--model",
        MODEL,
        "--preload",
    ]
    if not MATCH:
        server_cmd.append("--omit-generation-prompt")

    print("=== starting real HF OpenAI-compat server ===", flush=True)
    print("$ " + " ".join(server_cmd), flush=True)
    server = subprocess.Popen(
        server_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # stream load progress until healthy or server dies
        deadline = time.time() + 300
        while time.time() < deadline:
            if server.poll() is not None:
                out = server.stdout.read() if server.stdout else ""
                print(out, flush=True)
                raise RuntimeError(f"server exited early with code {server.returncode}")
            try:
                r = httpx.get(f"{URL}/v1/models", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            # drain a bit of stdout for preload messages
            time.sleep(0.5)
        else:
            raise RuntimeError("server failed to become healthy within timeout")

        print("=== doctor --check-backends vllm ===", flush=True)
        subprocess.run(
            [
                "uv",
                "run",
                "eleanity",
                "doctor",
                "--check-backends",
                "--backends",
                "vllm",
                "--format",
                "json",
            ],
            cwd=str(ROOT),
            env=env,
            check=False,
        )

        mode = "match (full AGP)" if MATCH else "omit-generation-prompt (real template-class bug)"
        print(f"\n=== compare transformers × HF-HTTP ({mode}) ===", flush=True)
        cmd = [
            "uv",
            "run",
            "eleanity",
            "compare",
            "--model",
            MODEL,
            "--backends",
            "transformers,vllm",
            "--backend-url",
            f"vllm={URL}",
            "--policy",
            "quantized",
            "--format",
            "quiet",
            "--no-parallel",
            "--no-gates",
            "--observe",
            "artifact,template,tokens,generation,api",
        ]
        print("$ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out, flush=True)

        run_id = None
        for part in out.replace("\n", " ").split():
            if part.startswith("run_id="):
                run_id = part.split("=", 1)[1]
        if run_id:
            result = ROOT / ".eleanity" / "runs" / run_id / "result.json"
            if result.is_file():
                data = json.loads(result.read_text(encoding="utf-8"))
                diag = data.get("diagnosis") or {}
                traces = data.get("traces") or []
                texts: dict[str, str | None] = {}
                if isinstance(traces, list):
                    for trace in traces:
                        if not isinstance(trace, dict):
                            continue
                        name = str(trace.get("backend") or "unknown")
                        layers = trace.get("layers") or {}
                        gen = layers.get("generation") or {}
                        texts[name] = (gen.get("data") or {}).get("text")
                elif isinstance(traces, dict):
                    for name, trace in traces.items():
                        layers = (trace or {}).get("layers") or {}
                        gen = layers.get("generation") or {}
                        texts[str(name)] = (gen.get("data") or {}).get("text")
                print("=== diagnosis extract ===", flush=True)
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": diag.get("status") or data.get("parity"),
                            "first_divergence": diag.get("first_divergence"),
                            "confidence": diag.get("confidence") or data.get("confidence"),
                            "coverage": (diag.get("coverage") or data.get("coverage") or {}).get(
                                "required_coverage_percent"
                            ),
                            "verified": diag.get("verified_layers") or data.get("verified_layers"),
                            "not_verified": diag.get("not_verified_layers")
                            or data.get("not_verified_layers"),
                            "generation_texts": texts,
                            "server": {
                                "url": URL,
                                "model": MODEL,
                                "weights": "real Hugging Face Transformers",
                                "omit_generation_prompt": not MATCH,
                            },
                            "summary": diag.get("summary"),
                            "reproduction_command": data.get("reproduction_command"),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
        return proc.returncode
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
