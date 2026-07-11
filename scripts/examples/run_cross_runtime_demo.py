#!/usr/bin/env python3
"""End-to-end demo: Transformers (real small model) × HTTP OpenAI-compat (mock).

1) Starts mock server (subprocess)
2) Runs eleanity compare transformers,vllm
3) Prints quiet + key diagnosis fields
4) Stops server

This is the "real stack pair" story without requiring LM Studio/vLLM installed.
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


def wait_healthy(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{URL}/v1/models", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError(f"mock server not healthy at {URL}")


def main() -> int:
    env = os.environ.copy()
    env["ELEANITY_VLLM_URL"] = URL
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/examples/mock_openai_diverge.py"), "--port", str(PORT)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_healthy()
        print("=== doctor --check-backends vllm ===", flush=True)
        subprocess.run(
            ["uv", "run", "eleanity", "doctor", "--check-backends", "--backends", "vllm", "--format", "json"],
            cwd=str(ROOT),
            env=env,
            check=False,
        )
        print("\n=== compare transformers × vllm (mock) ===", flush=True)
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
        # extract run_id
        run_id = None
        for part in out.replace("\n", " ").split():
            if part.startswith("run_id="):
                run_id = part.split("=", 1)[1]
        if run_id:
            result = ROOT / ".eleanity" / "runs" / run_id / "result.json"
            if result.is_file():
                data = json.loads(result.read_text(encoding="utf-8"))
                diag = data.get("diagnosis") or {}
                print("=== diagnosis extract ===", flush=True)
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": diag.get("status"),
                            "first_divergence": diag.get("first_divergence"),
                            "confidence": diag.get("confidence"),
                            "coverage": (diag.get("coverage") or {}).get("required_coverage_percent"),
                            "verified": diag.get("verified_layers"),
                            "not_verified": diag.get("not_verified_layers"),
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
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
