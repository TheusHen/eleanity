"""Generate a rich mock Eleanity HTML report and print its absolute path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from eleanity.reporters.html import write_html


def layer(state: str, data: dict | None = None, note: str | None = None) -> dict:
    return {"state": state, "data": data or {}, "note": note}


def fingerprint(backend: str, quant: str | None = None, dtype: str = "bfloat16") -> dict:
    template_hash = (
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
        if backend == "transformers"
        else "aa11bb22cc33dd44ee55ff6677889900aabbccddeeff001122334455667788"
    )
    return {
        "model_ref": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "commit_sha": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_hash": "0a81234bc67890d1234567890abcdef1234567890abcdef1234567890abcdef",
        "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_hash": "49bc1234d567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "chat_template_hash": template_hash,
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
        "quantization": quant,
        "dtype": dtype,
        "gguf_metadata": {"format": "gguf", "file_type": 15} if quant == "GGUF" else {},
        "lora_adapters": [],
        "special_tokens": {
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        "runtime_version": "4.46.0" if backend == "transformers" else "0.6.3",
        "python_version": "3.11.9",
        "os": "Linux-6.8.0-x86_64",
        "cpu_arch": "x86_64",
        "gpu": "NVIDIA H100 80GB",
        "cuda_or_rocm": "12.4",
        "backend_flags": {
            "runtime": backend,
            "trust_remote_code": False,
            "device_map": "auto",
        },
    }


def main() -> Path:
    run_id = "demo-mock-ui-2026-07-10"
    now = datetime.now(UTC).isoformat()

    tpl_tf = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nExplique recursão em uma frase.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    tpl_llama = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nExplique recursão em uma frase.<|im_end|>\n"
    )

    env = {
        "python_version": "3.11.9",
        "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
        "machine": "x86_64",
        "packages": {
            "transformers": "4.46.0",
            "torch": "2.4.1",
            "vllm": "0.6.3",
            "eleanity": "1.0.0",
        },
        "cuda_available": True,
        "cuda_version": "12.4",
        "gpu_name": "NVIDIA H100 80GB",
        "torch_version": "2.4.1",
    }

    fp_tf = fingerprint("transformers")
    fp_vllm = fingerprint("vllm")
    fp_llama = fingerprint("llamacpp", quant="GGUF", dtype="q4_k_m")

    trace_tf = {
        "trace_version": "0",
        "trace_id": str(uuid4()),
        "scenario_name": "qwen-basic-multiturn",
        "backend": "transformers",
        "baseline_backend": "transformers",
        "artifact_fingerprint": fp_tf,
        "environment": env,
        "layers": {
            "artifact": layer("OBSERVED", fp_tf),
            "template": layer(
                "OBSERVED",
                {
                    "text": tpl_tf,
                    "rendered_text": tpl_tf,
                    "rendered_byte_length": len(tpl_tf.encode()),
                    "rendered_char_length": len(tpl_tf),
                    "add_generation_prompt": True,
                    "chat_template_hash": fp_tf["chat_template_hash"],
                    "template_hash": fp_tf["chat_template_hash"],
                    "roles": ["system", "user"],
                    "special_markers": ["<|im_start|>", "<|im_end|>"],
                },
            ),
            "special_tokens": layer(
                "OBSERVED",
                {
                    "bos_token_id": 151643,
                    "eos_token_id": 151645,
                    "pad_token_id": 151643,
                    "unk_token_id": None,
                    "vocab_size": 152064,
                    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
                },
            ),
            "tokens": layer(
                "OBSERVED",
                {
                    "ids": list(range(40, 90)),
                    "token_ids": list(range(40, 90)),
                    "count": 50,
                    "special_token_count": 6,
                    "special_token_positions": [0, 8, 9, 22, 23, 49],
                },
            ),
            "logits": layer(
                "OBSERVED",
                {
                    "top_ids": [785, 220, 151643, 8948, 872, 198, 2610, 525, 10950, 13],
                    "top_logits": [12.41, 9.02, 7.88, 6.11, 5.40, 4.91, 3.22, 2.10, 1.05, 0.44],
                    "device": "cuda:0",
                },
            ),
            "generation": layer(
                "OBSERVED",
                {
                    "text": "Recursão é quando uma função se chama a si mesma com um caso base.",
                    "ids": list(range(1, 13)),
                    "stop_reason": "eos",
                    "seed": 42,
                    "completion_token_count": 12,
                },
            ),
            "structured": layer("NOT_OBSERVABLE", note="not requested in this scenario"),
            "streaming": layer("NOT_OBSERVABLE", note="not requested"),
            "api": layer("NOT_OBSERVABLE", note="local transformers"),
        },
        "warnings": [],
        "errors": [],
        "created_at": now,
        "duration_ms": 1842.5,
    }

    trace_vllm = {
        "trace_version": "0",
        "trace_id": str(uuid4()),
        "scenario_name": "qwen-basic-multiturn",
        "backend": "vllm",
        "baseline_backend": "transformers",
        "artifact_fingerprint": fp_vllm,
        "environment": env,
        "layers": {
            "artifact": layer("OBSERVED", fp_vllm),
            "template": layer(
                "NOT_OBSERVABLE",
                note="vLLM does not expose apply_chat_template text through this adapter",
            ),
            "special_tokens": layer("NOT_OBSERVABLE", note="not exposed by OpenAI-compatible surface"),
            "tokens": layer(
                "OBSERVED",
                {
                    "ids": list(range(40, 88)),
                    "token_ids": list(range(40, 88)),
                    "count": 48,
                    "source": "vllm/tokenize",
                },
            ),
            "logits": layer("NOT_OBSERVABLE", note="vLLM logits not exposed"),
            "generation": layer(
                "OBSERVED",
                {
                    "text": "Recursão é uma técnica em que a função se invoca até atingir o caso base.",
                    "ids": [],
                    "stop_reason": "stop",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 48,
                        "completion_tokens": 18,
                        "total_tokens": 66,
                    },
                    "http_status": 200,
                    "latency_ms": 92.4,
                },
            ),
            "structured": layer("NOT_OBSERVABLE", note="not requested"),
            "streaming": layer("NOT_OBSERVABLE", note="not requested"),
            "api": layer(
                "OBSERVED",
                {
                    "health_ok": True,
                    "http_status": 200,
                    "has_usage": True,
                    "finish_reason": "stop",
                    "openai_shape": True,
                    "latency_ms": 92.4,
                },
            ),
        },
        "warnings": ["vllm: template NOT_OBSERVABLE — chat template string not exposed"],
        "errors": [],
        "created_at": now,
        "duration_ms": 318.2,
    }

    trace_llama = {
        "trace_version": "0",
        "trace_id": str(uuid4()),
        "scenario_name": "qwen-basic-multiturn",
        "backend": "llamacpp",
        "baseline_backend": "transformers",
        "artifact_fingerprint": fp_llama,
        "environment": env,
        "layers": {
            "artifact": layer("OBSERVED", fp_llama),
            "template": layer(
                "OBSERVED",
                {
                    "text": tpl_llama,
                    "rendered_text": tpl_llama,
                    "rendered_byte_length": len(tpl_llama.encode()),
                    "rendered_char_length": len(tpl_llama),
                    "add_generation_prompt": False,
                    "chat_template_hash": fp_llama["chat_template_hash"],
                    "template_hash": fp_llama["chat_template_hash"],
                    "roles": ["system", "user"],
                    "special_markers": ["<|im_start|>", "<|im_end|>"],
                },
            ),
            "special_tokens": layer(
                "OBSERVED",
                {
                    "bos_token_id": 151643,
                    "eos_token_id": 151645,
                    "pad_token_id": 151645,
                    "unk_token_id": None,
                    "vocab_size": 152064,
                    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
                },
            ),
            "tokens": layer(
                "OBSERVED",
                {
                    "ids": list(range(40, 82)),
                    "token_ids": list(range(40, 82)),
                    "count": 42,
                    "special_token_count": 4,
                },
            ),
            "logits": layer("NOT_OBSERVABLE", note="llama.cpp logits not exposed by this adapter"),
            "generation": layer(
                "OBSERVED",
                {
                    "text": "É chamar a mesma função com um caso de parada.",
                    "ids": [9, 8, 7, 6, 5],
                    "stop_reason": "length",
                    "finish_reason": "length",
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 5,
                        "total_tokens": 47,
                    },
                },
            ),
            "structured": layer("NOT_OBSERVABLE", note="not requested"),
            "streaming": layer("NOT_OBSERVABLE", note="not requested"),
            "api": layer(
                "OBSERVED",
                {
                    "health_ok": True,
                    "http_status": 200,
                    "has_usage": True,
                    "finish_reason": "length",
                    "openai_shape": True,
                    "latency_ms": 41.0,
                },
            ),
        },
        "warnings": ["llamacpp: add_generation_prompt likely false — missing assistant turn marker"],
        "errors": [],
        "created_at": now,
        "duration_ms": 255.1,
    }

    payload = {
        "schema_version": "1",
        "run_type": "compare",
        "run_id": run_id,
        "baseline_backend": "transformers",
        "scenario": {
            "schema_version": "0.1",
            "name": "qwen-basic-multiturn",
            "description": "Basic deterministic Qwen parity test (MOCK UI DEMO)",
            "model": {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "trust_remote_code": False,
                "dtype": "auto",
                "device_map": "auto",
            },
            "parameters": {
                "temperature": 0,
                "top_p": 1.0,
                "max_tokens": 64,
                "seed": 42,
            },
            "observe": [
                "artifact",
                "template",
                "special_tokens",
                "tokens",
                "logits",
                "generation",
                "api",
            ],
            "parity_profile": "strict",
            "tolerance": 0.0,
            "backends": ["transformers", "vllm", "llamacpp"],
        },
        "environment": env,
        "traces": [trace_tf, trace_vllm, trace_llama],
        "comparisons": {
            "vllm": {
                "artifact": {"result": "PASS", "details": {}},
                "template": {
                    "result": "NOT_OBSERVABLE",
                    "details": {"reason": "candidate N/O"},
                },
                "special_tokens": {"result": "NOT_OBSERVABLE", "details": {}},
                "tokens": {
                    "result": "DIVERGENT",
                    "details": {
                        "first_difference": 41,
                        "expected_token_id": 81,
                        "received_token_id": None,
                        "downstream_percent": 18.0,
                        "downstream_different": 2,
                        "equal_prefix": 41,
                        "left_length": 50,
                        "right_length": 48,
                    },
                },
                "logits": {"result": "NOT_OBSERVABLE", "details": {}},
                "generation": {
                    "result": "DIVERGENT",
                    "details": {
                        "first_difference": 0,
                        "downstream_percent": 100.0,
                    },
                },
                "api": {"result": "PASS", "details": {}},
            },
            "llamacpp": {
                "artifact": {
                    "result": "DIVERGENT",
                    "details": {
                        "divergent_keys": [
                            "quantization",
                            "dtype",
                            "chat_template_hash",
                        ]
                    },
                },
                "template": {
                    "result": "DIVERGENT",
                    "details": {
                        "first_difference": 88,
                        "first_byte": 88,
                        "first_character": 88,
                        "line": 5,
                        "column": 1,
                        "left_length": 112,
                        "right_length": 92,
                        "missing_assistant_turn": True,
                        "baseline_snippet": "<|im_start|>assistant\\n",
                        "candidate_snippet": "",
                    },
                },
                "special_tokens": {
                    "result": "DIVERGENT",
                    "details": {"divergent_keys": ["pad_token_id"]},
                },
                "tokens": {
                    "result": "DIVERGENT",
                    "details": {
                        "first_difference": 41,
                        "expected_token_id": 81,
                        "received_token_id": None,
                        "downstream_percent": 93.8,
                        "downstream_different": 9,
                        "equal_prefix": 41,
                    },
                },
                "logits": {"result": "NOT_OBSERVABLE", "details": {}},
                "generation": {
                    "result": "DIVERGENT",
                    "details": {
                        "first_difference": 0,
                        "downstream_percent": 100.0,
                    },
                },
                "api": {
                    "result": "DIVERGENT",
                    "details": {"issues": ["finish_reason"]},
                },
            },
        },
        "consensus": {
            "pairs": 3,
            "status": "DIVERGENT",
            "layers": {
                "template": {
                    "status": "DIVERGENT",
                    "votes": {"DIVERGENT": 1, "NOT_OBSERVABLE": 1},
                },
                "tokens": {"status": "DIVERGENT", "votes": {"DIVERGENT": 2}},
            },
        },
        "diagnosis": {
            "status": "DIVERGENT",
            "first_divergence": "template",
            "first_divergence_detail": {
                "layer": "template",
                "location": {
                    "character": 88,
                    "byte": 88,
                    "line": 5,
                    "column": 1,
                    "token_index": 41,
                },
                "baseline": "<|im_start|>assistant\\n",
                "candidate": "",
            },
            "propagation": {
                "first_token_difference": 41,
                "different_tokens_percent": 93.8,
                "downstream_different": 9,
            },
            "propagation_percent": 93.8,
            "probable_causes": [
                {
                    "code": "MISSING_ASSISTANT_TURN_TOKEN",
                    "confidence": 0.94,
                    "message": ("O backend candidato não adicionou o token de início do turno do assistente."),
                },
                {
                    "code": "ADD_GENERATION_PROMPT_DIVERGENT",
                    "confidence": 0.91,
                    "message": "add_generation_prompt differs: baseline=True candidate=False.",
                },
                {
                    "code": "QUANTIZED_VS_FULL_PRECISION",
                    "confidence": 0.88,
                    "message": "Quantization differs: baseline=None candidate='GGUF'.",
                },
            ],
            "suggested_actions": [
                "Compare add_generation_prompt between the backends.",
                "Verify the chat_template stored in tokenizer_config.json.",
                "Confirm whether GGUF conversion preserved the template.",
                "Align quantization (or switch parity policy to quantized).",
            ],
            "hypothesis": ("O backend candidato não adicionou o token de início do turno do assistente."),
            "next_test": "Compare add_generation_prompt between the backends.",
            "summary": (
                "A primeira divergência está no template de chat, no caractere 88. "
                "Depois dela, 93.8% dos tokens diferem a partir da posição 41. "
                "A provável causa é o backend não adicionou o marcador de início "
                "do turno do assistente."
            ),
            "warnings": [
                "vllm: template NOT_OBSERVABLE — chat template string not exposed",
                "llamacpp: add_generation_prompt likely false — missing assistant turn marker",
            ],
        },
        "gates": {
            "passed": False,
            "summary": "1 gate failure(s): llamacpp/template=DIVERGENT exceeds max_status=PASS",
            "results": [
                {
                    "name": "prompt-must-match",
                    "passed": False,
                    "layer": "template",
                    "status": "DIVERGENT",
                    "message": "llamacpp/template=DIVERGENT exceeds max_status=PASS",
                    "backend": "llamacpp",
                },
                {
                    "name": "generation-soft",
                    "passed": True,
                    "message": "gate generation-soft passed",
                },
            ],
        },
        "timings_ms": {
            "transformers": 1842.5,
            "vllm": 318.2,
            "llamacpp": 255.1,
        },
        "total_duration_ms": 2415.8,
        "tokenizer_only": False,
        "reproduction_command": (
            "eleanity compare --model Qwen/Qwen2.5-7B-Instruct "
            "--backends transformers,vllm,llamacpp --baseline transformers --html"
        ),
        "capabilities": {
            "transformers": {
                "render": True,
                "tokenize": True,
                "logits": True,
                "generation": True,
                "template": True,
            },
            "vllm": {
                "render": False,
                "tokenize": True,
                "logits": False,
                "generation": True,
                "streaming": True,
            },
            "llamacpp": {
                "render": False,
                "tokenize": True,
                "logits": False,
                "generation": True,
            },
        },
    }

    out_dir = Path(".eleanity/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also publish under examples/ for easy sharing
    examples = Path("examples")
    examples.mkdir(exist_ok=True)
    examples.joinpath("report_mock.json").write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
    html_path = write_html(result_path, examples / "report_mock.html")
    # keep run dir copy too
    write_html(result_path)
    return html_path.resolve()


if __name__ == "__main__":
    path = main()
    print(path)
