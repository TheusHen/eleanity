# First-divergence examples

## A. PASS — Transformers self-parity (real weights)

```bash
uv sync --extra transformers
uv run eleanity pull HuggingFaceTB/SmolLM2-135M-Instruct --tokenizer-only
uv run eleanity compare \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backends transformers,transformers \
  --format quiet --no-parallel --no-gates \
  --observe artifact,template,special_tokens,tokens,generation
```

Example quiet line:

```text
status=PASS impact=NONE coverage=100.0 confidence=0.85 first_divergence=none
run_id=0aba046a-5846-45b2-94d9-4584bb4fe98a
```

| Field | Value |
| --- | --- |
| Model | HuggingFaceTB/SmolLM2-135M-Instruct @ main |
| Stack | transformers × transformers (CPU) |
| Quantization | bf16 HF weights (not GGUF) |
| Tokens | 31 equal |

## B. DIVERGENT — missing assistant generation prompt

```bash
uv run python scripts/examples/demo_template_divergence.py
```

| Field | Value |
| --- | --- |
| Model id | org/demo-model (same on both sides) |
| Baseline | fake with `add_generation_prompt=true` → `user: Hello\nassistant:` |
| Candidate | omits assistant turn → `user: Hello` |
| Policy | strict |
| Status | **DIVERGENT** |
| First layer | **template** |
| Location | character **11**, byte **11** |
| Cause | `CHAT_TEMPLATE_DIFFERENT` (conf 0.92) |

This reproduces the class of bug often seen when one runtime applies chat templates with generation prompt and another does not.

## C. Cross-runtime Transformers × OpenAI-compat HTTP (reproducible)

Helper (starts mock server, runs compare, stops server):

```bash
uv run python scripts/examples/run_cross_runtime_demo.py
```

Live capture:

```text
status=DIVERGENT impact=HIGH coverage=50.0 confidence=0.762
first_divergence=generation
run_id=ca69f154-5c0e-4a81-8401-f81fed942ab0
```

| Side | Stack | Generation text |
| --- | --- | --- |
| baseline | transformers · SmolLM2-135M-Instruct | `Hello! How can I help you today?` |
| candidate | vllm adapter → mock OpenAI server :8765 | `hi from divergent mock (...)` |

| Layer | Baseline | Candidate | Compare |
| --- | --- | --- | --- |
| artifact | OBSERVED | OBSERVED | (soft under quantized) |
| template | OBSERVED | NOT_SUPPORTED | not mutually verified |
| tokens | OBSERVED | NOT_EXPOSED | not mutually verified |
| generation | OBSERVED | OBSERVED | **DIVERGENT** |

Mock server: `scripts/examples/mock_openai_diverge.py`  
Point the same commands at real LM Studio / vLLM serve by changing `--backend-url vllm=http://127.0.0.1:PORT`.

## D. Real LM Studio / vLLM serve (your hardware)

```bash
export ELEANITY_VLLM_URL=http://127.0.0.1:1234
uv run eleanity doctor --check-backends --backends vllm --format json
uv run eleanity compare \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backends transformers,vllm \
  --backend-url vllm=http://127.0.0.1:1234 \
  --policy quantized \
  --format text --no-gates
```

Record: server version, quant (e.g. Q8), HF revision, quiet line, verified/not verified.
