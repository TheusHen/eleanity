# Scenario format (v0.1)

A **Scenario** is a complete, versioned test case. YAML fixtures under `fixtures/` and `examples/` should validate against `schemas/scenario.schema.json` and the Pydantic model `eleanity.models.schemas.Scenario`.

## Minimal example

```yaml
schema_version: "0.1"
name: qwen-basic-multiturn
description: Basic deterministic Qwen parity test
model:
  id: Qwen/Qwen2.5-0.5B-Instruct
  revision: main
  trust_remote_code: false
  dtype: auto
  device_map: auto
messages:
  - role: system
    content: You are a helpful assistant.
  - role: user
    content: Explain recursion in one sentence.
parameters:
  temperature: 0
  top_p: 1.0
  max_tokens: 64
  seed: 42
generation:
  add_generation_prompt: true
  continue_final_message: false
observe:
  - artifact
  - template
  - tokens
  - special_tokens
  - generation
parity_policy: strict
backends:
  - transformers
  - vllm
  - llamacpp
```

## Fields

| Field | Required | Notes |
| --- | --- | --- |
| `schema_version` | recommended | Currently `"0.1"` |
| `name` | yes | Stable identifier for CI |
| `description` | no | Human-readable intent |
| `model` | no | Load policy; `id` overrides CLI `--model` |
| `messages` | yes | Chat turns (`role` + `content`) |
| `parameters` | no | Sampling / generation knobs |
| `generation` | no | `add_generation_prompt`, `continue_final_message` |
| `observe` | no | Layers to capture (default: template, tokens, generation) |
| `parity_policy` / `parity_profile` | no | `strict`, `quantized`, `functional`, `api_conformance` |
| `tolerance` | no | Override numeric tolerance for logits |
| `backends` | no | Default backends when running the fixture |

## Observe layers

- `artifact` — model/tokenizer identity
- `template` / `rendered_prompt` — chat template output (alias)
- `special_tokens` — bos/eos/pad/unk map
- `tokens` — token IDs
- `logits` — top-k last-position logits when available
- `generation` / `stop_reason` — completion (+ stop reason inside generation)
- `structured`, `streaming`, `api` — post-MVP surfaces (often `NOT_OBSERVABLE`)

## Large models

For 7B+ checkpoints, set an explicit load policy:

```yaml
model:
  id: Qwen/Qwen2.5-7B-Instruct
  dtype: auto          # bf16/fp16 on CUDA, fp32 on CPU
  device_map: auto     # requires accelerate for multi-device
  attn_implementation: sdpa
  load_in_4bit: false
```

Absence of GPU or optional runtime extras must surface as `NOT_OBSERVABLE`, not as a hard crash of the compare pipeline for other backends.
