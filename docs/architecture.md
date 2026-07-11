# Architecture (0.2)

Eleanity answers: **where did two executions of the same model first diverge, how far did that difference propagate, and what is the most likely cause?**

It is deliberately **honest**: missing telemetry is `NOT_OBSERVABLE`, never `PASS`.

## Pipeline

```
Scenario YAML
    → CompareEngine
         ├─ parallel observe(adapter_i)
         │     artifact → special_tokens → template → tokens → logits
         │     → generation → structured → streaming → api
         ├─ PolicyEngine.compare_layers (pairwise vs baseline)
         ├─ consensus_summary (all pairs)
         ├─ rule-based Diagnoser (first causal layer)
         └─ Store: result.json + junit + github annotations
    → Terminal / HTML reporters
```

## CompareEngine

- Instantiates adapters via the **registry** (builtins + entry points).
- Observes backends **in parallel** (thread pool) when safe.
- Emits a versioned result envelope (`schema_version: "1"`) including:
  - traces
  - pairwise `comparisons`
  - `consensus`
  - `capabilities`
  - `diagnosis`
  - `reproduction_command`
  - environment fingerprint

## Adapters

Contract (`BackendAdapter` ABC):

| Method | Purpose |
| --- | --- |
| `fingerprint` | Artifact + runtime identity |
| `render` | Chat template / rendered prompt |
| `tokenize` | Token IDs |
| `forward` | Logits / top-k |
| `generate` | Completion |
| `special_tokens` | Optional |
| `stream_generate` | Optional SSE observation |
| `structured` | Optional JSON/tools |
| `api_probe` | Optional HTTP contract |
| `healthcheck` | Optional liveness |

Builtins: `transformers`, `vllm`, `llamacpp`, `ollama`, `openai`, `fake`.

HTTP family shares `OpenAICompatAdapter` so vLLM / llama.cpp / Ollama / gateways stay consistent.

## PolicyEngine

| Policy | Prompt/tokens | Logits | Focus |
| --- | --- | --- | --- |
| strict | exact | tight tolerance | determinism |
| quantized | exact input | soft numeric | GGUF/int8 drift |
| functional | incomparable if only text differs | not required | JSON/tools/stop |
| api_conformance | incomparable | not required | HTTP/SSE/usage |

## Diagnoser

Deterministic rules only (no LLM):

- template markers / `add_generation_prompt`
- BOS/EOS/PAD
- tokenizer / unicode / truncation
- quantization vs full precision
- finish_reason / seed
- capability gaps

Stops at the first causal divergent **comparable** layer; later layers are downstream.

## Design invariants

1. Baseline is pairwise reference, not ground truth.  
2. Core never imports torch/vllm at module import time.  
3. Optional extras cannot break base install.  
4. Adapters never write policy logic — only observations.  
5. Reports may redact prompts while keeping hashes and metrics.  
