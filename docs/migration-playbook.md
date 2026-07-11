# Runtime migration playbook (Fase B)

Use Eleanity when moving between Transformers ↔ vLLM ↔ SGLang ↔ llama.cpp / GGUF.

## 1. Pin the artifact

```yaml
model:
  id: org/model
  revision: <commit-sha>
  trust_remote_code: false
```

For GGUF:

```bash
uv run eleanity gguf ./model.Q4_K_M.gguf
```

Compare `chat_template_hash`, `parity_fingerprint`, architecture.

## 2. Template + tokens first (tokenizer-only)

```bash
uv run eleanity compare \
  --model org/model \
  --backends transformers,vllm \
  --tokenizer-only \
  --suite generic-chat
```

If `first_divergence=template`, fix chat template / `add_generation_prompt` before looking at sampling.

## 3. Quantized promotion

```yaml
policy: quantized
gates:
  - name: inputs
    layers: [template, tokens]
    max_status: PASS
  - name: logits-soft
    layers: [logits]
    max_status: PASS_WITH_TOLERANCE
    allow: [NOT_OBSERVABLE]
```

## 4. Tool calling / JSON

```bash
uv run eleanity test tool-calling --backends vllm,sglang
```

Policy `functional` validates JSON schema, tool names, arguments, stop reason — not free-text sameness.

## 5. Streaming contract

```yaml
observe: [generation, streaming, api]
parity_policy: api_conformance
```

Checks ordered SSE, terminal `done`, finish_reason, usage coherence.

## 6. Golden baseline

```bash
uv run eleanity save-golden <run-id> --backend transformers
uv run eleanity check-golden <new-run-id> --golden .eleanity/golden/...json
```

## 7. Multi-model batch

```bash
uv run eleanity batch \
  --models Qwen/Qwen2.5-0.5B-Instruct,demo \
  --backends fake,fake \
  --suite generic-chat \
  --tokenizer-only
```

## What not to do

- Do not declare semantic free-text equivalence as PASS.
- Do not treat Transformers as absolute truth — it is a pairwise baseline.
- Do not send prompts to SaaS by default (`export` sinks use redacted artifacts).
