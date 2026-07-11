# Adapter capability matrix (honesty over marketing)

Eleanity treats missing observation as **not PASS**.  
This matrix is the public contract for what each built-in adapter can typically expose.

Legend:

| Mark | Meaning |
| --- | --- |
| **yes** | Normally `OBSERVED` when requested |
| **partial** | Sometimes observed (endpoint-dependent, optional paths, or inferred) |
| **no** | Expected `NOT_EXPOSED` / `NOT_SUPPORTED` / unavailable |

| Adapter | template | tokens | special_tokens | logits | generation | tools / structured | streaming | API contract | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **fake** | yes | yes | yes | no | yes | partial | yes | partial | Deterministic offline adapter for CI |
| **transformers** | yes | yes | yes | partial | yes | partial | no | no | Requires `eleanity[transformers]`; logits only if forward path works |
| **vllm** (HTTP) | partial | partial | no | no | yes | partial | partial | yes | Needs `ELEANITY_VLLM_URL` (vLLM serve / LM Studio / OpenAI-compat). Template/tokenize only if server exposes extras |
| **vllm** (embedded) | partial | partial | no | no | partial | no | no | no | Install `vllm` separately in a restricted runtime environment; early |
| **llamacpp** | partial | partial | no | no | yes | partial | partial | yes | HTTP OpenAI-compat; local GGUF inspect via `eleanity gguf` |
| **ollama** | no | no | no | no | yes | partial | partial | yes | OpenAI-compat `/v1` when available |
| **sglang** | partial | partial | no | no | yes | partial | partial | yes | HTTP OpenAI-compat |
| **tgi** | partial | partial | no | no | yes | partial | partial | yes | HTTP OpenAI-compat |
| **openai** | no | no | no | no | yes | partial | partial | yes | Remote OpenAI-compatible APIs |

## How to read a limited run

If Transformers×vLLM shows:

- `generation: PASS`
- `template: NOT_EXPOSED` on vLLM
- status `PASS_WITH_LIMITED_COVERAGE`

…that is **correct honesty**, not a silent skip. Use `strict` tokenizer-only gates on Transformers×Transformers for template/token CI, and `quantized` / `api_conformance` for cross-runtime HTTP comparisons.

Regenerate a live row with:

```bash
eleanity compat-matrix --model demo --backends fake,transformers --format text
```
