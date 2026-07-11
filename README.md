# Eleanity 0.4

> **Same model. Same input. Find the first divergence.**

**CLI-first** local tool for LLM **runtime parity** diagnostics.  
It does not score model quality. It shows **where two runtimes of the same model started to disagree** (template, tokenizer, tokens, generation, API).

| For developers | For platform / CI teams |
| --- | --- |
| Before blaming the model, run `eleanity compare` and check whether template/tokenizer already diverged. | CI gate for runtime promotion and quantization: same policy, same scenario, actionable first divergence. |

## Product scope

**Shipped product surface = CLI only** (`text` / `json` / `quiet` / `sarif`).

Local-only code may exist in the tree (experimental reporters, optional modules) but is **not** part of the supported product and is not exposed as transferable HTML/shareable UI.

### Intentionally out of scope

- Quality benchmarks (MMLU) — use lm-eval  
- Generic multi-provider proxy  
- Free-text semantic equivalence as PASS  
- Transformers as an oracle  
- SaaS that uploads prompts by default  

## Install

```bash
uv sync --group dev
uv sync --extra transformers   # optional local HF backend
```

## Quick start

```bash
uv run eleanity init
uv run eleanity doctor
uv run eleanity compare --model demo --backends fake,fake --tokenizer-only --format quiet
uv run eleanity report <run-id> --format text
uv run eleanity runs ls
uv run eleanity playbook MISSING_ASSISTANT_TURN_TOKEN
```

## Core workflows

```bash
# Multi-backend parity
export ELEANITY_VLLM_URL=http://127.0.0.1:8000
uv run eleanity compare --backends transformers,vllm --tokenizer-only --format text

# Named flows
uv run eleanity migrate --from transformers --to vllm --model org/model
uv run eleanity promote --baseline full --candidate quant --backend transformers --policy quantized
uv run eleanity vendor-check --endpoint http://vendor:8000 --model id --reference transformers

# Batch / golden
uv run eleanity batch --models m1,m2 --backends fake,fake --suite generic-chat --fail-fast
uv run eleanity snapshot <run> --backend transformers
uv run eleanity check-golden <run> --golden .eleanity/golden/...

# Stability / replay
uv run eleanity stabilize --backend fake --repetitions 5 --format quiet
uv run eleanity replay <run-id> --format quiet
```

**Exit codes:** `0` pass · `1` divergent / gate fail · `2` config / error (`ELEANITY_E*`)

## Formal parity

| Topic | Doc |
| --- | --- |
| Status + comparators | [docs/parity-specification.md](docs/parity-specification.md) |
| Parity × impact | dual axis on every diagnosis |
| Execution capsule | [docs/execution-capsule.md](docs/execution-capsule.md) |
| Trace Spec v1 | [docs/trace-specification.md](docs/trace-specification.md) |
| Full CLI reference | [docs/cli.md](docs/cli.md) |
| Product evaluation | [docs/evaluation-assessment.md](docs/evaluation-assessment.md) |
| Public fixtures | `fixtures/public/` |

```bash
eleanity policy-spec --policy quantized
eleanity compare --backends fake,fake --require-self-consistency --format quiet
eleanity trace-validate .eleanity/runs/<id>/trace.v1.json
```

## CI

GitHub Actions under `.github/workflows/`:

| Workflow | Purpose |
| --- | --- |
| `ci.yml` | lint, unit, contract, integration, CLI smoke |
| `parity-local-ai.yml` | download tiny HF model + real transformers self-parity |
| `parity-public-fixtures.yml` | public fixtures on fake |
| `eleanity.yml` | reusable monorepo parity gate |
| `nightly.yml` | full pytest |

Local AI smoke:

```bash
# Linux/macOS
ELEANITY_CI_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct bash scripts/ci/run_local_ai_parity.sh

# Windows
.\scripts\ci\run_local_ai_parity.ps1
```

## CLI commands

```text
init · doctor · pull · inspect · compare · test · suites · ci
compare-endpoints · migrate · promote · vendor-check
report · runs ls|show|diff · playbook · gguf
snapshot/save-golden · check-golden · batch
stabilize · bisect · bisect-model · capture · replay
policy-spec · trace-validate · compat-matrix
check-adapter · certify · export
```

## Gates (`eleanity.yaml`)

```yaml
gates:
  - name: prompt-must-match
    layers: [template, tokens, special_tokens]
    max_status: PASS
    allow: [NOT_OBSERVABLE, NOT_EXPOSED, NOT_SUPPORTED]
```

## Adapters

`transformers` · `vllm` · `llamacpp` · `ollama` · `sglang` · `tgi` · `openai` · `fake`

## License

Apache-2.0
