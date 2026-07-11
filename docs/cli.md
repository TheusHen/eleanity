# Eleanity CLI (complete)

> Same model. Same input. Find the first divergence.

**CLI-only product.** Supported formats: `text` | `json` | `quiet` (plus `sarif` on `report`).  
There is no transferable HTML UI in the product surface.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | PASS / PASS_WITH_TOLERANCE (gates ok) |
| 1 | DIVERGENT or gate failure |
| 2 | Configuration / dependency / ERROR |

## Output formats

Most commands accept `--format text|json|quiet`.

```bash
eleanity compare --backends fake,fake --format json
eleanity compare --backends fake,fake --format quiet
# status=PASS first_divergence=none gates=True run_id=...
```

## Config precedence

**CLI flags > environment > eleanity.yaml > defaults**

| Env | Meaning |
| --- | --- |
| `ELEANITY_MODEL` | Default model |
| `ELEANITY_BACKENDS` | `a,b,c` |
| `ELEANITY_BASELINE` | Baseline backend |
| `ELEANITY_POLICY` | strict / quantized / … |
| `ELEANITY_TOKENIZER_ONLY` | 1/0 |
| `ELEANITY_REDACT_PROMPTS` | 1/0 |
| `ELEANITY_OFFLINE` | HF offline |
| `ELEANITY_PARALLEL` | 1/0 |
| `ELEANITY_VLLM_URL` etc. | Backend URLs |

```bash
eleanity init
eleanity compare                    # zero-flag: reads eleanity.yaml
eleanity compare --profile ci-tokenizer
eleanity compare --backends fake,fake --policy strict --observe template,tokens
```

## Core workflow

```bash
eleanity doctor
eleanity doctor --check-backends
eleanity pull <model> [--tokenizer-only] [--offline]
eleanity inspect <model> [--tokenizer-only]
eleanity compare [--backends a,b] [--baseline a] [--tokenizer-only] [--no-gates] \
  [--backend-url vllm=http://127.0.0.1:8000] [--format text|json|quiet]
eleanity test qwen-parity --fail-fast
eleanity ci --baseline M1 --candidate M2 --backend transformers
eleanity report <run-id> --format text    # causal diffs in the terminal
eleanity runs ls | show | diff
eleanity replay <run-id>
```

## Named product flows

```bash
eleanity migrate --from transformers --to vllm --model org/model
eleanity promote --baseline full.pt --candidate quant.gguf --backend transformers --policy quantized
eleanity vendor-check --endpoint http://vendor:8000 --model vendor-id --reference transformers --reference-model local-id
```

## Golden regression

```bash
eleanity snapshot <run-id> --backend transformers   # alias of save-golden
eleanity check-golden <run-id> --golden .eleanity/golden/....json
```

In `eleanity.yaml`:

```yaml
check_golden: false
golden_file: .eleanity/golden/baseline.json
```

## Batch

```bash
eleanity batch --models m1,m2 --backends fake,fake --suite generic-chat --fail-fast --format text
```

## Error codes

| Code | Meaning |
| --- | --- |
| `ELEANITY_E001` | Unknown backend |
| `ELEANITY_E002` | Missing optional dependency |
| `ELEANITY_E003` | Missing backend URL |
| `ELEANITY_E004` | Backend unhealthy |
| `ELEANITY_E005` | Config error |
| `ELEANITY_E007` | Run not found |
| `ELEANITY_E011` | Divergent (via exit 1) |
| `ELEANITY_E012` | Internal / unexpected |

## What not to use CLI for

- Model quality scores (MMLU)
- Semantic free-text equivalence as PASS
- Uploading prompts to SaaS by default

## Stability & determinism

```bash
eleanity stabilize --backend vllm --repetitions 10
eleanity compare --backends a,b --require-self-consistency --repetitions 5
```

## Bisect & capture

```bash
eleanity bisect --backend vllm --good v0.10.0 --bad v0.12.0 --versions v0.10.0,v0.11.0,v0.12.0
eleanity bisect-model --good-revision abc --bad-revision def --model org/model
eleanity capture openai-traces.jsonl --redact --sample 500 --output production-suite/
```

## Formal policy & trace

```bash
eleanity policy-spec --policy quantized
eleanity trace-validate .eleanity/runs/<id>/trace.v1.json
eleanity compat-matrix --model demo --backends fake,transformers
```

## Privacy

```bash
eleanity compare --redact-input --redact-output --hash-content --no-store --retention 24h
# remote sinks require --allow-remote
```

## Full command list

```text
init · doctor · pull · inspect · compare · test · ci
compare-endpoints · migrate · promote · vendor-check
report · runs ls|show|diff · suites · playbook · gguf
snapshot/save-golden · check-golden · batch
stabilize · bisect · bisect-model · capture · replay
policy-spec · trace-validate · compat-matrix
check-adapter · certify · export
```

## CI (GitHub Actions)

See [evaluation-assessment.md](evaluation-assessment.md) and workflows under `.github/workflows/`:

| Workflow | Purpose |
| --- | --- |
| `ci.yml` | lint, unit, contract, integration, CLI smoke |
| `parity-local-ai.yml` | pull tiny model + transformers self-parity + values |
| `parity-public-fixtures.yml` | fixture suites on fake |
| `eleanity.yml` | reusable gate for other repos |
| `nightly.yml` | full pytest + local AI |

Formal docs: [parity-specification.md](parity-specification.md) ·
[trace-specification.md](trace-specification.md) ·
[execution-capsule.md](execution-capsule.md)
