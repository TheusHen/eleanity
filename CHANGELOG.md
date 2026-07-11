# Changelog

## 1.0.0

### Stable release

- Stable CLI and Python API contracts
- Deterministic offline first-divergence demo
- Windows-safe terminal output for legacy encodings
- Required Ruff, mypy, unit, contract, regression, integration, and parity checks
- Idempotent protected-branch release workflow without a long-lived GitHub PAT
- Hardened optional-runtime security boundary and dependency lock

## 0.4.0

### Roadmap A+B+C foundation

**Fase A (CI-ready)** — monorepo docs (`docs/monorepo-ci.md`), gates, tokenizer-only, suites, runs, playbook HTML, GH Action (already in 0.3; hardened).

**Fase B (migration)**

- JSON Schema validation + real tool-call argument compare
- Robust streaming contract (order, terminal done, non-JSON frames, TTFT)
- Deep GGUF KV parser + parity fingerprint / chat_template hash
- Multi-model batch reports (`eleanity batch` → `batch.md`)
- Golden baseline gate (`check-golden`)

**Fase C (market standard — foundation)**

- Adapter SDK compliance (`check-adapter`, `adapters/sdk.py`)
- Runtime certification bronze/silver/gold (`certify`)
- Public scenario registry seed (`registry/scenarios/`)
- Self-hosted report server (`eleanity serve`)
- Artifact sinks: local / optional MLflow / W&B (redacted only)
- Migration playbook docs

### Non-goals (unchanged)

No quality benchmarks, no multi-provider proxy, no free-text semantic PASS, no Transformers-as-oracle, no prompt-uploading SaaS default.

## 0.3.0

### Project & CI (sections 1–4)

- `eleanity init` + **`eleanity.yaml`** (backends, profiles, suites, gates)
- **Production gates** with `max_status` / `allow` per layer
- **`--tokenizer-only`** (CI sem carregar pesos)
- **`eleanity pull`** (HF cache; tokenizer-only patterns)
- **`eleanity runs ls|show|diff`** + timings no result
- **Batch/suite runs** (`--suite`, `eleanity suites`)
- **Golden traces** (`save-golden` + compare helper)
- **SARIF** export + reusable **GitHub Action** (upload SARIF, PR comment)
- Adapters **sglang** / **tgi**
- **GGUF** shallow inspect (`eleanity gguf`)
- **Playbook** codes + HTML “o que fazer amanhã”
- Auth via **api_key_env** only (never persisted)
- Docker + compose skeleton
- Issue templates

## 0.2.0

### Production engine

- `CompareEngine` with **parallel backend observation** (`ThreadPoolExecutor`)
- Pairwise comparison matrix + **multi-backend consensus**
- `PolicyEngine` applies strict / quantized / functional / api_conformance rules per layer
- Structured logging (`event=...` lines) for CI debugging
- Run store: `result.json`, JUnit XML, GitHub annotations

### Adapters

- First-class **OpenAI-compatible HTTP adapter** (chat, tokenize, stream, structured, api_probe)
- **vLLM** and **llama.cpp** rebuilt on the OpenAI-compat base
- **Ollama** adapter (`ELEANITY_OLLAMA_URL`)
- **openai** generic adapter (`ELEANITY_OPENAI_URL` / `OPENAI_BASE_URL`)
- **Plugin registry** via `importlib.metadata` entry points (`eleanity.adapters`)
- Adapter ABC with optional `stream_generate`, `structured`, `api_probe`, `healthcheck`

### Comparators & policies

- Structured JSON / tool-call comparison
- API contract + streaming comparison
- Functional policy skips exact prompt/token equality (honest `INCOMPARABLE`)
- Quantized policy softens dtype/quantization artifact mismatches

### CLI

- `--parallel` / `--workers` on compare
- `doctor` lists registered adapters
- Version **0.2.0**

## 0.1.0

### Added

- MVP CLI: `doctor`, `inspect`, `compare`, `test`, `compare-endpoints`, `report`, `ci`
- Adapters: Transformers, vLLM (HTTP), llama.cpp (HTTP), Fake
- Scenario YAML with model load policy for large checkpoints
- Rich `PromptObservation` / `TokenObservation` payloads inside layer data
- Extended model fingerprint
- Observation traces with `warnings` and `errors`
- Comparators: byte/char/line prompt diff, token ops, special-token diagnostics
- Rule-based diagnoser with `probable_causes`, `suggested_actions`, propagation
- Result taxonomy including `ERROR`
- Terminal + JSON + HTML reports
- Qwen fixtures + tokenizer-torture suite
- GitHub Actions CI on Linux/Windows/macOS

### Notes

- vLLM and llama.cpp observations require endpoint URLs
- HTML reports redact prompts, generations, and token ID sequences
- CI exit codes: `0` pass, `1` divergent, `2` configuration/error
