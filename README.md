# Eleanity

**Same model. Same input. Find the first causal divergence.**

[![CI](https://github.com/TheusHen/eleanity/actions/workflows/ci.yml/badge.svg)](https://github.com/TheusHen/eleanity/actions/workflows/ci.yml)
[![Local AI parity](https://github.com/TheusHen/eleanity/actions/workflows/parity-local-ai.yml/badge.svg)](https://github.com/TheusHen/eleanity/actions/workflows/parity-local-ai.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](ROADMAP.md)

CLI-first tool for **LLM runtime parity**. It compares inference stacks on the same scenario and reports:

- **status** (`PASS`, `PASS_WITH_TOLERANCE`, `PASS_WITH_LIMITED_COVERAGE`, `INCONCLUSIVE`, `DIVERGENT`, `ERROR`)
- **first divergent layer** (when present)
- **coverage** of required layers and **diagnosis confidence**
- **functional impact** (separate from internal parity)
- a **reproduction command**

It does **not** score model quality (no MMLU). It checks whether two runtimes still share the same causal path: artifact → template → special tokens → token IDs → generation / API.

**Product surface is CLI only** (`text` / `json` / `quiet` / `sarif`).

> **Hosting note:** the repository currently lives at  
> [`TheusHen/eleanity`](https://github.com/TheusHen/eleanity) (alpha).  
> Transfer to `eleanity/eleanity` is planned when the org is available ([ROADMAP.md](ROADMAP.md)).

---

## Install (clean machine)

### From Git (available now)

```bash
# one-shot CLI without a permanent install
uvx --from git+https://github.com/TheusHen/eleanity.git eleanity --help

# or install into an environment
pip install "git+https://github.com/TheusHen/eleanity.git"
# optional HF backend (PEP 508 direct reference + extras)
pip install "eleanity[transformers] @ git+https://github.com/TheusHen/eleanity.git"
```

```bash
uv add git+https://github.com/TheusHen/eleanity.git
# or with transformers extra:
uv add "eleanity[transformers] @ git+https://github.com/TheusHen/eleanity.git"
```

### From a local checkout

```bash
git clone https://github.com/TheusHen/eleanity.git
cd eleanity
uv sync --group dev
uv run eleanity --help
```

### PyPI

After the first successful tag publish:

```bash
pip install eleanity
# pre-release / alpha pin if needed:
pip install "eleanity==0.4.0"
```

Publish via GitHub Actions (`publish.yml`):

- **Push tag** `vX.Y.Z` → version must match `pyproject.toml`
- **Actions → publish → Run workflow** on `main` (or any branch) → uses version from that tree, builds, uploads to PyPI, then creates/moves tag `v{version}` to that commit

```bash
# secret once:
gh secret set PYPI_API_TOKEN -R TheusHen/eleanity

# publish current main as the version in pyproject.toml:
gh workflow run publish.yml -R TheusHen/eleanity --ref main

# or push a tag that already matches pyproject version:
git tag v0.4.0 && git push origin v0.4.0
```

The job verifies wheel/sdist metadata, writes **SHA256SUMS**, and attaches them to the GitHub Release.

Until a release is published, use **git install** above.

Requires **Python 3.11+**.

---

## 60-second offline check

```bash
uv run eleanity compare --model demo --backends fake,fake \
  --format quiet --no-parallel --no-gates
```

Exact output from a live run:

```text
status=PASS impact=NONE coverage=100.0 confidence=0.85 first_divergence=none gates=True run_id=16c3f05b-70f9-4c6a-9170-3e5942f91bd6
```

---

## Real model self-parity (PASS)

Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (CPU).

### Commands

```bash
uv sync --group dev --extra transformers

uv run eleanity pull HuggingFaceTB/SmolLM2-135M-Instruct --tokenizer-only

uv run eleanity compare \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backends transformers,transformers \
  --format quiet --no-parallel --no-gates \
  --observe artifact,template,special_tokens,tokens,generation
```

### Exact quiet output (live run)

```text
status=PASS impact=NONE coverage=100.0 confidence=0.85 first_divergence=none gates=True run_id=0aba046a-5846-45b2-94d9-4584bb4fe98a
```

| Metric | Value |
| --- | --- |
| Engine total | ~10276 ms |
| First observation | ~9259 ms |
| Second (warm) | ~1017 ms |
| Token match | 31 / 31 |
| Required coverage | 100% (min 75%) |

```bash
uv run eleanity report 0aba046a-5846-45b2-94d9-4584bb4fe98a --format text
```

Layer table excerpt:

```text
Layer            Baseline obs   Candidate obs   Compare
artifact         OBSERVED       OBSERVED        PASS
special_tokens   OBSERVED       OBSERVED        PASS
template         OBSERVED       OBSERVED        PASS
tokens           OBSERVED       OBSERVED        PASS
generation       OBSERVED       OBSERVED        PASS

Template diff: PASS
Token diff:    PASS (count: 31)
```

This validates the **engine + Transformers path**.

---

## Cross-runtime DIVERGENT (Transformers × real HF OpenAI-compat HTTP)

**Both sides run real Hugging Face weights** (SmolLM2-135M-Instruct):

| Side | Runtime path |
| --- | --- |
| Baseline | `transformers` adapter (in-process) |
| Candidate | `vllm` adapter → local OpenAI-compat server that **loads the same HF model** via Transformers |

The HTTP server does **not** expose template/tokenize endpoints (same honesty profile as many LM Studio / gateway setups). By default it applies the chat template **without** the assistant generation prompt — a common production bug — so generation diverges for a causal reason, not a hard-coded string.

### Commands (exact)

```bash
uv sync --group dev --extra transformers

# starts real HF server, runs doctor + compare, stops server
uv run python scripts/examples/run_cross_runtime_demo.py
```

Under the hood:

```bash
# terminal A
uv run python scripts/examples/hf_openai_server.py \
  --port 8765 --preload --omit-generation-prompt \
  --model HuggingFaceTB/SmolLM2-135M-Instruct

# terminal B
export ELEANITY_VLLM_URL=http://127.0.0.1:8765
uv run eleanity doctor --check-backends --backends vllm --format json
uv run eleanity compare \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backends transformers,vllm \
  --backend-url vllm=http://127.0.0.1:8765 \
  --policy quantized \
  --format quiet --no-parallel --no-gates \
  --observe artifact,template,tokens,generation,api
```

Parity attempt (server uses full AGP): `ELEANITY_DEMO_MATCH=1 uv run python scripts/examples/run_cross_runtime_demo.py`

Point the same commands at LM Studio / vLLM serve by changing `--backend-url`. Offline fixed-string stub (CI without GPU weights): `scripts/examples/mock_openai_diverge.py`.

### Exact quiet output (live run)

```text
status=DIVERGENT impact=HIGH coverage=50.0 confidence=0.762 first_divergence=generation gates=False run_id=90028893-8848-463f-9331-daf5268f60b5
```

| Field | Value |
| --- | --- |
| Baseline | transformers · `HuggingFaceTB/SmolLM2-135M-Instruct` (HF weights, in-process) |
| Candidate | vllm adapter → `hf_openai_server.py` (same HF weights over HTTP, `--omit-generation-prompt`) |
| Policy | `quantized` |
| Status | **DIVERGENT** |
| First divergence | **generation** |
| Coverage | **50%** required layers (template/tokens not mutually observed on HTTP) |
| Verified | artifact, generation |
| Not verified | template (`NOT_SUPPORTED` on HTTP), tokens (`NOT_EXPOSED` on HTTP) |
| Generation texts | transformers: `Hello! How can I help you today?` · HTTP: `assistant\nHello! How can I help you today?` |
| Engine total | ~8287 ms |

Reproduction command stored on the run:

```text
eleanity compare --model HuggingFaceTB/SmolLM2-135M-Instruct --backends transformers,vllm \
  --baseline transformers --policy quantized \
  --observe artifact,template,tokens,generation,api --no-gates \
  --backend-url vllm=http://127.0.0.1:8765 --format text
```

This is the cross-stack claim: **two real runtimes, same model weights, localized first divergence, honest missing layers**.

---

## Template first-divergence (character-level)

Classic chat-template bug (missing assistant generation prompt), fully localized:

```bash
uv run python scripts/examples/demo_template_divergence.py
```

Exact output (live run):

```text
status:            DIVERGENT
first_divergence:  template
character:         11
byte:              11
baseline template: 'user: Hello\nassistant:'
candidate template:'user: Hello'
probable_cause:    [CHAT_TEMPLATE_DIFFERENT] conf=0.92
```

More write-ups: [docs/examples/first-divergence.md](docs/examples/first-divergence.md).

---

## Compared to common alternatives

| Approach | Result type | Causal first layer | CI exit contract | Missing-data honesty |
| --- | --- | --- | --- | --- |
| Manual print / eyeball | vibes | no | no | no |
| String golden tests | brittle text diff | rare | sometimes | often silent |
| lm-eval / quality benches | capability scores | no | yes | n/a |
| API latency dashboards | ops metrics | no | maybe | n/a |
| **Eleanity** | parity + location | **yes** | **0/1/2** | **yes (coverage)** |

---

## Adapter capability matrix (summary)

Full table: [docs/adapter-capabilities.md](docs/adapter-capabilities.md).

| Adapter | template | tokens | logits | generation | streaming | API |
| --- | --- | --- | --- | --- | --- | --- |
| fake | yes | yes | no | yes | yes | partial |
| transformers | yes | yes | partial | yes | no | no |
| vllm HTTP | partial | partial | no | yes | partial | yes |
| llamacpp HTTP | partial | partial | no | yes | partial | yes |
| ollama | no | no | no | yes | partial | yes |
| sglang / tgi / openai | partial/no | partial/no | no | yes | partial | yes |

---

## Everyday CLI

```bash
eleanity doctor --check-backends
eleanity compare --backends transformers,vllm --format text
eleanity test fixtures/public/tokenizer-edge.yaml --backends fake,fake --format quiet
eleanity report <run-id> --format text
eleanity replay <run-id>
eleanity stabilize --backend fake --repetitions 3 --format quiet
eleanity policy-spec --policy quantized
```

Exit codes: `0` pass family · `1` divergent / gate fail · `2` config / dependency error.

Full reference: [docs/cli.md](docs/cli.md)

---

## CI

| Workflow | Role |
| --- | --- |
| `ci.yml` | **Required quality:** ruff + unit/contract/integration + CLI smoke |
| `ci.yml` typecheck job | **Informational mypy** (does not fail the workflow in 0.4.x) |
| `parity-local-ai.yml` | Downloads SmolLM2-135M and runs real Transformers self-parity |
| `parity-public-fixtures.yml` | Public fixture suites |
| `publish.yml` | Build + publish to PyPI on `v*` tags (secret `PYPI_API_TOKEN`) |
| `eleanity.yml` | Reusable monorepo gate |

Local:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```

---

## Project docs

| Doc | Purpose |
| --- | --- |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/parity-specification.md](docs/parity-specification.md) | Status + comparator tables |
| [docs/adapter-capabilities.md](docs/adapter-capabilities.md) | Honesty matrix |
| [docs/trace-specification.md](docs/trace-specification.md) | Trace Spec v1 |
| [ROADMAP.md](ROADMAP.md) | Alpha boundaries |
| [SUPPORT.md](SUPPORT.md) | Where to get help |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |

---

## License

Apache License 2.0 — full text in [LICENSE](LICENSE).
