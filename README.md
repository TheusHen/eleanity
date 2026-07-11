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
# optional HF backend
pip install "git+https://github.com/TheusHen/eleanity.git[transformers]"
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

PyPI publish is wired (`publish.yml` on `v*` tags) but **not yet released**.  
Until `pip install eleanity` resolves on PyPI, use **git install** above. See [ROADMAP.md](ROADMAP.md).

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

This validates the **engine and Transformers path**. It is not yet a cross-runtime claim.

---

## First-divergence demo (DIVERGENT at template)

Cross-runtime HTTP pairs depend on your local servers. Independently of that, Eleanity localizes a classic failure mode: **one side omits the assistant generation prompt**.

```bash
uv run python scripts/examples/demo_template_divergence.py
```

Exact output (live run):

```text
=== Eleanity first-divergence demo ===
model:     org/demo-model
baseline:  fake (add_generation_prompt=true)
candidate: candidate-no-agp (omits assistant turn)
policy:    strict

status:            DIVERGENT
first_divergence:  template
character:         11
byte:              11
baseline_snippet:  "
assistant:"
candidate_snippet: ""

baseline template:
'user: Hello\nassistant:'
candidate template:
'user: Hello'

template comparison: DIVERGENT
first_character:     11
first_byte:          11

probable_cause: [CHAT_TEMPLATE_DIFFERENT] conf=0.92
  Chat template hash differs between backends.

summary: First divergence is in the chat template at character 11. After that,
100.0% of tokens differ from index 11. Likely cause: Chat template hash differs
between backends.
```

That is the core product claim: **layer + character index + snippets + cause code**, not only “outputs differ”.

### Cross-runtime (Transformers × OpenAI-compatible server)

When a server is available (vLLM serve, LM Studio, etc.):

```bash
export ELEANITY_VLLM_URL=http://127.0.0.1:1234   # no trailing /v1
uv run eleanity doctor --check-backends --backends vllm --format json

# Same logical model, two stack IDs (example names)
# transformers: HuggingFaceTB/SmolLM2-135M-Instruct
# server:       huggingfacetb.smollm2-135m-instruct
uv run eleanity compare --config eleanity.yaml \
  --backends transformers,vllm \
  --backend-url vllm=http://127.0.0.1:1234 \
  --policy quantized \
  --format text --no-gates
```

Expect honest partial observability on HTTP (template/tokens often `NOT_EXPOSED`).  
A green generation with missing template layers should surface as **`PASS_WITH_LIMITED_COVERAGE`**, not a silent full PASS. See [docs/adapter-capabilities.md](docs/adapter-capabilities.md).

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
| `publish.yml` | Build/publish on `v*` tags when PyPI credentials exist |
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
