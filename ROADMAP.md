# Roadmap

Eleanity **1.x** is the stable CLI and Python API line.

Hosted at **[TheusHen/eleanity](https://github.com/TheusHen/eleanity)** (public).  
Org transfer to `eleanity/eleanity` stays deferred until the GitHub org exists.

## What works today (shipped)

- CLI product surface: `text` / `json` / `quiet` / `sarif`
- Compare / test / ci / migrate / promote / vendor-check / batch
- Policies: `strict`, `quantized`, `functional`, `api_conformance`
- Coverage, confidence, min-coverage gates
- Fake adapter + Transformers path (`eleanity[transformers]`)
- OpenAI-compatible HTTP backends (vLLM serve, LM Studio, mock) with honest non-observability
- Golden snapshot / check, stabilize, replay
- Trace Spec v1 export beside `result.json`
- Full Apache-2.0 `LICENSE` + `NOTICE`
- Community files: `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, issue/PR templates
- Adapter capability matrix: [docs/adapter-capabilities.md](docs/adapter-capabilities.md)
- Live proofs in README:
  - Transformers self-parity **PASS** (SmolLM2-135M)
  - Template first-divergence **DIVERGENT** (`demo_template_divergence.py`)
  - Cross-runtime transformers × HTTP **DIVERGENT** (`run_cross_runtime_demo.py`)
- CI: required checks on `main` (ruff, unit, contract, CLI smoke)
- GitHub: public repo, discussions, branch protection, secret scanning, private vulnerability reporting
- Install path: `pip install git+https://github.com/TheusHen/eleanity.git` (and `uvx --from git+…`)
- Wheel/sdist build; `release.yml` publishes reviewed versions from `main`, creates immutable tags, and attaches SHA256SUMS + artifacts to GitHub Releases (`PYPI_API_TOKEN`)

## Explicitly incomplete

| Area | Status |
| --- | --- |
| PyPI published name `eleanity` | Published; automated releases require the `PYPI_API_TOKEN` repository secret |
| Org home `eleanity/eleanity` | Deferred; stay on `TheusHen/eleanity` |
| Deep logits on all HTTP backends | Partial / often NOT_EXPOSED |
| Fine-grained decode step IDs | Spec ready; adapters lag |
| Live capability certification matrix | Static matrix shipped; live CLI matrix early |
| Multi-tenant SaaS | Out of scope |
| Transferable HTML product UI | Out of scope (CLI-only) |

## Near-term

1. Transfer repository to GitHub org `eleanity` when the org exists
2. Deepen Transformers + one HTTP adapter (template/tokenize paths)
3. Expand public fixtures with known real-world divergences
4. Publish a live capability certification matrix

## Non-goals

- Model quality benchmarks (MMLU, etc.)
- Semantic free-text equivalence as PASS
- Cloud prompt storage by default
