# Roadmap (alpha)

Eleanity **0.4.x** is **Development Status: Alpha**.

## What works today

- CLI product surface: `text` / `json` / `quiet` / `sarif`  
- Compare / test / ci / migrate / promote / vendor-check / batch  
- Policies: `strict`, `quantized`, `functional`, `api_conformance`  
- Coverage, confidence, min-coverage gates  
- Fake adapter + Transformers path (with `eleanity[transformers]`)  
- OpenAI-compatible HTTP backends (vLLM serve, LM Studio, etc.) with honest non-observability  
- Golden snapshot / check, stabilize, replay  
- Trace Spec v1 export beside `result.json`  

## Explicitly incomplete / not production claims

| Area | Status |
| --- | --- |
| PyPI published wheels | Planned (install from git until then) |
| Org home `eleanity/eleanity` | Planned transfer; currently `TheusHen/eleanity` |
| Deep logits on all HTTP backends | Partial / often NOT_EXPOSED |
| Fine-grained decode step IDs | Spec ready; adapters lag |
| Public capability certification matrix (live) | Documented static matrix; live matrix CLI is early |
| Multi-tenant SaaS | Out of scope |
| Transferable HTML product UI | Out of scope (CLI-only product) |

## Near-term

1. Publish `eleanity` 0.4.0 to PyPI as pre-release when CI is green  
2. Transfer repository to GitHub org `eleanity` when org exists  
3. Deepen Transformers + one HTTP adapter (template/tokenize paths)  
4. Expand public fixtures with known real-world divergences  
5. Make mypy a hard CI gate after typing debt is cleared  

## Non-goals

- Model quality benchmarks (MMLU, etc.)  
- Semantic free-text equivalence as PASS  
- Cloud prompt storage by default  
