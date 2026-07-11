# Eleanity — full product evaluation

> Reference date: 2026-07 · stable version **1.x**
> Scope: **CLI-first** LLM runtime parity diagnostics (no transferable HTML product).

## 1. Executive summary

| Dimension | Score (0–10) | Read |
| ---: | ---: | --- |
| Problem clarity | **9.0** | Real, well-framed problem |
| Architecture / modularity | **7.5** | Extensible; uneven adapter depth |
| Correctness / honesty | **8.0** | Strong (never invents PASS); coverage still maturing |
| CLI / DX surface | **8.0** | Complete for stage; English product surface |
| Deep observability | **5.5** | Shallow HTTP; limited logits / fine layers |
| CI / engineering quality | **7.5–8** | Unit + contract + local-AI workflows |
| Enterprise readiness | **4.0** | No multi-tenant hard mode, crypto, support SLA |
| Moat / ecosystem | **4.5** | Spec + fixtures sketched; no external network yet |
| **Weighted average** | **~8.0** | **Stable CLI with explicit adapter boundaries** |

**Verdict:** Ready for platform/ML eng teams to use as a **local and CI parity gate** with small models and honest adapters. Not yet an industry standard for runtime release certification without deeper adapters.

## 2. What works well

- Formal policies and comparator tables (`docs/parity-specification.md`)
- CLI-only product: text / json / quiet / sarif
- Honest statuses: `PASS_WITH_LIMITED_COVERAGE`, `INCONCLUSIVE`, observation vs comparison
- Coverage %, confidence, min-coverage gates
- Trace Spec v1 + execution capsule
- Empirical runs: fake, transformers self-parity, LM Studio Q8 vs HF
- CI: unit, contract, fixtures, real tiny-model parity

## 3. Gaps

- Deep fine-layer observation on HTTP backends
- Public fixture network / maintainer matrix
- Enterprise retention / encryption / multi-tenant
- Bisect and capture still lightly proven

## 4. When to trust a status

| Status | CI use |
| --- | --- |
| PASS / PASS_WITH_TOLERANCE | hard fail gate OK if coverage met |
| PASS_WITH_LIMITED_COVERAGE | soft-fail / warning |
| INCONCLUSIVE | do not treat as green |
| DIVERGENT / ERROR | fail |

## 5. Conclusion

CLI-first parity tool ready for internal adoption and monorepo CI — not yet a public certification platform.
