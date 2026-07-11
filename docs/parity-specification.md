# Eleanity Formal Parity Specification

> Same model. Same input. **Defined** first divergence.

This document is the operational definition of every parity status and of every
policy’s per-layer comparator. `PASS_WITH_TOLERANCE` is never free-form.

## Status taxonomy

| Status | Meaning | When | Never |
| --- | --- | --- | --- |
| **PASS** | Required observed layers match under their comparator modes | exact equal; numerical zero drift; prefix full match | Missing data |
| **PASS_WITH_TOLERANCE** | Drift stays inside declared thresholds | numerical/prefix/topk with atol/rtol/agreement | exact-mode layers |
| **DIVERGENT** | Required observed layer fails its comparator | exact mismatch; drift beyond thresholds | When either side not OBSERVED |
| **INCONCLUSIVE** | Cannot decide | NOT_EXPOSED / INFERRED / REDACTED / self-inconsistent backend | Treated as equality |
| **UNSUPPORTED** | Adapter cannot expose a required layer | capability false | Silent skip as PASS |
| **ERROR** | Execution/observation failed | exceptions, FAILED state | Soft policy failures |

Legacy engine values still emitted for compatibility:

| Legacy | Formal |
| --- | --- |
| `INCOMPARABLE` | `INCONCLUSIVE` / `UNSUPPORTED` |
| `NOT_OBSERVABLE` | `INCONCLUSIVE` |

CLI: `eleanity policy-spec --policy quantized`

## Policy comparator tables

### `strict`

```yaml
policy: strict
comparators:
  chat_template: { mode: exact }
  rendered_prompt: { mode: exact }
  special_tokens: { mode: exact }
  input_token_ids: { mode: exact }
  prefill_logits: { mode: numerical, atol: 1.0e-5, rtol: 1.0e-5, top_k_agreement: 1.0, required: false }
  generated_token_ids: { mode: exact }
  finish_reason: { mode: exact }
```

### `quantized`

```yaml
policy: quantized
comparators:
  chat_template: { mode: exact }
  rendered_prompt: { mode: exact }
  input_token_ids: { mode: exact }
  prefill_logits:
    mode: numerical
    atol: 1.0e-4
    rtol: 1.0e-3
    top_k: 10
    top_k_agreement: 0.99
  generated_token_ids:
    mode: prefix
    exact_prefix_tokens: 16
  finish_reason: { mode: exact }
```

### `functional`

Structural/tool/stop parity only — template/logits/token ids may be `ignore`.

### `api_conformance`

HTTP shape, usage, finish_reason, streaming order — not weight-level logits.

## Dual axis: Parity × Impact

Every diagnosis also carries:

```text
Parity: DIVERGENT
Impact: NONE | LOW | MEDIUM | HIGH | CATASTROPHIC
```

Example: prefill logits diverge but final token sequence matches →  
`Parity=DIVERGENT`, `Impact=NONE`.

## Honesty rule

> Never treat NOT_EXPOSED, UNSUPPORTED, REDACTED, INFERRED, or FAILED as equality.

Implementation: `eleanity.spec.parity`, `eleanity.spec.observability`, `eleanity.spec.impact`.
