# Parity policies

Policies define what “agreement” means on numeric layers. They never convert `NOT_OBSERVABLE` into `PASS`.

| Policy | Default logits tolerance | Intent |
| --- | ---: | --- |
| `strict` | `0.0` | Exact match on comparable layers |
| `quantized` | `0.02` | Allow small numeric drift (GGUF / int8 / etc.) |
| `functional` | `0.1` | Broader numeric band for functional checks |
| `api_conformance` | `0.0` | Prefer contract/shape over logits |

YAML may use either:

```yaml
parity_policy: strict
# or
parity_profile: strict
```

Both map to the same internal `ParityProfile` enum. Override with explicit `tolerance` when needed.

## Interpretation rules

1. Baseline is a **pairwise reference**, not ground truth.
2. First causal divergence is the primary diagnosis.
3. Downstream differences are reported in the matrix but are not treated as independent root causes.
4. Quantization differences should usually appear first at `artifact` (dtype/quant flags) before logits.
