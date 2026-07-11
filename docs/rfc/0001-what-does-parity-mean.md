# RFC 0001: What does parity mean?

Parity means that two executions are equivalent at a declared observation layer under a declared policy. Eleanity walks causal layers: artifact, template, tokens, logits, generation, structured output, and API.

## Results

- `PASS`: exactly equal.
- `PASS_WITH_TOLERANCE`: numerical difference is within the policy tolerance.
- `DIVERGENT`: comparable evidence differs.
- `INCOMPARABLE`: both sides exist but cannot be safely compared.
- `NOT_OBSERVABLE`: a runtime does not expose the layer.

## Profiles

`strict` requires exact parity; `quantized` permits bounded numerical variation; `functional` permits bounded behavioral variation; `api_conformance` checks response and streaming contracts rather than internal tensors.
