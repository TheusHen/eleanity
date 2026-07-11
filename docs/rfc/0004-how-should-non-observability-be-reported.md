# RFC 0004: How should non-observability be reported?

Non-observability is a property of the evidence path, not a favorable comparison result. Eleanity reports it so users can distinguish a verified match from a layer that a runtime did not expose.

## Results

- `NOT_OBSERVABLE` means a runtime cannot expose the requested layer for this execution.
- `INCOMPARABLE` means both sides produced evidence that cannot be safely compared.
- Coverage measures the fraction of required layers that were mutually and safely observed.
- A pass with incomplete coverage must retain that limitation in its status, report, or gate decision.

## Profiles

`strict` treats required missing layers as a failed gate; `quantized` may tolerate unavailable optional numerical layers; `functional` emphasizes observable behavior while reporting hidden internals; `api_conformance` permits hidden internals but requires the declared API surface.
