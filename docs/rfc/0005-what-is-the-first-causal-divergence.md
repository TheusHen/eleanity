# RFC 0005: What is the first causal divergence?

The first causal divergence is the earliest declared layer where comparable baseline and candidate evidence differ. It localizes the first observed break in the causal path; it does not prove the underlying root cause.

## Results

- Layers are evaluated in declared causal order, from artifact and prompt construction through generation and API behavior.
- The first divergent comparable layer is reported even when later layers also differ.
- Earlier missing or incomparable layers lower confidence in a later localization.
- Diagnosis may suggest likely causes, but preserves the distinction between evidence and inference.

## Profiles

`strict` favors the earliest exact mismatch; `quantized` skips bounded numerical variation before declaring divergence; `functional` may prioritize the first behavioral break; `api_conformance` orders contract, streaming, and response observations.
