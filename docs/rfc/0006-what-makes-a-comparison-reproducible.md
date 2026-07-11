# RFC 0006: What makes a comparison reproducible?

A comparison is reproducible when another execution can reconstruct its declared inputs, policy, and observation plan well enough to re-evaluate the same parity claim. Reproducibility is about the claim and its evidence, not necessarily identical wall-clock timing.

## Results

- The execution capsule captures model and tokenizer identity, backend configuration, scenario, generation settings, policy, and requested layers.
- Reports preserve the selected baseline, candidate, status, coverage, first divergence, and a reproduction command.
- Seeds and deterministic settings are recorded whenever a generation claim depends on them.
- Secret values and prompt content may be redacted, but redaction must be declared so the limits of replay are clear.

## Profiles

`strict` pins all observable inputs; `quantized` also records precision and conversion details; `functional` records the behavioral oracle and allowed variation; `api_conformance` records endpoint, request contract, and streaming expectations.
