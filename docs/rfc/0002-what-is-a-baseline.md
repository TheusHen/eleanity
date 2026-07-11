# RFC 0002: What is a baseline?

A baseline is the declared reference execution for one comparison. It is evidence, not ground truth: Eleanity compares every candidate to it without claiming that the baseline is universally correct.

## Results

- A comparison names exactly one baseline unless it is explicitly a multi-baseline analysis.
- The baseline records its backend, artifact identity, scenario, policy, and observed layers.
- Changing the baseline creates a new comparison claim, even when the candidate is unchanged.
- A passing candidate is equivalent to the baseline under the declared policy, not certified correct in isolation.

## Profiles

`strict` normally uses a pinned reference artifact; `quantized` may use the unquantized execution as reference; `functional` may use an approved product behavior as reference; `api_conformance` uses the declared API contract as the reference surface.
