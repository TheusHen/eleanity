# RFC 0003: When are observations comparable?

Two observations are comparable when they describe the same declared layer for the same scenario and carry enough compatible evidence for the active policy to evaluate them. Equal-looking output alone does not make different observations comparable.

## Results

- Comparable observations share a layer, scenario identity, and comparison semantics.
- Evidence includes the relevant representation: bytes for templates, IDs for tokens, numbers for logits, and contract fields for API responses.
- Different encodings, schemas, or sampling conditions are `INCOMPARABLE` until a policy defines a safe normalization.
- An absent or unsupported layer is not converted into a match.

## Profiles

`strict` accepts only canonical, exact representations; `quantized` normalizes declared numerical variation; `functional` compares declared behavioral properties; `api_conformance` compares protocol shape, ordering, and required fields.
