# Eleanity Trace Specification v1

The durable product is not only the CLI — it is the **trace document**.

## Location

Each compare writes:

```text
.eleanity/runs/<run_id>/result.json     # engine payload (schema_version 1)
.eleanity/runs/<run_id>/trace.v1.json   # Trace Spec product document
```

JSON Schema: `schemas/eleanity-trace-v1.schema.json`

## Shape

```json
{
  "schema_version": "1.0.0",
  "run_id": "…",
  "execution_capsule": {},
  "subjects": { "baseline": {}, "candidate": {} },
  "observations": {},
  "comparisons": {},
  "first_divergence": {},
  "propagation": [],
  "parity": { "status": "PASS" },
  "impact": { "impact": "NONE" },
  "diagnostics": [],
  "gates": [],
  "privacy": {},
  "extensions": {},
  "document_hash": "…"
}
```

## Versioning

- Semantic version on `schema_version` (`1.x.y`)
- `eleanity trace-validate <file>` validates or migrates `result.json` → v1
- `document_hash` is SHA-256 of the sealed document (excluding itself)

## Execution Capsule

See `docs/execution-capsule.md`. Capsules are sealed per subject and shared.

## Redaction

Traces support partial data: REDACTED layers, privacy manifest, secret scrubbing.
Remote upload requires `--allow-remote`.
