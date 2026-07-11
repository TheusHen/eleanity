# Observation Trace

The observation trace is the canonical, versioned record of what a backend could observe for one scenario run.

## Shape (serialized)

```json
{
  "trace_version": "0",
  "trace_id": "...",
  "scenario_name": "qwen-basic-multiturn",
  "backend": "transformers",
  "baseline_backend": "transformers",
  "artifact_fingerprint": { "...": "..." },
  "environment": { "python_version": "3.11.x", "packages": {} },
  "layers": {
    "artifact": { "state": "OBSERVED", "data": {}, "note": null },
    "template": { "state": "OBSERVED", "data": { "text": "..." }, "note": null }
  },
  "created_at": "2026-07-10T00:00:00+00:00",
  "duration_ms": 12.5
}
```

## Layer states

| State | Meaning |
| --- | --- |
| `OBSERVED` | Backend produced comparable evidence |
| `NOT_OBSERVABLE` | Capability missing or dependency unavailable |
| `INCOMPARABLE` | Evidence exists but cannot be compared under the policy |

Missing capabilities **must not** become false failures. Comparators treat non-`OBSERVED` pairs as `NOT_OBSERVABLE`.

## Causal order

1. artifact  
2. template (rendered prompt)  
3. special_tokens  
4. tokens  
5. logits / forward  
6. generation  
7. structured / tools  
8. streaming  
9. api  

The diagnoser stops at the first divergent comparable layer and treats later diffs as downstream consequences.

## Privacy

- Full prompts, generations, and token ID sequences live in local `result.json`.
- Shareable HTML redacts those payloads and keeps hashes, counts, fingerprints, and comparison stats.
