# Execution Capsule

Even “same model + same input” is insufficient without freezing everything that
influences execution.

## Fields

```yaml
artifact:
  model_id / revision / weight_hash / tokenizer_hash / config_hash
runtime:
  name / version / commit / container_digest / library_versions
hardware:
  accelerator / driver / cuda / compute_capability
generation:
  seed / temperature / top_p / top_k / min_p / max_tokens /
  repetition_penalty / stop / logits_processors
execution:
  dtype / quantization / tensor_parallel / pipeline_parallel /
  batch_size / attention_backend / prefix_cache / speculative_decoding
privacy:
  redact_input / redact_output / hash_content / no_store / allow_remote / retention
```

Stored on every run as `execution_capsules` in `result.json` and inside
`trace.v1.json`. `capsule_hash` seals the structure for later replay comparison.

Implementation: `eleanity.spec.capsule.build_execution_capsule`.
