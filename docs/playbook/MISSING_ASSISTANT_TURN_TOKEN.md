# Missing assistant turn marker

**Code:** `MISSING_ASSISTANT_TURN_TOKEN`

Candidate did not emit the assistant-start special token in the rendered prompt.

## Suggested actions

- Compare `add_generation_prompt` flags between backends.
- Inspect `tokenizer.chat_template` / `tokenizer_config.json`.
- For GGUF, re-convert ensuring chat template metadata is preserved.

## Related files

- `tokenizer_config.json`
- `chat_template.jinja`
