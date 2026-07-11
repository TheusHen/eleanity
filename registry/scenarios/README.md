# Public scenario registry (parity-bench honest)

This directory is the seed of a **public, versioned catalog** of parity scenarios.

## Rules

1. Scenarios must be **deterministic** when claiming token/template parity (`temperature: 0`, fixed `seed`).
2. Never require proprietary prompts or private data.
3. Prefer **small models** for CI (`0.5B` class) unless the suite is marked `slow`.
4. Do **not** score model quality (no MMLU). Measure **runtime agreement**.
5. Each suite declares `parity_policy` and expected observe layers.

## Layout

```text
registry/scenarios/
  catalog.yaml          # index of suites
  generic/              # model-family agnostic chat
  qwen/                 # Qwen-focused suites (symlink/copy of fixtures)
  tools/                # tool calling / JSON schema
```

## Catalog entry

```yaml
- id: generic-chat
  path: generic/chat.yaml
  tags: [chat, ci]
  default_policy: strict
```

Consume via:

```bash
uv run eleanity test registry/scenarios/generic/chat.yaml
# or suite aliases once registered in eleanity.yaml
```
