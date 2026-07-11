# Drop Eleanity into a monorepo CI (Fase A)

**Goal:** a platform engineer can block template/tokenizer regressions by Friday.

## 1. Scaffold

```bash
uv add --dev eleanity   # or: pip install eleanity
uv run eleanity init
```

Creates:

- `eleanity.yaml` — backends, gates, suites
- `scenarios/basic.yaml`
- `.eleanity/runs`, `.eleanity/golden`

## 2. Cheap CI (no GPU)

```bash
uv run eleanity compare \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --backends transformers,fake \
  --tokenizer-only \
  --suite generic-chat
```

Tokenizer-only skips logits/generation weights.

## 3. Gates

In `eleanity.yaml`:

```yaml
gates:
  - name: prompt-must-match
    layers: [template, tokens, special_tokens]
    max_status: PASS
    allow: [NOT_OBSERVABLE]
```

Exit codes: `0` pass · `1` gate/divergence · `2` config error.

## 4. GitHub Action

```yaml
jobs:
  parity:
    uses: ./.github/workflows/eleanity.yml
    with:
      model: demo
      backends: fake,fake
      tokenizer_only: true
```

Or copy [eleanity.yml](../.github/workflows/eleanity.yml) — uploads SARIF, artifacts, PR comment.

## 5. Day-2 commands

```bash
uv run eleanity runs ls
uv run eleanity runs show <run-id>
uv run eleanity runs diff <old> <new>
uv run eleanity report <run-id> --format html
uv run eleanity playbook MISSING_ASSISTANT_TURN_TOKEN
```

## Positioning

> Before you blame the model, run `eleanity compare` and see if the template/tokenizer already diverged.
