# Support

Eleanity is a stable open-source CLI project. Support is community- and maintainer-best-effort.

## Where to ask

| Topic | Where |
| --- | --- |
| Bug / incorrect PASS or DIVERGENT | [GitHub Issues](https://github.com/TheusHen/eleanity/issues) — use the **Parity bug** form |
| New adapter / feature | [GitHub Issues](https://github.com/TheusHen/eleanity/issues) — **Feature / adapter** form |
| Security | [SECURITY.md](SECURITY.md) only |
| Usage questions | GitHub Issues with `question` label |

## Before opening an issue

1. Run `uv run eleanity doctor --format json`  
2. Capture a minimized command that reproduces the problem  
3. Redact prompts and secrets  
4. Include model id, revision, backend names, and policy  

## Support boundaries

- SLA response times  
- Compatibility with every remote server quirk  
- Deep logits/template exposure for HTTP-only backends  

See [ROADMAP.md](ROADMAP.md) for intentional limitations.
