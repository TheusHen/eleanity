# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.4.x (alpha) | Yes |
| < 0.4 | Best effort only |

## Reporting a vulnerability

Do **not** open a public issue for security problems.

Preferred options (in order):

1. **GitHub Private Vulnerability Reporting** on this repository  
   (`Security` → `Report a vulnerability`) when enabled.
2. Email the maintainers: **security@theushen.dev**  
   (fallback contact while the project is hosted under `TheusHen/eleanity`).

Include:

- Eleanity version / commit SHA  
- Reproduction steps (redacted)  
- Impact assessment  
- Whether private prompts, tokens, or credentials were involved  

We aim to acknowledge reports within **7 days**. Do not disclose publicly until a fix or coordinated disclosure date is agreed.

## Scope notes

Eleanity is **local-first**. Never attach:

- production prompts or PII  
- API keys, cookies, or auth headers  
- full unredacted traces from customer systems  

Prefer `eleanity report <run-id> --format json` with redaction enabled, or a minimized scenario YAML.
