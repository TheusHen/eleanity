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

## Optional runtime dependencies

PyTorch is installed only by the optional `transformers` extra. Eleanity uses
eager inference under `torch.inference_mode()` and does not invoke TorchScript,
`torch.jit.script`, or load serialized TorchScript programs. A unit guard scans
the shipped source to prevent accidental introduction of the affected JIT API.

Until PyTorch publishes a version identified as patched for CVE-2025-3000,
keep model/cache directories writable only by the account running Eleanity and
do not execute untrusted model code. The minimum supported PyTorch version is
kept above the specifically reported 2.6.0 release.
