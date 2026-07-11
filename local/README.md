# Local-only experiments

Anything under `local/` is **not** part of the shipped CLI product.

This directory is for machine-local experiments (optional HTML mockups, private notes, etc.).  
It is gitignored by default for private workstations; do not treat it as a product surface.

Supported product I/O:

```bash
eleanity compare --format text|json|quiet
eleanity report <run-id> --format text|json|sarif
```
