# Python API

Eleanity is **CLI-first**, but ships a stable **programmatic API** so you can embed
parity checks in Python pipelines without `subprocess`.

## Install

```bash
pip install eleanity
# or with HF backend:
pip install "eleanity[transformers]"
```

## High-level client (B)

```python
from eleanity import Eleanity

client = Eleanity()  # optional: Eleanity.from_yaml("eleanity.yaml")

result = client.compare(
    model="demo",
    backends=["fake", "fake"],
    policy="strict",
    no_gates=True,
    parallel=False,
)

print(result.status, result.first_divergence, result.coverage)
print(result.quiet_line())
if not result.passed:
    raise SystemExit(result.exit_code)
# or:
result.raise_for_status()
```

### Construction

| Constructor | Meaning |
| --- | --- |
| `Eleanity()` | Defaults + env + auto-discovered `eleanity.yaml` |
| `Eleanity.from_yaml(path)` | Load project file |
| `Eleanity.configure(...)` | Same as constructor |
| `client.with_options(policy="quantized")` | Copy with overrides |

Kwargs / resolve precedence matches the CLI: **call > env > yaml > defaults**.

### Client methods

| Method | Role |
| --- | --- |
| `compare(...)` | Multi-backend parity → `CompareOutcome` |
| `test(path, ...)` | YAML file / dir / suite → `TestReport` |
| `doctor(check_backends=...)` | Env + health → `DoctorReport` |
| `report(run_id, fmt=...)` | Load run (`dict` / `json` / `text` / `sarif`) |
| `replay(run_id)` | Re-run stored compare |
| `ci(baseline, candidate, backend=...)` | Two models, one backend |
| `migrate(from_backend=..., to_backend=...)` | Same model, two runtimes |
| `promote(baseline_model=..., candidate_model=...)` | Model promotion gate |
| `vendor_check(local_backend=..., remote_backend=...)` | Local vs HTTP OpenAI-compat |
| `stabilize(backend=..., repetitions=N)` | Self-consistency protocol |
| `inspect(model, backend=...)` | Fingerprint / capabilities |
| `list_runs` / `get_run` / `diff_runs` | Run history |
| `save_golden` / `check_golden` | Golden traces |
| `certify(backend)` | Adapter certification |
| `capture(input, output)` | OpenAI JSONL → scenarios |
| `policy_spec(policy)` | Formal comparator set |
| `playbook(code)` | Remediation text |
| `suites()` | Named suite list |
| `observe` / `compare_traces` / `diagnose` | Low-level on the client |
| `engine()` | Access underlying `CompareEngine` |

### `CompareOutcome`

- `status`, `first_divergence`, `coverage`, `confidence`, `gates_passed`
- `passed` (bool), `exit_code` (0/1/2), `raise_for_status()`
- `traces`, `diagnosis`, `comparisons`, `path`, `reproduction_command`
- `to_dict()`, `quiet_line()`

## Low-level API (C)

```python
from eleanity.api import (
    make_scenario,
    observe_backend,
    compare_traces,
    diagnose_traces,
    create_adapter,
    load_scenarios,
)

scenario = make_scenario(
    model="demo",
    messages=[{"role": "user", "content": "Hello"}],
    observe=["artifact", "template", "tokens", "generation"],
    policy="strict",
)

left = observe_backend("fake", "demo", scenario)
right = observe_backend("fake", "demo", scenario)
layers = compare_traces(left, right, scenario)
diagnosis = diagnose_traces([left, right])
```

| Helper | Role |
| --- | --- |
| `make_scenario(...)` | Build `Scenario` without YAML |
| `observe(adapter, scenario, model)` | Observe constructed adapter |
| `observe_backend(name, model, scenario)` | Create + observe |
| `compare_traces(left, right, ...)` | Layer diffs |
| `compare_trace_layers` | Alias used by core |
| `diagnose_traces` | Rule diagnoser |
| `evaluate_gates` | Gate evaluation |
| `create_adapter` / `adapter_for` / `register_adapter` | Adapter registry |
| `load_scenarios` / `load_scenario_file` / `load_suite` | Scenario I/O |

## Pipeline example

```python
from eleanity import Eleanity

def parity_gate() -> int:
    client = Eleanity.from_yaml("eleanity.yaml")
    result = client.compare(
        backends=["transformers", "vllm"],
        policy="quantized",
        backend_urls={"vllm": "http://127.0.0.1:8000"},
    )
    # metrics / logging
    print(result.to_dict())
    return result.exit_code

if __name__ == "__main__":
    raise SystemExit(parity_gate())
```

## Stability

- Public symbols: `eleanity` top-level exports and `eleanity.api.*` listed in `__all__`
- Internal modules (`eleanity.core.*`, `eleanity.cli.*`) may change without notice
- Prefer `Eleanity` + `CompareOutcome` over importing `CompareEngine` directly
