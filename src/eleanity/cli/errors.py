from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EleanityError(Exception):
    """Stable, actionable CLI error with machine-readable code."""

    code: str
    message: str
    hint: str | None = None
    exit_code: int = 2

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}"]
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)


# Stable catalog — do not renumber lightly (CI may match codes).
E001_UNKNOWN_BACKEND = "ELEANITY_E001"
E002_MISSING_DEP = "ELEANITY_E002"
E003_MISSING_URL = "ELEANITY_E003"
E004_BACKEND_UNHEALTHY = "ELEANITY_E004"
E005_CONFIG = "ELEANITY_E005"
E006_SCENARIO = "ELEANITY_E006"
E007_RUN_NOT_FOUND = "ELEANITY_E007"
E008_OFFLINE = "ELEANITY_E008"
E009_GOLDEN = "ELEANITY_E009"
E010_GATE = "ELEANITY_E010"
E011_DIVERGENT = "ELEANITY_E011"
E012_INTERNAL = "ELEANITY_E012"


def missing_dep(package: str, extra: str | None = None) -> EleanityError:
    hint = f"uv sync --extra {extra}" if extra else f"pip install {package}"
    return EleanityError(
        code=E002_MISSING_DEP,
        message=f"optional dependency not installed: {package}",
        hint=hint,
        exit_code=2,
    )


def missing_url(backend: str, env_var: str) -> EleanityError:
    return EleanityError(
        code=E003_MISSING_URL,
        message=f"backend '{backend}' requires a base URL",
        hint=f"set {env_var}=http://127.0.0.1:PORT or use --backend-url {backend}=http://...",
        exit_code=2,
    )


def unknown_backend(name: str, known: list[str]) -> EleanityError:
    return EleanityError(
        code=E001_UNKNOWN_BACKEND,
        message=f"unknown backend: {name}",
        hint=f"known: {', '.join(known)}",
        exit_code=2,
    )


def run_not_found(run_id: str) -> EleanityError:
    return EleanityError(
        code=E007_RUN_NOT_FOUND,
        message=f"run not found: {run_id}",
        hint="eleanity runs ls",
        exit_code=2,
    )


def config_error(message: str, hint: str | None = None) -> EleanityError:
    return EleanityError(code=E005_CONFIG, message=message, hint=hint, exit_code=2)
