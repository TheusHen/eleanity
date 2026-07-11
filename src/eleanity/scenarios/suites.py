from __future__ import annotations

from pathlib import Path

from eleanity.config.project import EleanityProject, SuiteRef
from eleanity.models.schemas import Scenario
from eleanity.scenarios.loader import load_scenarios


def resolve_suite_path(ref: SuiteRef | str, *, project: EleanityProject | None = None) -> Path:
    if isinstance(ref, SuiteRef):
        path = Path(ref.path)
    else:
        # lookup by name in project
        project = project or EleanityProject()
        match = next((s for s in project.suites if s.name == ref), None)
        if match is None:
            # built-in aliases
            builtins = {
                "qwen-parity": "fixtures/qwen/scenarios.yaml",
                "tokenizer-torture": "fixtures/qwen/tokenizer_torture.yaml",
                "tool-calling": "fixtures/suites/tool-calling.yaml",
                "generic-chat": "fixtures/suites/generic-chat.yaml",
            }
            if ref not in builtins:
                raise ValueError(f"unknown suite: {ref}")
            path = Path(builtins[ref])
        else:
            path = Path(match.path)
    if not path.is_file():
        raise FileNotFoundError(f"suite file not found: {path}")
    return path


def load_suite(name_or_path: str, *, project: EleanityProject | None = None) -> list[Scenario]:
    path = Path(name_or_path)
    if path.is_file():
        return load_scenarios(path)
    return load_scenarios(resolve_suite_path(name_or_path, project=project))


def list_builtin_suites() -> list[dict[str, str]]:
    return [
        {"name": "qwen-parity", "path": "fixtures/qwen/scenarios.yaml"},
        {"name": "tokenizer-torture", "path": "fixtures/qwen/tokenizer_torture.yaml"},
        {"name": "tool-calling", "path": "fixtures/suites/tool-calling.yaml"},
        {"name": "generic-chat", "path": "fixtures/suites/generic-chat.yaml"},
    ]
