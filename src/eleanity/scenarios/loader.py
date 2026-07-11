from __future__ import annotations

from pathlib import Path

import yaml

from eleanity.models.schemas import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load one or more scenarios from a YAML file.

    Supports:
    - single document: one scenario mapping
    - single document: ``{ scenarios: [ ... ] }`` or a list
    - multi-document stream separated by ``---`` (public fixtures)
    """

    text = Path(path).read_text(encoding="utf-8")
    documents = list(yaml.safe_load_all(text))
    items: list[object] = []
    for data in documents:
        if data is None:
            continue
        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict) and "scenarios" in data:
            nested = data.get("scenarios") or []
            if isinstance(nested, list):
                items.extend(nested)
            else:
                items.append(nested)
        else:
            items.append(data)
    if not items:
        raise ValueError(f"no scenarios found in {path}")
    return [Scenario.model_validate(item) for item in items]
