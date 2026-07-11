from pathlib import Path
import yaml
from eleanity.models.schemas import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = data.get("scenarios", data if isinstance(data, list) else [data])
    return [Scenario.model_validate(item) for item in items]
