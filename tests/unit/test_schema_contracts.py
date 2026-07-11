import json
from pathlib import Path

from eleanity.models.schemas import ObservationTrace, Scenario


ROOT = Path(__file__).parents[2]


def test_committed_json_schemas_match_the_pydantic_contracts():
    for filename, model in {
        "scenario.schema.json": Scenario,
        "observation_trace.schema.json": ObservationTrace,
    }.items():
        saved = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        assert saved == model.model_json_schema()
