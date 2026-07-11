from eleanity.comparators.api import compare_api
from eleanity.comparators.structured import compare_structured
from eleanity.models.schemas import ParityResult


def test_structured_json_parity():
    left = {"is_json": True, "parsed": {"ok": True}, "stop_reason": "stop"}
    right = {"is_json": True, "parsed": {"ok": True}, "stop_reason": "stop"}
    assert compare_structured(left, right).result == ParityResult.PASS


def test_structured_json_divergence():
    left = {"is_json": True, "parsed": {"ok": True}, "stop_reason": "stop"}
    right = {"is_json": True, "parsed": {"ok": False}, "stop_reason": "stop"}
    assert compare_structured(left, right).result == ParityResult.DIVERGENT


def test_api_contract_parity():
    left = {"http_status": 200, "finish_reason": "stop", "has_usage": True, "openai_shape": True, "health_ok": True}
    right = dict(left)
    assert compare_api(left, right).result == ParityResult.PASS


def test_api_contract_divergence_on_status():
    left = {"http_status": 200, "finish_reason": "stop", "has_usage": True, "openai_shape": True, "health_ok": True}
    right = {**left, "http_status": 500}
    assert compare_api(left, right).result == ParityResult.DIVERGENT
