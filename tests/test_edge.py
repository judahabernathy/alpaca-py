import importlib
import os
from typing import Any, Dict

import pytest
from fastapi import HTTPException


def reload_config_with_env(env: Dict[str, str]) -> Any:
    original = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        import config as cfg

        importlib.reload(cfg)
    finally:
        os.environ.clear()
        os.environ.update(original)

    os.environ["APCA_API_BASE_URL"] = cfg.APCA_API_BASE_URL
    os.environ["ALPACA_API_BASE_URL"] = cfg.APCA_API_BASE_URL
    return cfg


def test_base_url_defaults_to_paper():
    cfg = reload_config_with_env({})
    assert cfg.APCA_API_BASE_URL.startswith("https://paper-api.alpaca.markets")
    assert os.environ["APCA_API_BASE_URL"] == os.environ["ALPACA_API_BASE_URL"]


def test_alias_env_unifies_bases():
    cfg = reload_config_with_env({"ALPACA_API_BASE_URL": "https://api.alpaca.markets"})
    assert cfg.APCA_API_BASE_URL.endswith("api.alpaca.markets")
    assert os.environ["APCA_API_BASE_URL"] == "https://api.alpaca.markets"
    assert os.environ["ALPACA_API_BASE_URL"] == "https://api.alpaca.markets"


def test_ext_requires_limit_day_and_forbids_bracket(monkeypatch):
    os.environ["ALPHA_ENFORCE_EXT"] = "1"
    import app

    payload = {
        "symbol": "AAPL",
        "type": "market",
        "time_in_force": "day",
        "extended_hours": True,
        "order_class": None,
        "limit_price": None,
    }
    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(payload)
    assert excinfo.value.status_code == 400

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "gtc",
        "extended_hours": True,
        "order_class": None,
        "limit_price": 123.45,
    }
    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(payload)
    assert excinfo.value.status_code == 400

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "day",
        "extended_hours": True,
        "order_class": "bracket",
        "limit_price": 123.45,
    }
    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(payload)
    assert excinfo.value.status_code == 400

    os.environ.pop("ALPHA_ENFORCE_EXT", None)


def test_ttl_and_drift_gates(monkeypatch):
    os.environ["ALPHA_ENFORCE_EXT"] = "1"
    import app

    def fake_eval_stale(**_: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "stale", "age": 99, "drift": None, "debug": {}}

    def fake_eval_drift(**_: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "drift", "age": 1, "drift": 0.0101, "debug": {}}

    monkeypatch.setattr(app, "evaluate_limit_guard", fake_eval_stale)
    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(
            {
                "symbol": "AAPL",
                "type": "limit",
                "time_in_force": "day",
                "extended_hours": False,
                "order_class": None,
                "limit_price": 123,
            }
        )
    assert excinfo.value.status_code == 428

    monkeypatch.setattr(app, "evaluate_limit_guard", fake_eval_drift)
    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(
            {
                "symbol": "AAPL",
                "type": "limit",
                "time_in_force": "day",
                "extended_hours": False,
                "order_class": None,
                "limit_price": 123,
            }
        )
    assert excinfo.value.status_code == 409

    os.environ.pop("ALPHA_ENFORCE_EXT", None)


@pytest.mark.asyncio
async def test_request_with_retry_bubbles_429(monkeypatch):
    import app

    class FakeResponse:
        def __init__(self, status: int, headers: Dict[str, str], body: str) -> None:
            self.status = status
            self.headers = headers
            self._body = body

        async def text(self) -> str:
            return self._body

        async def release(self) -> None:
            return None

    class FakeSession:
        async def request(self, method: str, url: str, **_: Any) -> FakeResponse:
            return FakeResponse(429, {"Retry-After": "2"}, "rate limited")

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(FakeSession(), "GET", "http://example.test")

    assert excinfo.value.status_code == 429
    assert "rate limited" in str(excinfo.value.detail).lower()


def test_uuid_fallback_when_not_required():
    import app

    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
    client_id = app._resolve_client_order_id(None)
    assert isinstance(client_id, str) and len(client_id) >= 8


def test_require_client_id_when_flag_set():
    import app

    os.environ["ALPHA_REQUIRE_CLIENT_ID"] = "1"
    with pytest.raises(HTTPException) as excinfo:
        app._resolve_client_order_id(None)
    assert excinfo.value.status_code == 400
    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
