import os
import importlib
import pytest
from fastapi import HTTPException


def reload_config_with_env(env: dict):
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        import config as cfg
        importlib.reload(cfg)
        snapshot = os.environ.copy()
        return cfg, snapshot
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_base_url_defaults_to_paper():
    cfg, env = reload_config_with_env({})
    assert cfg.APCA_API_BASE_URL.startswith("https://paper-api.alpaca.markets")
    assert env["APCA_API_BASE_URL"] == env["ALPACA_API_BASE_URL"]


def test_alias_env_unifies_bases():
    cfg, env = reload_config_with_env({"ALPACA_API_BASE_URL": "https://api.alpaca.markets"})
    assert cfg.APCA_API_BASE_URL.endswith("api.alpaca.markets")
    assert env["APCA_API_BASE_URL"] == "https://api.alpaca.markets"
    assert env["ALPACA_API_BASE_URL"] == "https://api.alpaca.markets"


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
    with pytest.raises(HTTPException) as e:
        app._enforce_ext_policy(payload)
    assert e.value.status_code == 400

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "gtc",
        "extended_hours": True,
        "order_class": None,
        "limit_price": 123.45,
    }
    with pytest.raises(HTTPException) as e:
        app._enforce_ext_policy(payload)
    assert e.value.status_code == 400

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "day",
        "extended_hours": True,
        "order_class": "bracket",
        "limit_price": 123.45,
    }
    with pytest.raises(HTTPException) as e:
        app._enforce_ext_policy(payload)
    assert e.value.status_code == 400
    os.environ.pop("ALPHA_ENFORCE_EXT", None)


def test_ttl_and_drift_gates(monkeypatch):
    os.environ["ALPHA_ENFORCE_EXT"] = "1"
    import app

    def fake_eval_stale(**_):
        return {"ok": False, "reason": "stale", "age": 99, "drift": None, "debug": {}}

    def fake_eval_drift(**_):
        return {"ok": False, "reason": "drift", "age": 1, "drift": 0.0101, "debug": {}}

    monkeypatch.setattr(app, "evaluate_limit_guard", fake_eval_stale)
    with pytest.raises(HTTPException) as e:
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
    assert e.value.status_code == 428

    monkeypatch.setattr(app, "evaluate_limit_guard", fake_eval_drift)
    with pytest.raises(HTTPException) as e:
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
    assert e.value.status_code == 409
    os.environ.pop("ALPHA_ENFORCE_EXT", None)


@pytest.mark.asyncio
async def test_request_with_retry_bubbles_429(monkeypatch):
    import app

    class FakeResp:
        def __init__(self, status, headers, body):
            self.status = status
            self.headers = headers
            self._body = body

        async def text(self):
            return self._body

        async def release(self):
            return None

    class FakeSession:
        async def request(self, method, url, **kwargs):
            return FakeResp(429, {"Retry-After": "2"}, "rate limited")

    with pytest.raises(HTTPException) as e:
        await app._request_with_retry(FakeSession(), "GET", "http://x")
    assert e.value.status_code == 429
    assert "rate limited" in str(e.value.detail).lower()


def test_uuid_fallback_when_not_required(monkeypatch):
    import app

    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
    cid = app._resolve_client_order_id(None)
    assert isinstance(cid, str) and len(cid) >= 8


def test_require_client_id_when_flag_set(monkeypatch):
    import app

    os.environ["ALPHA_REQUIRE_CLIENT_ID"] = "1"
    with pytest.raises(HTTPException) as e:
        app._resolve_client_order_id(None)
    assert e.value.status_code == 400
    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
