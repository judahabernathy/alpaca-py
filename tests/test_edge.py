import importlib
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response


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
    return cfg


def test_base_url_defaults_to_paper():
    cfg = reload_config_with_env({})
    assert cfg.APCA_API_BASE_URL.startswith("https://paper-api.alpaca.markets")
    assert os.environ["APCA_API_BASE_URL"] == cfg.APCA_API_BASE_URL
    assert "ALPACA_API_BASE_URL" not in os.environ


def test_alias_env_unifies_bases():
    cfg = reload_config_with_env({"ALPACA_API_BASE_URL": "https://api.alpaca.markets"})
    assert cfg.APCA_API_BASE_URL.endswith("api.alpaca.markets")
    assert os.environ["APCA_API_BASE_URL"] == "https://api.alpaca.markets"
    assert "ALPACA_API_BASE_URL" not in os.environ


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



def test_client_id_passthrough_without_requirement():
    import app

    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
    assert app._resolve_client_order_id("custom-123") == "custom-123"



def test_client_id_passthrough_with_requirement(monkeypatch):
    import app

    monkeypatch.setenv("ALPHA_REQUIRE_CLIENT_ID", "1")
    assert app._resolve_client_order_id("user-provided") == "user-provided"



def test_evaluate_limit_guard_marks_stale(monkeypatch):
    import app

    symbol = "AAPL"
    stale_timestamp = datetime.now(timezone.utc) - timedelta(seconds=30)

    class FakeQuote:
        def __init__(self):
            self.timestamp = stale_timestamp
            self.bid_price = 100
            self.ask_price = 101

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {symbol: FakeQuote()}

    monkeypatch.setenv("ALPHA_QUOTE_TTL_SECONDS", "5")
    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": symbol, "limit_price": 101, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["ok"] is False
    assert result["reason"] == "stale"



def test_evaluate_limit_guard_marks_drift(monkeypatch):
    import app

    symbol = "TSLA"
    fresh_timestamp = datetime.now(timezone.utc)

    class FakeQuote:
        def __init__(self):
            self.timestamp = fresh_timestamp
            self.bid_price = 99
            self.ask_price = 100

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {symbol: FakeQuote()}

    monkeypatch.setenv("ALPHA_QUOTE_TTL_SECONDS", "60")
    monkeypatch.setenv("ALPHA_MAX_DRIFT_BPS", "50")  # 0.5%
    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": symbol, "limit_price": 101, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["ok"] is False
    assert result["reason"] == "drift"



def test_enforce_ext_policy_rejects_advanced_order_class(monkeypatch):
    import app

    monkeypatch.setenv("ALPHA_ENFORCE_EXT", "1")
    payload = {
        "symbol": "MSFT",
        "type": "limit",
        "time_in_force": "day",
        "extended_hours": True,
        "order_class": "oco",
        "limit_price": 300,
    }

    with pytest.raises(HTTPException) as excinfo:
        app._enforce_ext_policy(payload)

    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_request_with_retry_default_delay(monkeypatch):
    import app

    class FakeResponse:
        def __init__(self, attempt: int):
            self.status = 429
            self.headers = {"Retry-After": "not-a-number"}
            self._body = f"rate limited {attempt}"

        async def text(self):
            return self._body

        async def release(self):
            return None

    class FakeSession:
        def __init__(self):
            self.attempts = 0

        async def request(self, method, url, **kwargs):
            self.attempts += 1
            return FakeResponse(self.attempts)

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(app.asyncio, "sleep", fake_sleep)

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(FakeSession(), "GET", "http://example.test")

    assert excinfo.value.status_code == 429
    assert sleeps == [1.0, 1.0]


@pytest.mark.asyncio
async def test_request_with_retry_honors_max_attempts(monkeypatch):
    import app

    monkeypatch.setenv("ALPHA_HTTP_MAX_RETRIES", "2")

    class FakeResponse:
        def __init__(self):
            self.status = 429
            self.headers = {"Retry-After": "1"}
            self._body = "rate limited"

        async def text(self):
            return self._body

        async def release(self):
            return None

    class FakeSession:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            return FakeResponse()

    session = FakeSession()

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(session, "GET", "http://example.test")

    assert excinfo.value.status_code == 429
    assert session.calls == 2


@pytest.mark.asyncio
async def test_gateway_routes_and_market_data(monkeypatch):
    import app
    from types import SimpleNamespace

    class FakeResponse:
        def __init__(self, status=200, json_data=None, text=None, headers=None):
            self.status = status
            self._json = json_data if json_data is not None else {}
            self._text = text if text is not None else json.dumps(self._json)
            self.headers = headers or {}

        async def text(self):
            return self._text

        async def json(self):
            return self._json

        async def release(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, responses):
            self._responses = responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def _pop(self, method, url):
            expected_method, expected_url, response = self._responses.pop(0)
            assert expected_method == method
            assert expected_url == url
            return response

        def get(self, url, **kwargs):
            return self._pop("GET", url)

        def post(self, url, **kwargs):
            return self._pop("POST", url)

        def delete(self, url, **kwargs):
            return self._pop("DELETE", url)

        def request(self, method, url, **kwargs):
            return self._pop(method.upper(), url)

    base = app.API_BASE_URL
    responses = [
        ("GET", f"{base}/v2/account", FakeResponse(json_data={"id": "acct"})),
        ("GET", f"{base}/v2/orders", FakeResponse(json_data=[{"id": "order"}])),
        ("GET", f"{base}/v2/orders/abc", FakeResponse(json_data={"id": "abc"})),
        ("GET", f"{base}/v2/positions", FakeResponse(json_data=[])),
        ("GET", f"{base}/v2/positions/AAPL", FakeResponse(json_data={"symbol": "AAPL"})),
        ("DELETE", f"{base}/v2/positions", FakeResponse(json_data={"status": "all-closed"})),
        ("DELETE", f"{base}/v2/positions/AAPL", FakeResponse(json_data={"status": "closed"})),
        ("GET", f"{base}/v2/watchlists", FakeResponse(json_data=[])),
        ("POST", f"{base}/v2/watchlists", FakeResponse(json_data={"id": "wl"})),
        ("GET", f"{base}/v2/watchlists/wl", FakeResponse(json_data={"id": "wl"})),
        ("PUT", f"{base}/v2/watchlists/wl", FakeResponse(json_data={"id": "wl", "symbols": ["AAPL"]})),
        ("DELETE", f"{base}/v2/watchlists/wl", FakeResponse(status=204, text="")),
    ]

    session_factory = lambda **kwargs: FakeSession(responses)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)
    monkeypatch.setattr(app, "aiohttp", SimpleNamespace(ClientSession=session_factory))

    account = await app.account_get(x_api_key="edge-key")
    assert account == {"id": "acct"}

    orders = await app.orders_list(status=None, x_api_key="edge-key")
    assert orders == [{"id": "order"}]

    order = await app.orders_get_by_id("abc", x_api_key="edge-key")
    assert order["id"] == "abc"

    await app.positions_list_v2(x_api_key="edge-key")
    await app.positions_get("AAPL", x_api_key="edge-key")
    await app.positions_close_all(cancel_orders=False, x_api_key="edge-key")
    await app.positions_close("AAPL", cancel_orders=False, x_api_key="edge-key")

    watchlist_request = app.WatchlistIn(name="wl", symbols=["AAPL"])
    await app.watchlists_list_v2(x_api_key="edge-key")
    await app.watchlists_create_v2(watchlist_request, x_api_key="edge-key")
    await app.watchlists_get_v2("wl", x_api_key="edge-key")
    await app.watchlists_update_v2("wl", watchlist_request, x_api_key="edge-key")
    delete_resp = await app.watchlists_delete_v2("wl", x_api_key="edge-key")
    assert isinstance(delete_resp, Response)
    assert delete_resp.status_code == 204

    class FakeFrame:
        def __init__(self, payload):
            self._payload = payload

        def reset_index(self):
            return self

        def to_dict(self, orient):
            return self._payload

    class FakeMarketData:
        def get_stock_quotes(self, request):
            return SimpleNamespace(df=FakeFrame([{"symbol": "AAPL"}]))

        def get_stock_trades(self, request):
            return SimpleNamespace(df=FakeFrame([{"trade": 1}]))

        def get_stock_bars(self, request):
            return SimpleNamespace(df=FakeFrame([{"bar": 1}]))

    monkeypatch.setattr(app, "md_client", lambda: FakeMarketData())
    quotes = app.get_quotes_v2("AAPL", "2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00", x_api_key="edge-key")
    trades = app.get_trades_v2("AAPL", "2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00", x_api_key="edge-key")
    bars = app.get_bars_v2("AAPL", timeframe="1Day", start="2024-01-01T00:00:00", end="2024-01-02T00:00:00", x_api_key="edge-key")

    assert quotes[0]["symbol"] == "AAPL"
    assert trades[0]["trade"] == 1
    assert bars[0]["bar"] == 1
    assert responses == []


@pytest.mark.asyncio
async def test_spec_serving(monkeypatch, tmp_path):
    import app

    path_json = tmp_path / "openapi.json"
    path_json.write_text('{"openapi": "3.1.0"}')

    file_resp = app._serve_spec_file(path_json, "application/json")
    assert isinstance(file_resp, FileResponse)

    missing = tmp_path / "missing.yaml"
    json_resp = app._serve_spec_file(missing, "application/json")
    assert isinstance(json_resp, JSONResponse)

    yaml_resp = app._serve_spec_file(missing, "application/yaml")
    assert isinstance(yaml_resp, PlainTextResponse)

    # Ensure OpenAPI schema carries descriptive text
    spec = app._custom_openapi()
    desc = spec["info"]["description"]
    assert "rate" in desc
    assert "Retry-After" in desc

    assert app.well_known_openapi_json().media_type == "application/json"
    assert app.well_known_openapi_yaml().media_type == "application/yaml"
    assert app.openapi_json_alias().media_type == "application/json"
    assert app.openapi_yaml_alias().media_type == "application/yaml"

def test_evaluate_limit_guard_missing_context():
    import app

    result = app.evaluate_limit_guard(
        payload={"limit_price": None, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["skipped"] == "missing-order-context"


def test_evaluate_limit_guard_quote_error(monkeypatch):
    import app

    class RaisingClient:
        def get_stock_latest_quote(self, request):
            raise RuntimeError("boom")

    monkeypatch.setattr(app, "md_client", lambda: RaisingClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": "AAPL", "limit_price": 1, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["quote_error"] == "boom"


def test_evaluate_limit_guard_quote_missing(monkeypatch):
    import app

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {}

    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": "AAPL", "limit_price": 1, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["quote_missing"] is True


def test_evaluate_limit_guard_timestamp_missing(monkeypatch):
    import app

    class FakeQuote:
        timestamp = None
        bid_price = 1
        ask_price = 1

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {"AAPL": FakeQuote()}

    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": "AAPL", "limit_price": 1, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["timestamp_missing"] is True


def test_evaluate_limit_guard_reference_missing(monkeypatch):
    import app
    from datetime import datetime, timezone

    class FakeQuote:
        timestamp = datetime.now(timezone.utc)
        bid_price = None
        ask_price = None

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {"AAPL": FakeQuote()}

    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": "AAPL", "limit_price": 1, "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["reference_missing"] is None


def test_evaluate_limit_guard_conversion_error(monkeypatch):
    import app
    from datetime import datetime, timezone

    class FakeQuote:
        timestamp = datetime.now(timezone.utc)
        bid_price = 1
        ask_price = 1

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {"AAPL": FakeQuote()}

    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    result = app.evaluate_limit_guard(
        payload={"symbol": "AAPL", "limit_price": "oops", "side": "buy"},
        extended_hours=False,
        time_in_force="day",
        order_type="limit",
    )

    assert result["debug"]["conversion_error"] is True


def test_order_payload_from_model_variants():
    import app

    model = app.CreateOrder(symbol="AAPL", side="buy", type="market", time_in_force="day")
    model_payload = app._order_payload_from_model(model)
    dict_payload = app._order_payload_from_model({"symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "day", "limit_price": None})

    assert "symbol" in model_payload
    assert "limit_price" not in dict_payload


def test_prepare_order_payload_inserts_id(monkeypatch):
    import app

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {"symbol": "AAPL"}
    result = app._prepare_order_payload(payload)

    assert result["client_order_id"] == "generated"


@pytest.mark.asyncio
async def test_submit_order_async_success(monkeypatch):
    import app

    class FakeResponse:
        status = 200

        async def json(self):
            return {"ok": True}

        async def text(self):
            return json.dumps({"ok": True})

        async def release(self):
            return None

    async def fake_request(session, method, url, **kwargs):
        return FakeResponse()

    class DummySession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)
    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app.aiohttp, "ClientSession", lambda **kwargs: DummySession())

    result = await app._submit_order_async({"symbol": "AAPL"})
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_submit_order_async_error(monkeypatch):
    import app

    class FakeResponse:
        def __init__(self):
            self.status = 500

        async def text(self):
            return "boom"

        async def release(self):
            return None

    async def fake_request(session, method, url, **kwargs):
        return FakeResponse()

    class DummySession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)
    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app.aiohttp, "ClientSession", lambda **kwargs: DummySession())

    with pytest.raises(HTTPException) as excinfo:
        await app._submit_order_async({"symbol": "AAPL"})

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_submit_order_async_fallback_to_text(monkeypatch):
    import app
    from aiohttp import ContentTypeError

    class FakeResponse:
        status = 200

        async def json(self):
            raise ContentTypeError(request_info=None, history=None, message="no-json")

        async def text(self):
            return json.dumps({"fallback": True})

        async def release(self):
            return None

    async def fake_request(session, method, url, **kwargs):
        return FakeResponse()

    class DummySession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)
    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app.aiohttp, "ClientSession", lambda **kwargs: DummySession())

    result = await app._submit_order_async({"symbol": "AAPL"})
    assert result["fallback"] is True


def test_submit_order_sync_uses_event_loop(monkeypatch):
    import app

    async def fake_async(payload):
        return {"ok": True}

    monkeypatch.setattr(app, "_submit_order_async", fake_async)

    result = app._submit_order_sync({"symbol": "AAPL"})
    assert result["ok"] is True


def test_parse_timeframe_errors():
    import app

    with pytest.raises(HTTPException):
        app.parse_timeframe("bad")

    with pytest.raises(HTTPException):
        app.parse_timeframe("5Year")


def test_alpaca_headers_requires_credentials(monkeypatch):
    import app

    monkeypatch.setattr(app, "API_KEY_ID", None)
    monkeypatch.setattr(app, "API_SECRET_KEY", None)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    with pytest.raises(HTTPException):
        app._alpaca_headers()

    monkeypatch.setattr(app, "API_KEY_ID", "key")
    monkeypatch.setattr(app, "API_SECRET_KEY", "secret")
    headers = app._alpaca_headers()
    assert headers["APCA-API-KEY-ID"] == "key"

    monkeypatch.setattr(app, "API_KEY_ID", None)
    monkeypatch.setattr(app, "API_SECRET_KEY", None)

    with pytest.raises(HTTPException):
        app._alpaca_headers()

    monkeypatch.setattr(app, "API_KEY_ID", "key")
    monkeypatch.setattr(app, "API_SECRET_KEY", "secret")
    headers = app._alpaca_headers()
    assert headers["APCA-API-KEY-ID"] == "key"


def test_require_gateway_key(monkeypatch):
    import app

    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)

    with pytest.raises(HTTPException):
        app._require_gateway_key(header_key="wrong")

    app._require_gateway_key(header_key="edge-key")
