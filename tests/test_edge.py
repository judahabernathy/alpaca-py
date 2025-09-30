import importlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

import edge.app as app
import edge.http as edge_http
import edge.logging as edge_logging


def test_legacy_app_module_reexports_edge_helpers():
    legacy = importlib.import_module("app")

    assert legacy.app is app.app
    assert legacy.EdgeHttpClient is app.EdgeHttpClient
    assert legacy._request_with_retry is app._request_with_retry


@pytest.mark.asyncio
async def test_lifespan_manages_http_client(monkeypatch):
    events = []

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            events.append(("init", base_url))

        async def startup(self) -> None:
            events.append(("startup",))

        async def shutdown(self) -> None:
            events.append(("shutdown",))

    monkeypatch.setattr(app, "EdgeHttpClient", FakeClient)
    application = app.app

    async with app._app_lifespan(application):
        assert isinstance(application.state.http_client, FakeClient)

    assert application.state.http_client is None
    assert events == [("init", app.API_BASE_URL), ("startup",), ("shutdown",)]


def test_get_http_client_requires_startup(monkeypatch):
    monkeypatch.setattr(app.app.state, "http_client", None, raising=False)

    with pytest.raises(HTTPException) as excinfo:
        app._get_http_client()

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_edge_http_client_request_injects_correlation(monkeypatch):
    recorded = {}
    logged = []

    class FakeResponse:
        status = 201
        headers = {"Content-Type": "application/json"}

        async def text(self) -> str:
            return '{"ok": true}'

        async def release(self) -> None:
            recorded["released"] = True

    class FakeSession:
        def __init__(self, timeout=None) -> None:
            recorded["timeout"] = timeout

        async def request(self, method, url, headers=None, **kwargs):
            recorded["request"] = (method, url, headers, kwargs)
            return FakeResponse()

        async def close(self) -> None:
            recorded["closed"] = True

    monkeypatch.setattr(edge_http.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(edge_http, "log_request", lambda method, url, status, latency_s, **fields: logged.append((method, url, status)))
    monkeypatch.setattr(edge_http, "log_error", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("log_error should not be called")))
    monkeypatch.setattr(edge_http, "get_correlation_id", lambda: "cid-123")

    client = edge_http.EdgeHttpClient("https://service")
    await client.startup()

    status, headers, body = await client.request("get", "/resource", headers={"X-Test": "1"})

    assert status == 201
    assert body == '{"ok": true}'
    assert headers["Content-Type"] == "application/json"

    method, url, logged_status = logged[0]
    assert method == "GET"
    assert url == "https://service/resource"
    assert logged_status == 201

    method_called, url_called, sent_headers, kwargs = recorded["request"]
    assert method_called == "GET"
    assert url_called == "https://service/resource"
    assert sent_headers["X-Correlation-ID"] == "cid-123"
    assert sent_headers["X-Test"] == "1"
    assert kwargs == {}

    await client.shutdown()

    assert recorded.get("closed") is True
    assert recorded.get("released") is True


@pytest.mark.asyncio
async def test_edge_http_client_logs_errors(monkeypatch):
    errors = []

    class BoomSession:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def request(self, *args, **kwargs):
            raise RuntimeError("boom")

        async def close(self) -> None:
            errors.append("closed")

    monkeypatch.setattr(edge_http.aiohttp, "ClientSession", lambda timeout=None: BoomSession())
    monkeypatch.setattr(edge_http, "log_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(edge_http, "log_error", lambda event, **fields: errors.append((event, fields)))
    monkeypatch.setattr(edge_http, "get_correlation_id", lambda: None)

    client = edge_http.EdgeHttpClient("https://service")
    await client.startup()

    with pytest.raises(RuntimeError):
        await client.request("get", "/resource")

    await client.shutdown()

    assert errors[0][0] == "http_error"


def test_logging_helpers_emit_structured_payloads(caplog):
    app.configure_logging()
    caplog.set_level("INFO")

    token = app.set_correlation_id("cid-test")
    try:
        app.log_event("test_event", foo="bar")
        app.log_error("test_error", meaning=42)
        start = time.perf_counter() - 0.01
        app.log_duration("timed", start, extra="value")
        edge_logging.log_request("GET", "https://example.test", 200, 0.123, path="/")
    finally:
        app.reset_correlation_id(token)

    messages = [json.loads(record.message) for record in caplog.records]

    assert messages[0]["event"] == "test_event"
    assert messages[0]["foo"] == "bar"
    assert messages[0]["correlation_id"] == "cid-test"

    assert messages[1]["event"] == "test_error"
    assert messages[1]["meaning"] == 42

    assert messages[2]["event"] == "timed"
    assert "latency_ms" in messages[2]

    assert messages[3]["event"] == "http_request"
    assert messages[3]["method"] == "GET"
    assert messages[3]["status"] == 200


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

    def fake_eval_stale(symbol, limit_price, side):
        raise HTTPException(status_code=428, detail={"reason": "stale"})

    def fake_eval_drift(symbol, limit_price, side):
        raise HTTPException(status_code=409, detail={"reason": "drift"})

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


    assert excinfo.value.status_code == 409

    os.environ.pop("ALPHA_ENFORCE_EXT", None)


@pytest.mark.asyncio
async def test_request_with_retry_bubbles_429(monkeypatch):
    class FakeHttpClient:
        async def request(self, method: str, path: str, headers=None, **kwargs: Any):
            return 429, {"Retry-After": "2"}, "rate limited"

    monkeypatch.setattr(app, "_alpaca_headers", lambda: {})

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(FakeHttpClient(), "GET", "/example")

    assert excinfo.value.status_code == 429
    assert "rate limited" in str(excinfo.value.detail).lower()


def test_uuid_fallback_when_not_required():

    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
    client_id = app._resolve_client_order_id(None)
    assert isinstance(client_id, str) and len(client_id) >= 8


def test_require_client_id_when_flag_set():

    os.environ["ALPHA_REQUIRE_CLIENT_ID"] = "1"
    with pytest.raises(HTTPException) as excinfo:
        app._resolve_client_order_id(None)
    assert excinfo.value.status_code == 400
    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)



def test_client_id_passthrough_without_requirement():

    os.environ.pop("ALPHA_REQUIRE_CLIENT_ID", None)
    assert app._resolve_client_order_id("custom-123") == "custom-123"



def test_client_id_passthrough_with_requirement(monkeypatch):

    monkeypatch.setenv("ALPHA_REQUIRE_CLIENT_ID", "1")
    assert app._resolve_client_order_id("user-provided") == "user-provided"



def test_evaluate_limit_guard_marks_stale(monkeypatch):
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

    with pytest.raises(HTTPException) as excinfo:
        app.evaluate_limit_guard(symbol, 101, "buy")

    assert excinfo.value.status_code == 428
    assert excinfo.value.detail["reason"] == "stale"



def test_evaluate_limit_guard_marks_drift(monkeypatch):
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
    monkeypatch.setenv("ALPHA_MAX_DRIFT_BPS", "50")
    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    with pytest.raises(HTTPException) as excinfo:
        app.evaluate_limit_guard(symbol, 101, "buy")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["reason"] == "drift"



def test_evaluate_limit_guard_allows_missing_context():
    app.evaluate_limit_guard("", None, "buy")



def test_evaluate_limit_guard_ignores_quote_errors(monkeypatch):
    class RaisingClient:
        def get_stock_latest_quote(self, request):
            raise RuntimeError("boom")

    monkeypatch.setattr(app, "md_client", lambda: RaisingClient())
    app.evaluate_limit_guard("AAPL", 1, "buy")



def test_evaluate_limit_guard_allows_valid_sell(monkeypatch):
    symbol = "TSLA"
    fresh_timestamp = datetime.now(timezone.utc)

    class FakeQuote:
        def __init__(self):
            self.timestamp = fresh_timestamp
            self.bid_price = 101
            self.ask_price = 102

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {symbol: FakeQuote()}

    monkeypatch.setenv("ALPHA_MAX_DRIFT_BPS", "150")
    monkeypatch.setenv("ALPHA_QUOTE_TTL_SECONDS", "45")
    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    app.evaluate_limit_guard(symbol, 100.5, "sell")


def test_evaluate_limit_guard_allows_valid_payload(monkeypatch):
    symbol = "AAPL"
    fresh_timestamp = datetime.now(timezone.utc)

    class FakeQuote:
        def __init__(self):
            self.timestamp = fresh_timestamp
            self.bid_price = 99
            self.ask_price = 100

    class FakeClient:
        def get_stock_latest_quote(self, request):
            return {symbol: FakeQuote()}

    monkeypatch.setenv("ALPHA_MAX_DRIFT_BPS", "200")
    monkeypatch.setenv("ALPHA_QUOTE_TTL_SECONDS", "30")
    monkeypatch.setattr(app, "md_client", lambda: FakeClient())

    # Should not raise for a limit well within drift tolerance
    app.evaluate_limit_guard(symbol, 100.1, "buy")


def test_evaluate_limit_guard_invalid_limit_price():
    with pytest.raises(HTTPException) as excinfo:
        app.evaluate_limit_guard("AAPL", "oops", "buy")

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["reason"] == "invalid_limit_price"


def test_enforce_ext_policy_rejects_advanced_order_class(monkeypatch):

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
    class FakeHttpClient:
        def __init__(self):
            self.attempts = 0

        async def request(self, method, path, headers=None, **kwargs):
            self.attempts += 1
            return 429, {}, f"rate limited {self.attempts}"

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(app, "_alpaca_headers", lambda: {})
    monkeypatch.setattr(app.asyncio, "sleep", fake_sleep)

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(FakeHttpClient(), "GET", "/example")

    assert excinfo.value.status_code == 429
    assert sleeps == [1.0, 1.0]


@pytest.mark.asyncio
async def test_request_with_retry_honors_max_attempts(monkeypatch):
    monkeypatch.setenv("ALPHA_HTTP_MAX_RETRIES", "2")

    class FakeHttpClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, path, headers=None, **kwargs):
            self.calls += 1
            return 429, {"Retry-After": "1"}, "rate limited"

    client = FakeHttpClient()
    monkeypatch.setattr(app, "_alpaca_headers", lambda: {})

    with pytest.raises(HTTPException) as excinfo:
        await app._request_with_retry(client, "GET", "/example")

    assert excinfo.value.status_code == 429
    assert client.calls == 2


@pytest.mark.asyncio
async def test_gateway_routes_and_market_data(monkeypatch):
    from types import SimpleNamespace

    class FakeHttpClient:
        def __init__(self, responses):
            self._responses = responses

        async def request(self, method, path, headers=None, **kwargs):
            expected_method, expected_path, status, content_type, body = self._responses.pop(0)
            assert expected_method == method
            assert expected_path == path
            return status, {"Content-Type": content_type}, body

    responses = [
        ("GET", "/v2/account", 200, "application/json", json.dumps({"id": "acct"})),
        ("GET", "/v2/orders", 200, "application/json", json.dumps([{ "id": "order" }])),
        ("GET", "/v2/orders/abc", 200, "application/json", json.dumps({"id": "abc"})),
        ("GET", "/v2/positions", 200, "application/json", json.dumps([])),
        ("GET", "/v2/positions/AAPL", 200, "application/json", json.dumps({"symbol": "AAPL"})),
        ("DELETE", "/v2/positions", 200, "application/json", json.dumps({"status": "all-closed"})),
        ("DELETE", "/v2/positions/AAPL", 200, "application/json", json.dumps({"status": "closed"})),
        ("GET", "/v2/watchlists", 200, "application/json", json.dumps([])),
        ("POST", "/v2/watchlists", 200, "application/json", json.dumps({"id": "wl"})),
        ("GET", "/v2/watchlists/wl", 200, "application/json", json.dumps({"id": "wl"})),
        ("PUT", "/v2/watchlists/wl", 200, "application/json", json.dumps({"id": "wl", "symbols": ["AAPL"]})),
        ("DELETE", "/v2/watchlists/wl", 204, "text/plain", ""),
    ]

    fake_client = FakeHttpClient(responses)
    monkeypatch.setattr(app, "_get_http_client", lambda: fake_client)
    monkeypatch.setenv("EDGE_API_KEY", "edge-key")

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
    assert fake_client._responses == []

@pytest.mark.asyncio
async def test_spec_serving(monkeypatch, tmp_path):

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

def test_order_payload_from_model_variants():

    model = app.CreateOrder(symbol="AAPL", side="buy", type="market", time_in_force="day")
    model_payload = app._order_payload_from_model(model)
    dict_payload = app._order_payload_from_model({"symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "day", "limit_price": None})

    assert "symbol" in model_payload
    assert "limit_price" not in dict_payload


def test_prepare_order_payload_inserts_id(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {"symbol": "AAPL"}
    result = app._prepare_order_payload(payload)

    assert result["client_order_id"] == "generated"


@pytest.mark.asyncio
async def test_submit_order_async_success(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 200, {"Content-Type": "application/json"}, json.dumps({"ok": True})

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_http_client", lambda: object())

    result = await app._submit_order_async({"symbol": "AAPL"})
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_submit_order_async_error(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 500, {"Content-Type": "text/plain"}, "boom"

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_http_client", lambda: object())

    with pytest.raises(HTTPException) as excinfo:
        await app._submit_order_async({"symbol": "AAPL"})

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_submit_order_async_invalid_json(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 200, {"Content-Type": "application/json"}, "not json"

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_http_client", lambda: object())

    with pytest.raises(HTTPException):
        await app._submit_order_async({"symbol": "AAPL"})


@pytest.mark.asyncio
async def test_order_create_endpoint(monkeypatch):
    payloads = {}

    def fake_require(header):
        payloads['auth'] = header

    def fake_payload(model):
        return {'symbol': model.symbol}

    async def fake_submit(payload):
        payloads['submitted'] = payload
        return {'ok': True}

    monkeypatch.setattr(app, "_require_gateway_key", fake_require)
    monkeypatch.setattr(app, "_order_payload_from_model", fake_payload)
    monkeypatch.setattr(app, "_submit_order_async", fake_submit)

    model = app.CreateOrder(symbol="AAPL", side="buy", type="market", time_in_force="day")
    result = await app.order_create(model, x_api_key="key")

    assert result == {'ok': True}
    assert payloads['auth'] == "key"
    assert payloads['submitted'] == {'symbol': 'AAPL'}


def test_order_create_sync_endpoint(monkeypatch):
    payloads = {}

    def fake_require(header):
        payloads['auth'] = header

    def fake_payload(model):
        return {'symbol': model.symbol}

    def fake_submit(payload):
        payloads['submitted'] = payload
        return {'ok': True}

    monkeypatch.setattr(app, "_require_gateway_key", fake_require)
    monkeypatch.setattr(app, "_order_payload_from_model", fake_payload)
    monkeypatch.setattr(app, "_submit_order_sync", fake_submit)

    model = app.CreateOrder(symbol="AAPL", side="buy", type="market", time_in_force="day")
    result = app.order_create_sync(model, x_api_key="key")

    assert result == {'ok': True}
    assert payloads['auth'] == "key"
    assert payloads['submitted'] == {'symbol': 'AAPL'}


def test_submit_order_sync_uses_event_loop(monkeypatch):
    async def fake_async(payload):
        return {"ok": True}

    monkeypatch.setattr(app, "_submit_order_async", fake_async)

    result = app._submit_order_sync({"symbol": "AAPL"})
    assert result["ok"] is True


def test_parse_timeframe_errors():

    with pytest.raises(HTTPException):
        app.parse_timeframe("bad")

    with pytest.raises(HTTPException):
        app.parse_timeframe("5Year")


def test_alpaca_headers_requires_credentials(monkeypatch):

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

    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)

    with pytest.raises(HTTPException):
        app._require_gateway_key(header_key="wrong")

    app._require_gateway_key(header_key="edge-key")





