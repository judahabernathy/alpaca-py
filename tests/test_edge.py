import importlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request
from urllib.parse import urlencode

import edge.app as app
import edge.http as edge_http
import edge.logging as edge_logging
import edge.extras as extras




def build_request(
    path: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
) -> Request:
    query_string = urlencode(query or {}, doseq=True)
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string.encode("utf-8"),
        "headers": [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in (headers or {}).items()
        ],
        "server": ("testserver", 80),
        "scheme": "http",
    }

    async def receive():
        return {"type": "http.request"}

    return Request(scope, receive)


def gateway_request(path: str, *, method: str = "GET", query: Optional[Dict[str, Any]] = None) -> Request:
    return build_request(path, method=method, headers={"X-API-Key": "edge-key"}, query=query)


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
        assert isinstance(application.state.trading_http_client, FakeClient)
        assert isinstance(application.state.data_http_client, FakeClient)

    assert application.state.http_client is None
    assert getattr(application.state, "trading_http_client", None) is None
    assert getattr(application.state, "data_http_client", None) is None
    assert events == [
        ("init", app.API_BASE_URL),
        ("init", app.DATA_BASE_URL),
        ("startup",),
        ("startup",),
        ("shutdown",),
        ("shutdown",),
    ]


def test_get_http_client_requires_startup(monkeypatch):
    monkeypatch.setattr(app.app.state, "http_client", None, raising=False)
    monkeypatch.setattr(app.app.state, "trading_http_client", None, raising=False)

    with pytest.raises(HTTPException) as excinfo:
        app._get_http_client()

    assert excinfo.value.status_code == 500


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
    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

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
    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

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
            expected_method, expected_path, status, content_type, body, expected_params = self._responses.pop(0)
            assert expected_method == method
            assert expected_path == path
            if expected_params is not None:
                assert kwargs.get("params") == expected_params
            return status, {"Content-Type": content_type}, body

    responses = [
        ("GET", "/v2/account", 200, "application/json", json.dumps({"id": "acct"}), None),
        ("GET", "/v2/orders", 200, "application/json", json.dumps([{ "id": "order" }]), None),
        (
            "GET",
            "/v2/orders",
            200,
            "application/json",
            json.dumps([{ "id": "filtered" }]),
            {"status": "open", "symbols": "AAPL,MSFT"},
        ),
        ("GET", "/v2/orders/abc", 200, "application/json", json.dumps({"id": "abc"}), None),
        ("GET", "/v2/positions", 200, "application/json", json.dumps([]), None),
        ("GET", "/v2/positions/AAPL", 200, "application/json", json.dumps({"symbol": "AAPL"}), None),
        ("DELETE", "/v2/positions", 200, "application/json", json.dumps({"status": "all-closed"}), {"cancel_orders": "false"}),
        ("DELETE", "/v2/positions/AAPL", 200, "application/json", json.dumps({"status": "closed"}), {"cancel_orders": "false"}),
        ("GET", "/v2/watchlists", 200, "application/json", json.dumps([]), None),
        ("POST", "/v2/watchlists", 200, "application/json", json.dumps({"id": "wl"}), None),
        ("GET", "/v2/watchlists/wl", 200, "application/json", json.dumps({"id": "wl"}), None),
        ("PUT", "/v2/watchlists/wl", 200, "application/json", json.dumps({"id": "wl", "symbols": ["AAPL"]}), None),
        ("DELETE", "/v2/watchlists/wl", 204, "text/plain", "", None),
        ("GET", "/v2/account/activities", 200, "application/json", json.dumps([]), {"direction": "desc"}),
    ]

    fake_client = FakeHttpClient(responses)
    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.delenv("X_API_KEY", raising=False)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)
    monkeypatch.setattr(app, "_get_trading_http_client", lambda: fake_client)
    monkeypatch.setattr(app, "_get_data_http_client", lambda: fake_client)
    monkeypatch.setattr(app, "_http_client_for_path", lambda path: fake_client)

    account = await app.account_get(gateway_request("/v2/account"))
    assert account == {"id": "acct"}

    orders = await app.orders_list(gateway_request('/v2/orders'))
    orders_payload = json.loads(orders.body.decode())
    assert orders_payload == [{'id': 'order'}]

    filtered = await app.orders_list(
        gateway_request('/v2/orders', query={'status': 'open', 'symbols': 'AAPL,MSFT'})
    )
    filtered_payload = json.loads(filtered.body.decode())
    assert filtered_payload == [{'id': 'filtered'}]

    order = await app.orders_get_by_id('abc', gateway_request('/v2/orders/abc'))
    order_payload = json.loads(order.body.decode())
    assert order_payload['id'] == 'abc'

    await app.positions_list_v2(gateway_request('/v2/positions'))
    await app.positions_get("AAPL", gateway_request("/v2/positions/AAPL"))
    await app.positions_close_all(gateway_request("/v2/positions", method="DELETE"), cancel_orders=False)
    await app.positions_close(
        "AAPL", gateway_request("/v2/positions/AAPL", method="DELETE"), cancel_orders=False
    )

    watchlist_request = app.WatchlistIn(name="wl", symbols=["AAPL"])
    await app.watchlists_list_v2(gateway_request("/v2/watchlists"))
    await app.watchlists_create_v2(
        watchlist_request, gateway_request("/v2/watchlists", method="POST")
    )
    await app.watchlists_get_v2("wl", gateway_request("/v2/watchlists/wl"))
    await app.watchlists_update_v2(
        "wl", watchlist_request, gateway_request("/v2/watchlists/wl", method="PUT")
    )
    delete_resp = await app.watchlists_delete_v2("wl", gateway_request("/v2/watchlists/wl", method="DELETE"))
    assert isinstance(delete_resp, Response)
    assert delete_resp.status_code == 204

    activities = await app.account_activities(
        gateway_request("/v2/account/activities", query={"direction": "desc"}), direction="desc"
    )
    assert activities == []

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
    quotes = app.get_quotes_v2(
        gateway_request("/v2/quotes"),
        "AAPL",
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
    )
    trades = app.get_trades_v2(
        gateway_request("/v2/trades"),
        "AAPL",
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
    )
    bars = app.get_bars_v2(
        gateway_request("/v2/bars"),
        "AAPL",
        timeframe="1Day",
        start="2024-01-01T00:00:00",
        end="2024-01-02T00:00:00",
    )

    assert quotes[0]["symbol"] == "AAPL"
    assert trades[0]["trade"] == 1
    assert bars[0]["bar"] == 1
    assert fake_client._responses == []

def test_spec_serving(monkeypatch):

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/openapi.json",
        "headers": [],
        "query_string": b"",
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    }

    request = Request(scope, _receive)

    response = extras.well_known_openapi(request)
    assert isinstance(response, JSONResponse)
    schema = json.loads(response.body.decode())
    assert schema["servers"][0]["url"] == "http://127.0.0.1:8000"
    assert schema["security"] == [{"EdgeApiKey": []}]

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


def test_prepare_order_payload_removes_redundant_stop(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "day",
        "order_class": "bracket",
        "stop_price": 225.0,
        "take_profit": {"limit_price": 230.0},
        "stop_loss": {"stop_price": 225.0, "limit_price": 224.0},
        "client_order_id": "existing-cid",
    }
    result = app._prepare_order_payload(payload)

    assert "stop_price" not in result
    assert result["stop_loss"] == {"stop_price": 225.0, "limit_price": 224.0}
    assert result["take_profit"] == {"limit_price": 230.0}
    assert result["client_order_id"] == "existing-cid"


def test_prepare_order_payload_drops_trailing_fields(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {
        "symbol": "AAPL",
        "type": "limit",
        "time_in_force": "day",
        "trail_price": 1.5,
        "trail_percent": 0,
    }
    result = app._prepare_order_payload(payload)

    assert "trail_price" not in result
    assert "trail_percent" not in result


def test_prepare_order_payload_keeps_trailing_fields_for_trailing_stop(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {
        "symbol": "AAPL",
        "type": "trailing_stop",
        "time_in_force": "day",
        "trail_percent": 2.0,
    }
    result = app._prepare_order_payload(payload)

    assert result["trail_percent"] == 2.0
    assert "trail_price" not in result


def test_prepare_order_payload_prunes_stop_loss_none_values(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {
        "symbol": "AAPL",
        "type": "market",
        "time_in_force": "day",
        "stop_price": 200.0,
        "stop_loss": {"stop_price": 200.0, "limit_price": None},
    }
    result = app._prepare_order_payload(payload)

    assert result["stop_loss"] == {"stop_price": 200.0}
    assert "stop_price" not in result


def test_prepare_order_payload_keeps_stop_price_for_stop_orders(monkeypatch):

    monkeypatch.setattr(app, "_enforce_ext_policy", lambda payload: payload)
    monkeypatch.setattr(app, "_resolve_client_order_id", lambda cid: cid or "generated")

    payload = {
        "symbol": "AAPL",
        "type": "stop",
        "time_in_force": "day",
        "stop_price": 205.5,
    }
    result = app._prepare_order_payload(payload)

    assert result["stop_price"] == 205.5


@pytest.mark.asyncio
async def test_submit_order_async_success(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 200, {"Content-Type": "application/json"}, json.dumps({"ok": True})

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_trading_http_client", lambda: object())

    result = await app._submit_order_async({"symbol": "AAPL"})
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_submit_order_async_error(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 500, {"Content-Type": "text/plain"}, "boom"

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_trading_http_client", lambda: object())

    with pytest.raises(HTTPException) as excinfo:
        await app._submit_order_async({"symbol": "AAPL"})

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_submit_order_async_invalid_json(monkeypatch):
    monkeypatch.setattr(app, "_prepare_order_payload", lambda payload: payload)

    async def fake_request(client, method, path, **kwargs):
        return 200, {"Content-Type": "application/json"}, "not json"

    monkeypatch.setattr(app, "_request_with_retry", fake_request)
    monkeypatch.setattr(app, "_get_trading_http_client", lambda: object())

    with pytest.raises(HTTPException):
        await app._submit_order_async({"symbol": "AAPL"})


@pytest.mark.asyncio
async def test_order_create_endpoint(monkeypatch):
    payloads = {}

    def fake_require(request: Request):
        payloads['auth'] = app._gateway_header_value(request)

    def fake_payload(model):
        return {'symbol': model.symbol}

    async def fake_submit(payload):
        payloads['submitted'] = payload
        return {'ok': True}

    monkeypatch.setattr(app, '_require_gateway_key_from_request', fake_require)
    monkeypatch.setattr(app, '_order_payload_from_model', fake_payload)
    monkeypatch.setattr(app, '_submit_order_async', fake_submit)

    model = app.CreateOrder(symbol='AAPL', side='buy', type='market', time_in_force='day')
    result = await app.order_create(model, gateway_request('/v2/orders', method='POST'))

    assert result == {'ok': True}
    assert payloads['auth'] == 'edge-key'
    assert payloads['submitted'] == {'symbol': 'AAPL'}


def test_order_create_sync_endpoint(monkeypatch):
    payloads = {}

    def fake_require(request: Request):
        payloads['auth'] = app._gateway_header_value(request)

    def fake_payload(model):
        return {'symbol': model.symbol}

    def fake_submit(payload):
        payloads['submitted'] = payload
        return {'ok': True}

    monkeypatch.setattr(app, '_require_gateway_key_from_request', fake_require)
    monkeypatch.setattr(app, '_order_payload_from_model', fake_payload)
    monkeypatch.setattr(app, '_submit_order_sync', fake_submit)

    model = app.CreateOrder(symbol='AAPL', side='buy', type='market', time_in_force='day')
    result = app.order_create_sync(model, gateway_request('/v2/orders', method='POST'))

    assert result == {'ok': True}
    assert payloads['auth'] == 'edge-key'
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


def test_default_server_url_and_normalisation(monkeypatch):

    monkeypatch.delenv("SERVER_URL", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_STATIC_URL", raising=False)
    assert app._default_server_url() == app.PRODUCTION_SERVER_URL

    monkeypatch.setenv("SERVER_URL", "http://example.test/api")
    assert app._default_server_url() == "http://example.test/api"

    monkeypatch.setenv("SERVER_URL", "http://alpaca-py-production.up.railway.app")
    assert app._default_server_url() == app.PRODUCTION_SERVER_URL

    assert app._normalise_server_url("http://alpaca-py-production.up.railway.app") == app.PRODUCTION_SERVER_URL



def test_http_client_for_path_selects_data(monkeypatch):

    monkeypatch.setattr(app, '_get_data_http_client', lambda: 'data')
    monkeypatch.setattr(app, '_get_trading_http_client', lambda: 'trading')

    assert app._http_client_for_path('/v2/stocks/aapl') == 'data'
    assert app._http_client_for_path('/v2/orders') == 'trading'


def test_alpaca_credentials_present(monkeypatch):

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert app._alpaca_credentials_present() is False

    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    assert app._alpaca_credentials_present() is True



def test_edge_http_client_auth_headers(monkeypatch):

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    client = edge_http.EdgeHttpClient("https://example.com")
    with pytest.raises(HTTPException) as exc:
        client._auth_headers()
    assert exc.value.status_code == 503

    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    headers = client._auth_headers()
    assert headers["APCA-API-KEY-ID"] == "id"
    assert headers["APCA-API-SECRET-KEY"] == "secret"


def test_require_gateway_key(monkeypatch):

    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.delenv("X_API_KEY", raising=False)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)

    with pytest.raises(HTTPException):
        app._require_gateway_key(header_key="wrong")

    app._require_gateway_key(header_key="edge-key")

    monkeypatch.setattr(app, "EDGE_API_KEY", None, raising=False)
    monkeypatch.setenv("EDGE_API_KEY", "env-key")

    with pytest.raises(HTTPException):
        app._require_gateway_key(header_key="edge-key")

    app._require_gateway_key(header_key="env-key")

    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.setenv("X_API_KEY", "override")

    with pytest.raises(HTTPException):
        app._require_gateway_key(header_key="edge-key")

    app._require_gateway_key(header_key="override")



def test_readyz_detail(monkeypatch):

    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

    payload = extras.readyz(gateway_request('/readyz'))
    assert payload['ok'] is True
    assert payload['env']['alpaca_credentials'] is True
    assert payload['env']['api_base_url']



def test_readyz_reports_failures(monkeypatch):

    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_BASE_URL", "")

    payload = extras.readyz(gateway_request('/readyz'))
    assert payload['ok'] is False
    assert payload['env']['alpaca_credentials'] is False


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_account_activities_translates_unauthorized(monkeypatch):

    monkeypatch.delenv("EDGE_API_KEY", raising=False)
    monkeypatch.delenv("X_API_KEY", raising=False)
    monkeypatch.setattr(app, "EDGE_API_KEY", "edge-key", raising=False)

    async def fake_request_with_retry(client, method, path, **kwargs):
        assert kwargs.get("params") == {"direction": "desc"}
        return 401, {"Content-Type": "application/json"}, '{"message": "unauthorized"}'

    monkeypatch.setattr(app, "_get_trading_http_client", lambda: object())
    monkeypatch.setattr(app, "_request_with_retry", fake_request_with_retry)

    response = await app.account_activities(
        gateway_request('/v2/account/activities'), direction="desc"
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert json.loads(response.body.decode())["detail"].startswith("Activities not enabled")

