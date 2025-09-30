import asyncio
import json
import os
import re
import time
from copy import deepcopy
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union, cast
from uuid import uuid4

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.params import Param
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import APCA_API_BASE_URL

from .http import EdgeHttpClient
from .logging import (
    configure_logging,
    log_duration,
    log_error,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)

API_BASE_URL = APCA_API_BASE_URL
API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
EDGE_API_KEY = (
    os.getenv("EDGE_API_KEY")
    or os.getenv("GATEWAY_API_KEY")
    or os.getenv("X_API_KEY")
    or ""
).strip() or None

configure_logging()


@asynccontextmanager
async def _app_lifespan(application: FastAPI):
    client = EdgeHttpClient(API_BASE_URL)
    await client.startup()
    application.state.http_client = client
    log_event("startup_complete")
    try:
        yield
    finally:
        client_ref = getattr(application.state, "http_client", None)
        if client_ref is not None:
            await client_ref.shutdown()
            application.state.http_client = None
        log_event("shutdown_complete")


app = FastAPI(title="Alpaca Wrapper", version="1.0.3", lifespan=_app_lifespan)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    token = set_correlation_id(correlation_id)
    start_time = time.perf_counter()
    try:
        log_event("incoming_request", method=request.method, path=str(request.url.path))
        api_key_header = request.headers.get("X-API-Key")
        log_event("gateway_key_header", present=bool(api_key_header), length=len(api_key_header or ""))
        response = await call_next(request)
        response.headers.setdefault("X-Correlation-ID", correlation_id)
        log_duration("request_complete", start_time, status=response.status_code)
        return response
    finally:
        reset_correlation_id(token)

def evaluate_limit_guard(symbol: str, limit_price: Any, side: str) -> None:
    symbol_code = (symbol or "").upper()
    if not symbol_code or limit_price is None:
        return

    try:
        limit_value = float(cast(Union[str, float, int], limit_price))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail={"reason": "invalid_limit_price"}) from None

    normalised_side = (side or "").lower()
    if normalised_side not in {"buy", "sell"}:
        return

    try:
        ttl_seconds = float(os.getenv("ALPHA_QUOTE_TTL_SECONDS", "10"))
    except ValueError:
        ttl_seconds = 10.0

    try:
        drift_bps = float(os.getenv("ALPHA_MAX_DRIFT_BPS", "100"))
    except ValueError:
        drift_bps = 100.0

    feed = os.getenv("ALPHA_QUOTE_FEED", "iex")

    try:
        latest = md_client().get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol_code, feed=feed)  # type: ignore[arg-type]
        )
    except Exception:
        return

    quote = latest.get(symbol_code) if isinstance(latest, dict) else None
    if quote is None:
        return

    timestamp = getattr(quote, "timestamp", None)
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            timestamp = None
    if timestamp is None:
        return
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age = max((now - timestamp).total_seconds(), 0.0)
    if age > max(ttl_seconds, 0.0):
        raise HTTPException(
            status_code=428,
            detail={"reason": "stale", "age": age, "ttl_seconds": ttl_seconds, "feed": feed},
        )

    reference_attr = "ask_price" if normalised_side == "buy" else "bid_price"
    reference = getattr(quote, reference_attr, None)
    if reference in (None, ""):
        return

    try:
        reference_value = float(cast(Union[str, float, int], reference))
    except (TypeError, ValueError):
        return

    if reference_value <= 0:
        return

    drift = (limit_value - reference_value) / reference_value
    max_drift = max(drift_bps, 0.0) / 10000.0
    exceeds = drift > max_drift if normalised_side == "buy" else (-drift) > max_drift
    if exceeds:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "drift",
                "drift": drift,
                "max_drift": max_drift,
                "side": normalised_side,
            },
        )


def _resolve_client_order_id(client_order_id: Optional[str]) -> str:
    require = os.getenv("ALPHA_REQUIRE_CLIENT_ID") == "1"
    if client_order_id:
        return client_order_id
    if require:
        raise HTTPException(status_code=400, detail="client_order_id required")
    return uuid4().hex


def _enforce_ext_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not os.getenv("ALPHA_ENFORCE_EXT"):
        return payload

    order_type = (payload.get("type") or "").lower()
    tif = (payload.get("time_in_force") or "").lower()
    extended_hours = bool(payload.get("extended_hours"))

    if extended_hours:
        if order_type != "limit":
            raise HTTPException(status_code=400, detail="Extended hours orders must be limit orders")
        if tif != "day":
            raise HTTPException(status_code=400, detail="Extended hours limit orders must be DAY")
        if payload.get("order_class"):
            raise HTTPException(status_code=400, detail="Extended hours does not support advanced order classes")
        if payload.get("limit_price") is None:
            raise HTTPException(status_code=400, detail="Extended hours limit orders require limit_price")

    if order_type == "limit" and payload.get("limit_price") is not None:
        try:
            evaluate_limit_guard((payload.get("symbol") or ""), payload.get("limit_price"), (payload.get("side") or ""))
        except HTTPException as exc:
            if exc.status_code in {400, 409, 428}:
                raise
            raise


    return payload


def _max_retry_attempts() -> int:
    try:
        value = int(os.getenv("ALPHA_HTTP_MAX_RETRIES", "3"))
    except ValueError:
        value = 3
    return max(1, value)


def _retry_delay(retry_after: Optional[str]) -> float:
    if not retry_after:
        return 1.0
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return 1.0



async def _request_with_retry(
    http_client: EdgeHttpClient, method: str, path: str, **kwargs: Any
) -> tuple[int, Dict[str, str], str]:
    attempts = 0
    max_attempts = _max_retry_attempts()

    while True:
        status, headers, body = await http_client.request(
            method,
            path,
            headers=_alpaca_headers(),
            **kwargs,
        )
        if status != 429:
            return status, headers, body

        attempts += 1
        if attempts >= max_attempts:
            log_error("retry_exhausted", method=method, path=path, status=status)
            raise HTTPException(status_code=429, detail=body or "rate limited")

        await asyncio.sleep(_retry_delay(headers.get("Retry-After")))








def _gateway_key() -> Optional[str]:
    def _normalise(value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        return value or None

    env_key = _normalise(os.getenv("EDGE_API_KEY"))
    if env_key is None:
        env_key = _normalise(os.getenv("GATEWAY_API_KEY"))
    if env_key is None:
        env_key = _normalise(os.getenv("X_API_KEY"))

    attr_key = _normalise(EDGE_API_KEY)
    if attr_key and attr_key != env_key:
        return attr_key
    return env_key or attr_key


def _require_gateway_key(header_key: Optional[str]) -> None:
    expected_key = _gateway_key()
    if expected_key and header_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _alpaca_headers():
    key_id = API_KEY_ID or os.getenv("APCA_API_KEY_ID")
    secret_key = API_SECRET_KEY or os.getenv("APCA_API_SECRET_KEY")
    if not (key_id and secret_key):
        raise HTTPException(status_code=500, detail="Upstream credentials not configured")
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
    }

def _get_http_client() -> EdgeHttpClient:
    client = getattr(app.state, "http_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="HTTP client is unavailable")
    return client


def _decode_json(headers: Dict[str, str], body: str) -> Any:
    content_type = headers.get("Content-Type", "")
    if "application/json" in content_type.lower():
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="Invalid JSON from upstream") from exc
    return body







def md_client() -> StockHistoricalDataClient:
    api_key = API_KEY_ID or os.getenv("APCA_API_KEY_ID")
    secret_key = API_SECRET_KEY or os.getenv("APCA_API_SECRET_KEY")
    kwargs: Dict[str, Any] = {}
    if api_key and secret_key:
        kwargs = {"api_key": api_key, "secret_key": secret_key}
    return StockHistoricalDataClient(**kwargs)

class CreateOrder(BaseModel):
    symbol: str
    side: str
    qty: Optional[float] = None
    notional: Optional[float] = None
    type: str
    time_in_force: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_class: Optional[str] = None
    take_profit: Optional[Dict[str, Any]] = None
    stop_loss: Optional[Dict[str, Any]] = None
    extended_hours: Optional[bool] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: Optional[str] = None


class WatchlistIn(BaseModel):
    name: str
    symbols: List[str]


def _order_payload_from_model(order: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(order, BaseModel):
        return order.model_dump(exclude_none=True)
    return {k: v for k, v in dict(order).items() if v is not None}


def _prepare_order_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalised = dict(payload)
    _enforce_ext_policy(normalised)
    normalised["client_order_id"] = _resolve_client_order_id(normalised.get("client_order_id"))
    return normalised


async def _submit_order_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _prepare_order_payload(payload)
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "POST",
        "/v2/orders",
        json=prepared,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body or "order rejected")
    return _decode_json(headers, body)


def _submit_order_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    previous_loop = None
    try:
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_submit_order_async(payload))
        loop.run_until_complete(loop.shutdown_asyncgens())
        return result
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()

def parse_timeframe(tf_str: str) -> TimeFrame:
    m = re.match(r'^(\d+)([A-Za-z]+)$', tf_str)
    if not m:
        raise HTTPException(status_code=400, detail="Invalid timeframe format")
    amount = int(m.group(1))
    unit = m.group(2).lower()
    units = {
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    if unit not in units:
        raise HTTPException(status_code=400, detail="Unsupported timeframe unit")
    return TimeFrame(amount, units[unit])


@app.get("/healthz")
def healthz():
    client_ready = isinstance(getattr(app.state, "http_client", None), EdgeHttpClient)
    creds_ok = True
    try:
        _alpaca_headers()
    except HTTPException:
        creds_ok = False
    dependencies = {
        "http_client": client_ready,
        "alpaca_credentials": creds_ok,
        "api_base_url": bool(API_BASE_URL),
    }
    return {
        "ok": all(dependencies.values()),
        "dependencies": dependencies,
    }




@app.get("/readyz")
async def readyz(
    detail: bool = Query(False),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Readiness probe that verifies upstream dependencies."""
    _require_gateway_key(x_api_key)
    client_ready = True
    try:
        _get_http_client()
    except HTTPException:
        client_ready = False

    creds_ok = True
    try:
        _alpaca_headers()
    except HTTPException:
        creds_ok = False

    base_url_ok = bool(API_BASE_URL)
    dependencies = {
        "http_client": client_ready,
        "alpaca_credentials": creds_ok,
        "api_base_url": base_url_ok,
    }
    ready = all(dependencies.values())
    failing = [name for name, status in dependencies.items() if not status]
    detail_payload: Any
    if detail:
        detail_payload = dependencies
    else:
        detail_payload = {"failing": failing}
    return {"ready": ready, "detail": detail_payload}


# -- Orders
@app.post("/v2/orders/sync")
def order_create_sync(
    order: CreateOrder,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Synchronously mirror `/v2/orders` in the current thread.

    Applies the limit guard (extended-hours must be DAY limit orders)
    and surfaces upstream HTTP 429 responses with their `Retry-After`.
    """
    _require_gateway_key(x_api_key)
    payload = _order_payload_from_model(order)
    return _submit_order_sync(payload)


@app.post("/v2/orders")
async def order_create(
    order: CreateOrder,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Submit an order via Alpaca's `/v2/orders`. 

    Requires `X-API-Key`, forwards 429 responses with `Retry-After`, and
    applies the limit guard (extended-hours must be DAY limit orders).
    """
    _require_gateway_key(x_api_key)
    payload = _order_payload_from_model(order)
    return await _submit_order_async(payload)


@app.get("/v2/account")
async def account_get(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    status, headers, body = await _request_with_retry(_get_http_client(), "GET", "/v2/account")
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.get("/v2/account/activities")
async def account_activities(
    activity_type: Optional[str] = None,
    date: Optional[str] = None,
    until: Optional[str] = None,
    after: Optional[str] = None,
    direction: Optional[str] = None,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    _require_gateway_key(x_api_key)
    params = {
        "activity_type": activity_type,
        "date": date,
        "until": until,
        "after": after,
        "direction": direction,
        "page_size": page_size,
        "page_token": page_token,
    }
    filtered = {
        key: (str(value) if isinstance(value, (int, float)) else value)
        for key, value in params.items()
        if value is not None
    }
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "GET",
        "/v2/account/activities",
        params=filtered or None,
    )
    if status == 401:
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Activities not enabled for this account; use Orders for fills.",
            },
        )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.get("/v2/orders")
async def orders_list(
    status: Optional[str] = Query(None),
    symbols: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    _require_gateway_key(x_api_key)
    params: Dict[str, Any] = {}
    status_value = None if isinstance(status, Param) else status
    symbols_value = None if isinstance(symbols, Param) else symbols
    if status_value:
        params["status"] = status_value
    normalized_symbols: List[str] = []
    if symbols_value:
        values = symbols_value if isinstance(symbols_value, (list, tuple, set)) else [symbols_value]
        for value in values:
            normalized_symbols.extend(part.strip() for part in str(value).split(",") if part.strip())
    if normalized_symbols:
        params["symbols"] = ",".join(normalized_symbols)
    status_code, headers, body = await _request_with_retry(
        _get_http_client(),
        "GET",
        "/v2/orders",
        params=params or None,
    )
    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=body)
    return _decode_json(headers, body)


@app.get("/v2/orders/{order_id}")
async def orders_get_by_id(
    order_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "GET",
        f"/v2/orders/{order_id}",
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.delete("/v2/orders")
async def orders_cancel_all(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Cancel all open orders."""
    return await _proxy_alpaca_request(
        "DELETE",
        "/v2/orders",
        x_api_key,
    )


@app.delete("/v2/orders/{order_id}")
async def orders_cancel_by_id(
    order_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Cancel a specific order by id."""
    return await _proxy_alpaca_request(
        "DELETE",
        f"/v2/orders/{order_id}",
        x_api_key,
    )


@app.get("/v2/positions")
async def positions_list_v2(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "GET",
        "/v2/positions",
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.get("/v2/positions/{symbol}")
async def positions_get(
    symbol: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "GET",
        f"/v2/positions/{symbol}",
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.delete("/v2/positions")
async def positions_close_all(
    cancel_orders: bool = Query(False),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    params = {"cancel_orders": str(cancel_orders).lower()}
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "DELETE",
        "/v2/positions",
        params=params,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


@app.delete("/v2/positions/{symbol}")
async def positions_close(
    symbol: str,
    cancel_orders: bool = Query(False),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    _require_gateway_key(x_api_key)
    params = {"cancel_orders": str(cancel_orders).lower()}
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        "DELETE",
        f"/v2/positions/{symbol}",
        params=params,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)


async def _proxy_alpaca_request(
    method: str,
    path: str,
    x_api_key: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
):
    _require_gateway_key(x_api_key)
    status, headers, body = await _request_with_retry(
        _get_http_client(),
        method,
        path,
        json=payload,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)

    content_type = (headers.get("Content-Type") or "").lower()
    if status == 204 or not body:
        return Response(status_code=status)
    if "application/json" in content_type:
        data = _decode_json(headers, body)
        if isinstance(data, (dict, list)):
            return JSONResponse(status_code=status, content=data)
    return Response(
        status_code=status,
        content=body,
        media_type=headers.get("Content-Type") or "text/plain",
    )


@app.get("/v2/watchlists")
async def watchlists_list_v2(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    return await _proxy_alpaca_request("GET", "/v2/watchlists", x_api_key)


@app.post("/v2/watchlists")
async def watchlists_create_v2(
    watchlist: WatchlistIn,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    return await _proxy_alpaca_request(
        "POST",
        "/v2/watchlists",
        x_api_key,
        payload=watchlist.model_dump(),
    )



@app.get("/v2/watchlists/{watchlist_id}")
async def watchlists_get_v2(
    watchlist_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    return await _proxy_alpaca_request(
        "GET", f"/v2/watchlists/{watchlist_id}", x_api_key
    )


@app.put("/v2/watchlists/{watchlist_id}")
async def watchlists_update_v2(
    watchlist_id: str,
    watchlist: WatchlistIn,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    return await _proxy_alpaca_request(
        "PUT",
        f"/v2/watchlists/{watchlist_id}",
        x_api_key,
        payload=watchlist.model_dump(),
    )


@app.delete("/v2/watchlists/{watchlist_id}")
async def watchlists_delete_v2(
    watchlist_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    return await _proxy_alpaca_request(
        "DELETE", f"/v2/watchlists/{watchlist_id}", x_api_key
    )



# -- Market Data
@app.get("/v2/quotes")
def get_quotes_v2(
    symbol: str,
    start: str,
    end: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> list[dict]:
    _require_gateway_key(x_api_key)
    req = StockQuotesRequest(
        symbol_or_symbols=[symbol],
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
    )
    quotes = md_client().get_stock_quotes(req)
    quotes_df = getattr(quotes, "df", None)
    if quotes_df is None:
        raise HTTPException(status_code=502, detail="Quotes response missing dataframe")
    return quotes_df.reset_index().to_dict(orient="records")


@app.get("/v2/trades")
def get_trades_v2(
    symbol: str,
    start: str,
    end: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> list[dict]:
    _require_gateway_key(x_api_key)
    req = StockTradesRequest(
        symbol_or_symbols=[symbol],
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
    )
    trades = md_client().get_stock_trades(req)
    trades_df = getattr(trades, "df", None)
    if trades_df is None:
        raise HTTPException(status_code=502, detail="Trades response missing dataframe")
    return trades_df.reset_index().to_dict(orient="records")


@app.get("/v2/bars")
def get_bars_v2(
    symbol: str,
    timeframe: str = Query("1Day"),
    start: str = Query(...),
    end: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> list[dict]:
    _require_gateway_key(x_api_key)
    tf = parse_timeframe(timeframe)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end) if end else None,
    )
    bars = md_client().get_stock_bars(req)
    bars_df = getattr(bars, "df", None)
    if bars_df is None:
        raise HTTPException(status_code=502, detail="Bars response missing dataframe")
    return bars_df.reset_index().to_dict(orient="records")




# --- ChatGPT plugin support ---

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://chat.openai.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

try:
    app.mount("/.well-known", StaticFiles(directory=".well-known"), name="wellknown")
except Exception:
    pass

# --- GPT Actions OpenAPI patch ---

def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="Alpaca Wrapper",
        version="1.0.0",
        routes=app.routes,
    )
    # Force OpenAPI 3.1 and a proper base URL for GPT Actions
    schema["openapi"] = "3.1.0"
    default_server = os.getenv("SERVER_URL") or "https://alpaca-py-production.up.railway.app"
    schema["servers"] = [{"url": default_server}]

    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["EdgeApiKey"] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    schema["security"] = [{"EdgeApiKey": []}]

    info = schema.setdefault("info", {})
    info["description"] = dedent(
        """
        All endpoints require the `X-API-Key` header; the `api_key` query parameter is
        supported as a compatibility fallback. The service propagates Alpaca's rate
        limits as HTTP 429 responses and surfaces the upstream `Retry-After` header.
        When `extended_hours` is enabled the order payload must remain a DAY limit
        order, include `limit_price`, and omit advanced `order_class` values.
        """
    ).strip()

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]

SPEC_JSON_PATH = (Path(__file__).resolve().parent / ".well-known" / "openapi.json")
SPEC_YAML_PATH = SPEC_JSON_PATH.with_suffix(".yaml")


def _resolve_server_url(request: Optional[Request]) -> Optional[str]:
    if request is not None:
        return str(request.base_url).rstrip("/")
    env_url = os.getenv("SERVER_URL")
    if env_url:
        return env_url.rstrip("/")
    return None


def _serve_spec_file(path: Path, media_type: str, request: Optional[Request] = None):
    if path.exists():
        if path.suffix == ".json":
            spec: Dict[str, Any] = json.loads(path.read_text())
        else:
            loaded = yaml.safe_load(path.read_text())
            spec = loaded if isinstance(loaded, dict) else {}
    else:
        spec = deepcopy(app.openapi())
    spec_copy = deepcopy(spec)
    resolved_server = _resolve_server_url(request)
    if resolved_server:
        spec_copy["servers"] = [{"url": resolved_server}]
    if media_type == "application/json":
        return JSONResponse(spec_copy)
    return PlainTextResponse(
        yaml.safe_dump(spec_copy, sort_keys=False, allow_unicode=True),
        media_type=media_type,
    )


@app.get("/.well-known/openapi.json", include_in_schema=False)
def well_known_openapi_json(request: Request):
    return _serve_spec_file(SPEC_JSON_PATH, "application/json", request)


@app.get("/.well-known/openapi.yaml", include_in_schema=False)
def well_known_openapi_yaml(request: Request):
    return _serve_spec_file(SPEC_YAML_PATH, "application/yaml", request)


@app.get("/openapi.json", include_in_schema=False)
def openapi_json_alias(request: Request):
    return _serve_spec_file(SPEC_JSON_PATH, "application/json", request)


@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml_alias(request: Request):
    return _serve_spec_file(SPEC_YAML_PATH, "application/yaml", request)


# --- end patch ---



