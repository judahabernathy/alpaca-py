import os
import re
import asyncio
from datetime import datetime
from typing import List, Optional, Any, Dict, Coroutine
from uuid import uuid4

import aiohttp
import config
from fastapi import FastAPI, Header, HTTPException, Query, Path, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    CreateWatchlistRequest, UpdateWatchlistRequest
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, StockQuotesRequest, StockTradesRequest
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

DEFAULT_LIMIT_GUARD_RESULT: Dict[str, Any] = {
    "ok": True,
    "reason": None,
    "age": None,
    "drift": None,
    "debug": {},
    "message": None,
}


def _resolve_client_order_id(candidate: Optional[str]) -> str:
    if candidate:
        return candidate
    if os.getenv("ALPHA_REQUIRE_CLIENT_ID"):
        raise HTTPException(status_code=400, detail="client_order_id required")
    return uuid4().hex


async def _request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 0.5,
    **kwargs,
) -> aiohttp.ClientResponse:
    attempt = 0
    while True:
        response = await session.request(method, url, **kwargs)
        if response.status == 429:
            body = await response.text()
            retry_after = response.headers.get("Retry-After")
            await response.release()
            if attempt >= retries - 1:
                detail = {
                    "message": body or "Rate limited",
                    "retry_after": retry_after,
                }
                raise HTTPException(status_code=429, detail=detail)
            delay = float(retry_after or backoff)
            await asyncio.sleep(delay)
            attempt += 1
            backoff *= 2
            continue
        if response.status >= 400:
            body = await response.text()
            await response.release()
            raise HTTPException(status_code=response.status, detail=body or response.reason)
        return response


def _run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def evaluate_limit_guard(**payload: Any) -> Dict[str, Any]:
    url = os.getenv("ALPHA_LIMIT_GUARD_URL")
    if not url:
        return dict(DEFAULT_LIMIT_GUARD_RESULT)

    timeout = aiohttp.ClientTimeout(total=float(os.getenv("ALPHA_LIMIT_GUARD_TIMEOUT", "2")))

    async def _invoke() -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await _request_with_retry(session, "POST", url, json=payload)
            try:
                data = await response.json(content_type=None)
            except aiohttp.ContentTypeError:
                text = await response.text()
                await response.release()
                raise HTTPException(status_code=502, detail=f"Invalid JSON from guard: {text}")
            await response.release()
            return data

    try:
        data = _run_coroutine(_invoke())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fallback path
        result = dict(DEFAULT_LIMIT_GUARD_RESULT)
        result["debug"] = {"error": str(exc)}
        return result

    result = dict(DEFAULT_LIMIT_GUARD_RESULT)
    result.update({k: data.get(k) for k in ["ok", "reason", "age", "drift", "debug", "message"] if k in data})
    return result


def _enforce_ext_policy(payload: Dict[str, Any]) -> None:
    if not os.getenv("ALPHA_ENFORCE_EXT"):
        return

    order_type = (payload.get("type") or "").lower()
    tif = (payload.get("time_in_force") or "").lower()
    extended = bool(payload.get("extended_hours"))
    order_class = (payload.get("order_class") or "").lower() if payload.get("order_class") else None

    if extended:
        if order_type != "limit":
            raise HTTPException(status_code=400, detail="Extended hours orders must be limit orders")
        if tif != "day":
            raise HTTPException(status_code=400, detail="Extended hours orders require day time_in_force")
        if payload.get("limit_price") is None:
            raise HTTPException(status_code=400, detail="limit_price required for extended hours orders")
        if order_class:
            raise HTTPException(status_code=400, detail="Extended hours orders do not support advanced order classes")

    if order_type == "limit":
        guard_result = evaluate_limit_guard(
            symbol=payload.get("symbol"),
            limit_price=payload.get("limit_price"),
            side=payload.get("side"),
            extended_hours=extended,
            time_in_force=tif,
        )
        if not guard_result.get("ok", True):
            reason = (guard_result.get("reason") or "limit_guard").lower()
            status_map = {"stale": 428, "ttl": 428, "drift": 409}
            status = status_map.get(reason, 400)
            detail = {
                "message": guard_result.get("message") or f"Limit order blocked: {reason}",
                "reason": reason,
                "age": guard_result.get("age"),
                "drift": guard_result.get("drift"),
                "debug": guard_result.get("debug"),
            }
            raise HTTPException(status_code=status, detail=detail)

app = FastAPI(title="Alpaca Wrapper")

def check_key(x_api_key: Optional[str]):
    service_key = os.getenv("X_API_KEY")
    if not service_key or x_api_key != service_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

def trading_client() -> TradingClient:
    base_url = config.APCA_API_BASE_URL
    return TradingClient(
        api_key=os.environ["APCA_API_KEY_ID"],
        secret_key=os.environ["APCA_API_SECRET_KEY"],
        paper=("paper" in base_url)
    )

def md_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        api_key=os.environ["APCA_API_KEY_ID"],
        secret_key=os.environ["APCA_API_SECRET_KEY"]
    )

class OrderIn(BaseModel):
    symbol: str
    side: str
    qty: float | None = None
    notional: float | None = None
    type: str
    time_in_force: str
    limit_price: float | None = None
    extended_hours: bool = False
    order_class: Optional[str] = None
    client_order_id: Optional[str] = None

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
def healthz(): return {"ok": True}

# -- Orders
@app.post("/v1/order")
def submit_order(order: OrderIn, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    payload = order.model_dump()
    _enforce_ext_policy(payload)
    client_order_id = _resolve_client_order_id(payload.get("client_order_id"))
    tc = trading_client()
    side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
    tif = TimeInForce(order.time_in_force.lower())
    if order.type.lower() == "market":
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            notional=order.notional,
            side=side,
            time_in_force=tif,
        )
    elif order.type.lower() == "limit":
        if order.limit_price is None:
            raise HTTPException(400, "limit_price required for limit orders")
        req = LimitOrderRequest(symbol=order.symbol, qty=order.qty,
                                side=side, time_in_force=tif, limit_price=order.limit_price)
    else:
        raise HTTPException(400, "unsupported order type")
    setattr(req, "client_order_id", client_order_id)
    setattr(req, "extended_hours", order.extended_hours)
    if order.order_class:
        setattr(req, "order_class", order.order_class)
    o = tc.submit_order(order_data=req)
    return o.model_dump() if hasattr(o, "model_dump") else o.__dict__

@app.get("/v1/orders")
def list_orders(status: str = Query("open"), x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    tc = trading_client()
    orders = tc.get_orders()
    return [o.model_dump() for o in orders]

@app.get("/v1/orders/{order_id}")
def get_order(order_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    tc = trading_client()
    o = tc.get_order_by_id(order_id)
    return o.model_dump() if hasattr(o, "model_dump") else o.__dict__

@app.delete(
    "/v1/orders/{order_id}",
    status_code=204,
    summary="Cancel order",
    response_description="Order cancelled",
)
def cancel_order(order_id: str, x_api_key: Optional[str] = Header(None)):
    """Cancel an open order by its ID.

    Alpaca's trading API uses the ``/v2/orders/{order_id}`` endpoint for
    cancellations and expects the DELETE request to omit a payload.  Delegate
    to :class:`~alpaca.trading.client.TradingClient` so the underlying request
    is constructed correctly and no JSON body is sent.
    """

    check_key(x_api_key)

    tc = trading_client()
    # The TradingClient handles targeting the v2 endpoint and omits the
    # request body, which matches the behaviour expected by Alpaca's REST API.
    tc.cancel_order_by_id(order_id)

    # The decorator defines a 204 status code, so simply return an empty response.
    return Response(status_code=204)

# -- Account
@app.get("/v1/account")
def get_account(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    acct = trading_client().get_account()
    return acct.model_dump() if hasattr(acct, "model_dump") else acct.__dict__

# -- Positions
@app.get("/v1/positions")
def list_positions(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    positions = trading_client().get_all_positions()
    return [p.model_dump() for p in positions]

@app.get("/v1/positions/{symbol}")
def get_position(symbol: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    pos = trading_client().get_open_position(symbol)
    return pos.model_dump() if hasattr(pos, "model_dump") else pos.__dict__

@app.delete("/v1/positions/{symbol}")
def close_position(symbol: str, cancel_orders: bool = Query(False), x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    return trading_client().close_position(symbol, cancel_orders=cancel_orders)

@app.delete("/v1/positions")
def close_all_positions(cancel_orders: bool = Query(False), x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    return trading_client().close_all_positions(cancel_orders=cancel_orders)

# -- Watchlists
@app.get("/v1/watchlists")
def list_watchlists(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    wls = trading_client().get_watchlists()
    return [wl.model_dump() for wl in wls]

class WatchlistIn(BaseModel):
    name: str
    symbols: List[str]

@app.post("/v1/watchlists")
def create_watchlist(w: WatchlistIn, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    req = CreateWatchlistRequest(name=w.name, symbols=w.symbols)
    wl = trading_client().create_watchlist(req)
    return wl.model_dump()

@app.get("/v1/watchlists/{watchlist_id}")
def get_watchlist(watchlist_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    wl = trading_client().get_watchlist(watchlist_id)
    return wl.model_dump()

@app.put("/v1/watchlists/{watchlist_id}")
def update_watchlist(watchlist_id: str, w: WatchlistIn, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    req = UpdateWatchlistRequest(name=w.name, symbols=w.symbols)
    wl = trading_client().update_watchlist(watchlist_id, req)
    return wl.model_dump()

@app.delete("/v1/watchlists/{watchlist_id}")
def delete_watchlist(watchlist_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    return trading_client().delete_watchlist(watchlist_id)

# -- Quotes & Trades (basic market data)
@app.get("/v1/quotes")
def get_quotes(symbol: str, start: str, end: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    req = StockQuotesRequest(symbol_or_symbols=[symbol],
                             start=datetime.fromisoformat(start),
                             end=datetime.fromisoformat(end))
    quotes = md_client().get_stock_quotes(req)
    return quotes.df.reset_index().to_dict(orient="records")

@app.get("/v1/trades")
def get_trades(symbol: str, start: str, end: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    req = StockTradesRequest(symbol_or_symbols=[symbol],
                             start=datetime.fromisoformat(start),
                             end=datetime.fromisoformat(end))
    trades = md_client().get_stock_trades(req)
    return trades.df.reset_index().to_dict(orient="records")

@app.get("/v1/bars")
def get_bars(symbol: str, timeframe: str = Query("1Day"), start: str = Query(...),
             end: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    tf = parse_timeframe(timeframe)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end) if end else None
    )
    bars = md_client().get_stock_bars(req)
    return bars.df.reset_index().to_dict(orient="records")

@app.get("/openapi.yaml")
def spec():
    return FileResponse("openapi.yaml")


