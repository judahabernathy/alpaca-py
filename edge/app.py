import asyncio
import json
import os
import re
import time
from copy import deepcopy
from functools import lru_cache
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union, cast, Literal
from uuid import uuid4
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.params import Param
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import APCA_API_BASE_URL, APCA_DATA_BASE_URL, DEFAULT_DATA_BASE_URL

from .http import EdgeHttpClient
from .logging import (
    configure_logging,
    log_duration,
    log_error,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)

DEFAULT_API_BASE_URL = "https://paper-api.alpaca.markets"
PRODUCTION_SERVER_URL = "https://alpaca-py-production.up.railway.app"
API_BASE_URL = (os.getenv("APCA_API_BASE_URL") or APCA_API_BASE_URL or DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL
API_BASE_URL = API_BASE_URL.rstrip("/")
DATA_BASE_URL = (os.getenv("APCA_DATA_BASE_URL") or APCA_DATA_BASE_URL or DEFAULT_DATA_BASE_URL).strip() or DEFAULT_DATA_BASE_URL
DATA_BASE_URL = DATA_BASE_URL.rstrip("/")
API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
EDGE_API_KEY = (
    os.getenv("EDGE_API_KEY")
    or os.getenv("X_API_KEY")
    or ""
).strip() or None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-5-mini"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
try:
    OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "45"))
except ValueError:
    OPENAI_TIMEOUT = 45.0

configure_logging()


@lru_cache(maxsize=1)
def _cached_openai_client(api_key: str, base_url: Optional[str], timeout: float) -> AsyncOpenAI:
    headers = {"User-Agent": "alpaca-py/order-plan/1.0"}
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, default_headers=headers)



def _get_openai_client() -> AsyncOpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured for this deployment.",
        )
    return _cached_openai_client(OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_TIMEOUT)


@asynccontextmanager
async def _app_lifespan(application: FastAPI):
    trading_client = EdgeHttpClient(API_BASE_URL)
    data_client = EdgeHttpClient(DATA_BASE_URL)
    await trading_client.startup()
    await data_client.startup()
    application.state.trading_http_client = trading_client
    application.state.data_http_client = data_client
    application.state.http_client = trading_client  # legacy compatibility
    log_event("startup_complete")
    try:
        yield
    finally:
        seen: set[int] = set()
        for attr in ("trading_http_client", "data_http_client", "http_client"):
            client_ref = getattr(application.state, attr, None)
            if client_ref is None:
                continue
            if id(client_ref) not in seen:
                try:
                    await client_ref.shutdown()
                finally:
                    seen.add(id(client_ref))
            setattr(application.state, attr, None)
        log_event("shutdown_complete")


app = FastAPI(title="Alpaca Wrapper", version="1.0.3", lifespan=_app_lifespan)

from .extras import router as extras_router
app.include_router(extras_router)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    token = set_correlation_id(correlation_id)
    start_time = time.perf_counter()
    try:
        log_event("incoming_request", method=request.method, path=str(request.url.path))
        api_key_header = _gateway_header_value(request)
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
        if value is None:
            return None
        value = value.strip()
        return value or None

    for env_var in ("EDGE_API_KEY", "X_API_KEY"):
        env_value = _normalise(os.getenv(env_var))
        if env_value is not None:
            return env_value

    attr_value = EDGE_API_KEY
    if isinstance(attr_value, str):
        attr_value = attr_value.strip() or None
    return attr_value



def _require_gateway_key(header_key: Optional[str]) -> None:
    expected_key = _gateway_key()
    if not expected_key:
        return
    provided_key = (header_key or "").strip()
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")



def _gateway_header_value(request: Request) -> Optional[str]:
    header_value = request.headers.get("x-api-key")
    if header_value is None:
        header_value = request.headers.get("X-API-Key")
    return header_value



def _require_gateway_key_from_request(request: Request) -> None:
    _require_gateway_key(_gateway_header_value(request))



def _passthrough_json(status: int, headers: Dict[str, str], body: str) -> Response:
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    media_type = headers.get("Content-Type") or "application/json"
    response = Response(content=body or "", media_type=media_type, status_code=status)
    for name, value in headers.items():
        lower = name.lower()
        if lower in {"content-length", "content-type"}:
            continue
        response.headers.setdefault(name, value)
    return response



def _alpaca_credentials_present() -> bool:
    key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
    return bool(key_id and secret_key)


def _resolved_api_base_url() -> str:
    configured = os.getenv("APCA_API_BASE_URL", "").strip()
    if not configured:
        configured = DEFAULT_API_BASE_URL
    return configured.rstrip("/")


def _normalise_server_url(url: str) -> str:
    parsed = urlparse((url or "").strip() or PRODUCTION_SERVER_URL)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    production_host = urlparse(PRODUCTION_SERVER_URL).netloc.lower()
    if parsed.netloc.lower() == production_host and parsed.scheme != "https":
        parsed = parsed._replace(scheme="https")
    if not parsed.netloc:
        parsed = urlparse(PRODUCTION_SERVER_URL)
    normalised_path = parsed.path.rstrip("/") or ""
    return urlunparse(parsed._replace(path=normalised_path))


def _default_server_url() -> str:
    for env_var in ("SERVER_URL", "PUBLIC_BASE_URL", "RAILWAY_STATIC_URL"):
        configured = os.getenv(env_var, "").strip()
        if configured:
            return _normalise_server_url(configured)
    return PRODUCTION_SERVER_URL


def _get_trading_http_client() -> EdgeHttpClient:
    client = getattr(app.state, "trading_http_client", None) or getattr(app.state, "http_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Trading HTTP client is unavailable")
    return client



def _get_data_http_client() -> EdgeHttpClient:
    client = getattr(app.state, "data_http_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Data HTTP client is unavailable")
    return client



def _get_http_client() -> EdgeHttpClient:
    return _get_trading_http_client()


def _http_client_for_path(path: str) -> EdgeHttpClient:
    normalised = (path or "").lower()
    if normalised.startswith('/v2/stocks') or normalised.startswith('/v2/marketdata'):
        return _get_data_http_client()
    return _get_trading_http_client()


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

class OrderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=12, description="Ticker symbol to trade.")
    side: Literal["buy", "sell"] = Field(..., description="Order side.")
    order_type: str = Field(..., min_length=2, max_length=24, description="Order type, e.g. market or limit.")
    time_in_force: Optional[str] = Field(None, max_length=12, description="Time-in-force directive such as day or gtc.")
    qty: Optional[float] = Field(None, ge=0, description="Quantity of shares to trade.")
    notional: Optional[float] = Field(None, ge=0, description="Target notional in USD when quantity is omitted.")
    limit_price: Optional[float] = Field(None, description="Limit price for limit or stop-limit orders.")
    stop_price: Optional[float] = Field(None, description="Stop price for stop or stop-limit orders.")
    trail_price: Optional[float] = Field(None, description="Trailing stop price.")
    trail_percent: Optional[float] = Field(None, description="Trailing stop percentage.")
    take_profit: Optional[Dict[str, Union[float, int, str]]] = Field(
        None,
        description="Optional take-profit leg details (e.g. limit_price).",
    )
    stop_loss: Optional[Dict[str, Union[float, int, str]]] = Field(
        None,
        description="Optional stop-loss leg details (e.g. stop_price).",
    )
    notes: Optional[str] = Field(
        None,
        max_length=400,
        description="Additional human context about the order intent.",
    )


class OrderPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: List[OrderIntent] = Field(..., min_length=1, description="Orders to evaluate.")
    include_account: bool = Field(True, description="Include Alpaca account snapshot in the analysis context.")
    include_positions: bool = Field(True, description="Include open positions to highlight sizing overlaps.")
    include_open_orders: bool = Field(False, description="Include currently open orders to detect duplicates.")
    risk_notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Additional risk or compliance constraints for the model to respect.",
    )


class OrderPlanAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Order field to adjust, e.g. limit_price.")
    suggested_value: Optional[str] = Field(
        None, description="Suggested replacement value, expressed as a string."
    )
    rationale: str = Field(..., description="Reason for adjusting this field.")


class OrderPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Symbol under review.")
    side: Literal["buy", "sell"] = Field(..., description="Side of the proposed trade.")
    status: Literal["approve", "revise", "reject"] = Field(
        ..., description="High-level recommendation for the order."
    )
    summary: str = Field(..., description="Short natural-language summary of the guidance.")
    reasoning: List[str] = Field(
        default_factory=list, description="Key reasoning bullet points."
    )
    adjustments: List[OrderPlanAdjustment] = Field(
        default_factory=list, description="Structured adjustments before execution."
    )
    risk_flags: List[str] = Field(
        default_factory=list, description="Risks or guardrails to call out."
    )


class OrderPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_ready: bool = Field(..., description="Whether the orders can be sent as-is.")
    plan_summary: str = Field(..., description="Overall synopsis of the execution plan.")
    orders: List[OrderPlanItem] = Field(
        default_factory=list, description="Per-order recommendations."
    )
    account_notes: List[str] = Field(
        default_factory=list, description="Account-level callouts (margin, buying power, etc.)."
    )
    follow_up_tasks: List[str] = Field(
        default_factory=list, description="Actionable next steps for the assistant or trader."
    )

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
        _get_trading_http_client(),
        "POST",
        "/v2/orders",
        json=prepared,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body or "order rejected")
    return _decode_json(headers, body)


def _simplify_account_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keep = [
        "status",
        "equity",
        "cash",
        "portfolio_value",
        "buying_power",
        "multiplier",
        "pattern_day_trader",
        "shorting_enabled",
        "daytrade_count",
    ]
    snapshot = {field: data.get(field) for field in keep if data.get(field) not in (None, "", [])}
    if "cash" not in snapshot and data.get("cash_balance") not in (None, "", []):
        snapshot["cash"] = data.get("cash_balance")
    maintenance = data.get("maintenance_margin")
    if maintenance not in (None, "", []):
        snapshot.setdefault("maintenance_margin", maintenance)
    return snapshot


async def _fetch_account_snapshot() -> Dict[str, Any]:
    try:
        status, headers, body = await _request_with_retry(
            _get_trading_http_client(),
            "GET",
            "/v2/account",
        )
    except HTTPException as exc:
        return {"error": str(exc.detail), "status": exc.status_code}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"account_request_failed: {exc}"}

    if status >= 400:
        return {"error": body or f"HTTP {status}", "status": status}

    parsed = _decode_json(headers, body)
    snapshot = _simplify_account_payload(parsed)
    if snapshot:
        return snapshot
    return {"note": "account snapshot empty"}


def _simplify_position(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    keep = ["symbol", "qty", "avg_entry_price", "market_value", "side", "unrealized_pl", "unrealized_plpc"]
    simplified = {field: entry.get(field) for field in keep if entry.get(field) not in (None, "", [])}
    return simplified


async def _fetch_positions_snapshot(limit: int = 10) -> Dict[str, Any]:
    try:
        status, headers, body = await _request_with_retry(
            _get_trading_http_client(),
            "GET",
            "/v2/positions",
        )
    except HTTPException as exc:
        return {"items": [], "error": str(exc.detail), "status": exc.status_code}
    except Exception as exc:  # pragma: no cover - defensive
        return {"items": [], "error": f"positions_request_failed: {exc}"}

    if status >= 400:
        return {"items": [], "error": body or f"HTTP {status}", "status": status}

    parsed = _decode_json(headers, body)
    items: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for entry in parsed[:limit]:
            simplified = _simplify_position(entry)
            if simplified:
                items.append(simplified)
    return {"items": items}


def _simplify_open_order(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    keep = ["id", "symbol", "side", "qty", "notional", "type", "time_in_force", "status", "limit_price", "stop_price", "submitted_at"]
    simplified = {field: entry.get(field) for field in keep if entry.get(field) not in (None, "", [])}
    legs = entry.get("legs")
    if isinstance(legs, list):
        leg_summaries = []
        for leg in legs[:3]:
            leg_summary = {
                "symbol": leg.get("symbol"),
                "side": leg.get("side"),
                "qty": leg.get("qty"),
                "type": leg.get("type"),
                "limit_price": leg.get("limit_price"),
                "status": leg.get("status"),
            }
            filtered_leg = {k: v for k, v in leg_summary.items() if v not in (None, "", [])}
            if filtered_leg:
                leg_summaries.append(filtered_leg)
        if leg_summaries:
            simplified["legs"] = leg_summaries
    return simplified


async def _fetch_open_orders_snapshot(limit: int = 10) -> Dict[str, Any]:
    params = {"status": "open", "nested": "true"}
    try:
        status, headers, body = await _request_with_retry(
            _get_trading_http_client(),
            "GET",
            "/v2/orders",
            params=params,
        )
    except HTTPException as exc:
        return {"items": [], "error": str(exc.detail), "status": exc.status_code}
    except Exception as exc:  # pragma: no cover - defensive
        return {"items": [], "error": f"open_orders_request_failed: {exc}"}

    if status >= 400:
        return {"items": [], "error": body or f"HTTP {status}", "status": status}

    parsed = _decode_json(headers, body)
    items: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for entry in parsed[:limit]:
            simplified = _simplify_open_order(entry)
            if simplified:
                items.append(simplified)
    return {"items": items}


async def _collect_order_plan_context(payload: OrderPlanRequest) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "orders": [order.model_dump(exclude_none=True) for order in payload.orders],
    }
    if payload.risk_notes:
        context["risk_notes"] = payload.risk_notes

    tasks: List[asyncio.Task[Any]] = []
    account_task = positions_task = open_orders_task = None

    if payload.include_account:
        account_task = asyncio.create_task(_fetch_account_snapshot())
        tasks.append(account_task)
    if payload.include_positions:
        positions_task = asyncio.create_task(_fetch_positions_snapshot())
        tasks.append(positions_task)
    if payload.include_open_orders:
        open_orders_task = asyncio.create_task(_fetch_open_orders_snapshot())
        tasks.append(open_orders_task)

    if tasks:
        await asyncio.gather(*tasks)

    if account_task:
        context["account"] = account_task.result()
    if positions_task:
        context["positions"] = positions_task.result()
    if open_orders_task:
        context["open_orders"] = open_orders_task.result()

    return context


async def _call_order_plan_model(context: Dict[str, Any]) -> OrderPlanResponse:
    client = _get_openai_client()
    system_prompt = dedent(
        """
        You are a trading operations specialist assisting with Alpaca order intake.
        Review the proposed orders, account snapshot, open positions, and any open orders.
        Respond strictly with the provided JSON schema, focusing on compliance, sizing,
        margin usage, and duplicate or conflicting orders. Keep reasoning concise and
        guidance actionable.
        """
    ).strip()

    payload_text = json.dumps(context, ensure_ascii=False)
    schema_json = json.dumps(OrderPlanResponse.model_json_schema(), ensure_ascii=False)

    last_validation_error: ValidationError | None = None
    parsed: Any | None = None
    for attempt in range(2):
        user_content = [
            {"type": "input_text", "text": "JSON schema (strict): " + schema_json},
            {"type": "input_text", "text": "Evaluate the proposed Alpaca orders and respond with the JSON schema."},
            {"type": "input_text", "text": payload_text},
        ]
        if attempt:
            user_content.insert(0, {"type": "input_text", "text": "Retry: the last response was invalid JSON. Return only valid JSON that matches the schema."})
        try:
            parsed = await client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                max_output_tokens=1100,
                text_format=OrderPlanResponse,
            )
            break
        except ValidationError as exc:
            last_validation_error = exc
            if attempt == 1:
                raise HTTPException(status_code=502, detail=f"OpenAI payload validation failed: {exc}") from exc
            continue
        except APIConnectionError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI connection error: {exc}") from exc
        except APIStatusError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc.status_code}") from exc
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

    if parsed is None:
        raise HTTPException(status_code=502, detail=f"OpenAI payload validation failed: {last_validation_error}") from last_validation_error

    if isinstance(parsed, OrderPlanResponse):
        return parsed
    try:
        return OrderPlanResponse.model_validate(parsed)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"OpenAI payload validation failed: {exc}") from exc
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
    trading_ready = isinstance(getattr(app.state, "trading_http_client", None) or getattr(app.state, "http_client", None), EdgeHttpClient)
    data_ready = isinstance(getattr(app.state, "data_http_client", None), EdgeHttpClient)
    creds_ok = _alpaca_credentials_present()
    dependencies = {
        "trading_http_client": trading_ready,
        "data_http_client": data_ready,
        "alpaca_credentials": creds_ok,
        "api_base_url": bool(API_BASE_URL),
        "data_api_base_url": bool(DATA_BASE_URL),
    }
    return {
        "ok": all(dependencies.values()),
        "dependencies": dependencies,
    }



# -- Orders
@app.post("/v2/orders/sync")
async def order_create_sync(
    order: CreateOrder,
    request: Request,
):
    """Synchronously mirror `/v2/orders` in the current thread.

    Applies the limit guard (extended-hours must be DAY limit orders)
    and surfaces upstream HTTP 429 responses with their `Retry-After`.
    """
    _require_gateway_key_from_request(request)
    payload = _order_payload_from_model(order)
    return await _submit_order_async(payload)




@app.post("/v2/orders")
async def order_create(
    order: CreateOrder,
    request: Request,
):
    """Submit an order via Alpaca's `/v2/orders`.

    Requires `X-API-Key`, forwards 429 responses with `Retry-After`, and
    applies the limit guard (extended-hours must be DAY limit orders).
    """
    _require_gateway_key_from_request(request)
    payload = _order_payload_from_model(order)
    return await _submit_order_async(payload)



@app.post(
    "/analysis/order-plan",
    response_model=OrderPlanResponse,
    summary="Review proposed Alpaca orders with GPT-5",
    description="Use GPT-5 Structured Output to review proposed orders and return a JSON execution plan with adjustments and risk notes.",
)
async def analyse_order_plan(payload: OrderPlanRequest, request: Request) -> OrderPlanResponse:
    _require_gateway_key_from_request(request)
    context = await _collect_order_plan_context(payload)
    result = await _call_order_plan_model(context)
    return result


@app.get("/v2/account")
async def account_get(request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(_get_trading_http_client(), "GET", "/v2/account")
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)



@app.get("/v2/account/activities")
async def account_activities(
    request: Request,
    activity_type: Optional[str] = None,
    date: Optional[str] = None,
    until: Optional[str] = None,
    after: Optional[str] = None,
    direction: Optional[str] = None,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
):
    _require_gateway_key_from_request(request)
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
        _get_trading_http_client(),
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
async def orders_list(request: Request):
    _require_gateway_key_from_request(request)
    params = dict(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/orders",
        params=params or None,
    )
    return _passthrough_json(status, headers, body)



@app.get("/v2/orders/{order_id}")
async def orders_get_by_id(order_id: str, request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/orders/{order_id}",
    )
    return _passthrough_json(status, headers, body)



@app.delete("/v2/orders")
async def orders_cancel_all(request: Request):
    """Cancel all open orders."""
    return await _proxy_alpaca_request(
        "DELETE",
        "/v2/orders",
        request,
    )



@app.delete("/v2/orders/{order_id}")
async def orders_cancel_by_id(order_id: str, request: Request):
    """Cancel a specific order by id."""
    return await _proxy_alpaca_request(
        "DELETE",
        f"/v2/orders/{order_id}",
        request,
    )



@app.get("/v2/positions")
async def positions_list_v2(request: Request):
    _require_gateway_key_from_request(request)
    params = dict(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/positions",
        params=params or None,
    )
    return _passthrough_json(status, headers, body)



@app.get("/v2/positions/{symbol}")
async def positions_get(symbol: str, request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/positions/{symbol}",
    )
    return _passthrough_json(status, headers, body)



@app.delete("/v2/positions")
async def positions_close_all(
    request: Request,
    cancel_orders: bool = Query(False),
):
    _require_gateway_key_from_request(request)
    params = {"cancel_orders": str(cancel_orders).lower()}
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "DELETE",
        "/v2/positions",
        params=params,
    )
    return _passthrough_json(status, headers, body)



@app.delete("/v2/positions/{symbol}")
async def positions_close(
    symbol: str,
    request: Request,
    cancel_orders: bool = Query(False),
):
    _require_gateway_key_from_request(request)
    params = {"cancel_orders": str(cancel_orders).lower()}
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "DELETE",
        f"/v2/positions/{symbol}",
        params=params,
    )
    return _passthrough_json(status, headers, body)



async def _proxy_alpaca_request(
    method: str,
    path: str,
    request: Request,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _http_client_for_path(path),
        method,
        path,
        json=payload,
        params=params,
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
async def watchlists_list_v2(request: Request):
    _require_gateway_key_from_request(request)
    params = dict(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/watchlists",
        params=params or None,
    )
    return _passthrough_json(status, headers, body)



@app.post("/v2/watchlists")
async def watchlists_create_v2(
    watchlist: WatchlistIn,
    request: Request,
):
    return await _proxy_alpaca_request(
        "POST",
        "/v2/watchlists",
        request,
        payload=watchlist.model_dump(),
    )




@app.get("/v2/watchlists/{watchlist_id}")
async def watchlists_get_v2(watchlist_id: str, request: Request):
    _require_gateway_key_from_request(request)
    params = dict(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/watchlists/{watchlist_id}",
        params=params or None,
    )
    return _passthrough_json(status, headers, body)



@app.put("/v2/watchlists/{watchlist_id}")
async def watchlists_update_v2(
    watchlist_id: str,
    watchlist: WatchlistIn,
    request: Request,
):
    return await _proxy_alpaca_request(
        "PUT",
        f"/v2/watchlists/{watchlist_id}",
        request,
        payload=watchlist.model_dump(),
    )



@app.delete("/v2/watchlists/{watchlist_id}")
async def watchlists_delete_v2(watchlist_id: str, request: Request):
    return await _proxy_alpaca_request(
        "DELETE", f"/v2/watchlists/{watchlist_id}", request
    )




# -- Market Data
@app.get("/v2/quotes")
def get_quotes_v2(
    request: Request,
    symbol: str,
    start: str,
    end: str,
) -> list[dict]:
    _require_gateway_key_from_request(request)
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
    request: Request,
    symbol: str,
    start: str,
    end: str,
) -> list[dict]:
    _require_gateway_key_from_request(request)
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
    request: Request,
    symbol: str,
    timeframe: str = Query("1Day"),
    start: str = Query(...),
    end: Optional[str] = None,
) -> list[dict]:
    _require_gateway_key_from_request(request)
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
AI_PLUGIN_PATH = Path(__file__).resolve().parent / ".well-known" / "ai-plugin.json"


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
def ai_plugin_manifest() -> FileResponse:
    if AI_PLUGIN_PATH.exists():
        return FileResponse(str(AI_PLUGIN_PATH), media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")


# --- ChatGPT plugin support ---

cors_origins_env = os.getenv("EDGE_CORS_ALLOW_ORIGINS", "")
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = ["https://chat.openai.com"]

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# --- GPT Actions OpenAPI patch ---

def _build_openapi_schema(routes) -> Dict[str, Any]:
    schema = get_openapi(
        title="Alpaca Wrapper",
        version="1.1.0",
        routes=routes,
    )
    schema["openapi"] = "3.1.0"
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
    return schema



def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = _build_openapi_schema(app.routes)
    schema["servers"] = [{"url": _default_server_url()}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]


# --- end patch ---



