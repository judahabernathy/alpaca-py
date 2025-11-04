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
from typing import Any, Dict, List, Optional, Union, cast, Literal, Tuple, AsyncGenerator
from uuid import uuid4
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, Header, HTTPException, Query, Request, status, Body
from fastapi.params import Param
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, FileResponse
from starlette.responses import EventSourceResponse
import anyio
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


def _read_prompt_variables_env(var_name: str) -> Optional[Dict[str, Any]]:
    raw = os.getenv(var_name)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log_error(
            "prompt_variables_env_invalid",
            {
                "env_var": var_name,
                "reason": "json_decode",
            },
        )
        return None
    if not isinstance(parsed, dict):
        log_error(
            "prompt_variables_env_invalid",
            {
                "env_var": var_name,
                "reason": "not_mapping",
            },
        )
        return None
    return parsed


def _build_default_order_plan_prompt() -> Optional[Dict[str, Any]]:
    prompt_id = (os.getenv("ORDER_PLAN_PROMPT_ID") or "").strip()
    if not prompt_id:
        _read_prompt_variables_env("ORDER_PLAN_PROMPT_VARIABLES")
        return None
    prompt: Dict[str, Any] = {"id": prompt_id}
    prompt_version = (os.getenv("ORDER_PLAN_PROMPT_VERSION") or "").strip()
    if prompt_version:
        prompt["version"] = prompt_version
    variables = _read_prompt_variables_env("ORDER_PLAN_PROMPT_VARIABLES")
    if variables:
        prompt["variables"] = variables
    return prompt

configure_logging()

_ORDER_PLAN_PROMPT_DEFAULT = _build_default_order_plan_prompt()


class OrderReject(Exception):
    def __init__(
        self,
        *,
        reason_code: str,
        detail: str,
        title: str = "Order rejected",
        status: int = 409,
        context: Optional[Dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail
        self.title = title
        self.status = status
        self.context = context or {}

class OrderRejectResponse(BaseModel):
    type: str = Field(default='https://errors.alpaca/ORDER_REJECTED')
    title: str = 'Order rejected'
    status: Literal[409] = 409
    detail: str
    reason_code: str
    context: Optional[Dict[str, Any]] = None




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


@app.exception_handler(OrderReject)
async def order_reject_handler(_: Request, exc: OrderReject) -> JSONResponse:
    problem = {
        "type": f"https://errors.alpaca/{exc.reason_code}",
        "title": exc.title,
        "status": exc.status,
        "detail": exc.detail,
        "reason_code": exc.reason_code,
    }
    if exc.context:
        problem["context"] = exc.context
    return JSONResponse(problem, status_code=exc.status, media_type="application/problem+json")

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
        raise OrderReject(reason_code="INVALID_LIMIT_PRICE", detail="Limit price must be numeric.")

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
        detail = f"Quote age {age * 1000:.0f}ms exceeds ttl {ttl_seconds * 1000:.0f}ms."
        raise OrderReject(
            reason_code="QUOTE_TOO_OLD",
            detail=detail,
            context={"age_seconds": age, "ttl_seconds": ttl_seconds, "feed": feed},
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
        drift_bps_actual = abs(drift) * 10000.0
        max_bps = max_drift * 10000.0
        message = f"Drift {drift_bps_actual:.1f}bps exceeds max {max_bps:.1f}bps."
        raise OrderReject(
            reason_code="BIDASK_DRIFT",
            detail=message,
            context={"drift": drift, "max_drift": max_drift, "side": normalised_side},
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
            raise OrderReject(reason_code="EXT_INVALID_TYPE", detail="Extended hours orders must be limit orders.")
        if tif != "day":
            raise OrderReject(reason_code="EXT_INVALID_TIF", detail="Extended hours limit orders must be DAY.")
        if payload.get("order_class"):
            raise OrderReject(reason_code="EXT_UNSUPPORTED_CLASS", detail="Extended hours does not support advanced order classes.")
        if payload.get("limit_price") is None:
            raise OrderReject(reason_code="EXT_MISSING_LIMIT", detail="Extended hours limit orders require limit_price.")

    if order_type == "limit" and payload.get("limit_price") is not None:
        try:
            evaluate_limit_guard((payload.get("symbol") or ""), payload.get("limit_price"), (payload.get("side") or ""))
        except OrderReject:
            raise
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
        if lower in {"content-length", "content-type", "content-encoding"}:
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



class TakeProfitLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit_price: float = Field(
        ...,
        description="Limit price for the take-profit child leg.",
    )


class StopLossLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_price: float = Field(
        ...,
        description="Trigger price for the stop-loss child leg (e.g., 225.0).",
    )
    limit_price: Optional[float] = Field(
        None,
        description=(
            "Optional limit price for a stop-limit child leg. "
            "Omit to use a market stop; set slightly below the stop (e.g., 224.0) when a limit is required."
        ),
    )

class OrderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=12, description="Ticker symbol to trade.")
    side: Literal["buy", "sell"] = Field(..., description="Order side.")
    order_type: str = Field(..., min_length=2, max_length=24, description="Order type, e.g. market or limit.")
    time_in_force: Optional[str] = Field(None, max_length=12, description="Time-in-force directive such as day or gtc.")
    qty: Optional[float] = Field(None, ge=0, description="Quantity of shares to trade.")
    notional: Optional[float] = Field(None, ge=0, description="Target notional in USD when quantity is omitted.")
    limit_price: Optional[float] = Field(None, description="Limit price for limit or stop-limit orders.")
    stop_price: Optional[float] = Field(None, description="Top-level stop price for standalone stop or stop-limit orders.")
    trail_price: Optional[float] = Field(None, description="Trailing stop price (ignored unless type is trailing_stop).")
    trail_percent: Optional[float] = Field(None, description="Trailing stop percentage (ignored unless type is trailing_stop).")
    take_profit: Optional[TakeProfitLeg] = Field(
        None,
        description="Optional take-profit child order details.",
    )
    stop_loss: Optional[StopLossLeg] = Field(
        None,
        description="Optional stop-loss child order details.",
    )
    notes: Optional[str] = Field(
        None,
        max_length=400,
        description="Additional human context about the order intent.",
    )




class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(
        None,
        min_length=1,
        description="Prompt template identifier registered with OpenAI.",
    )
    version: Optional[str] = Field(
        None,
        min_length=1,
        description="Prompt template version to request.",
    )
    variables: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional variable substitutions to inject into the prompt template.",
    )


class PromptMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Prompt template identifier.")
    version: Optional[str] = Field(
        None,
        min_length=1,
        description="Prompt template version.",
    )
    variables: Optional[Dict[str, Any]] = Field(
        None,
        description="Default variables supplied with the prompt template.",
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


prompt: Optional[PromptTemplate] = Field(
    None,
    description="Optional reusable prompt template reference for the GPT call.",
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
    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: str
    qty: Optional[float] = None
    notional: Optional[float] = None
    type: str
    time_in_force: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_class: Optional[str] = None
    take_profit: Optional[TakeProfitLeg] = None
    stop_loss: Optional[StopLossLeg] = None
    extended_hours: Optional[bool] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: Optional[str] = None


class WatchlistIn(BaseModel):
    name: str
    symbols: List[str]


class AccountConfigurationUpdate(BaseModel):
    """Matches Alpaca account configuration fields (Oct 2025)."""

    model_config = ConfigDict(extra="allow")

    dtbp_check: Optional[str] = Field(
        default=None,
        description="Day-trade buying power check enforcement (none, entry, exit, both).",
    )
    no_shorting: Optional[bool] = Field(
        default=None, description="Disable all short sales when true."
    )
    suspend_trade: Optional[bool] = Field(
        default=None, description="If true, new orders are blocked at the account level."
    )
    trade_confirm_email: Optional[str] = Field(
        default=None,
        description="Email preference for order confirmations (all, none, trade_activity).",
    )
    fractional_trading: Optional[bool] = Field(
        default=None, description="Enable fractional share trading."
    )
    max_margin_multiplier: Optional[str] = Field(
        default=None, description="Maximum intraday margin multiplier (e.g. '2', '4')."
    )
    pdt_check: Optional[str] = Field(
        default=None,
        description="Pattern day trading check behaviour (enforced, bypassed, entry_only).",
    )


class WatchlistEntryPatch(BaseModel):
    """Payload for adding an asset to an existing watchlist."""

    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = Field(
        default=None,
        description="Ticker symbol to add (symbol or asset_id required by Oct 2025 docs).",
        min_length=1,
    )
    asset_id: Optional[str] = Field(
        default=None,
        description="UUID of the asset when symbol lookup is ambiguous.",
        min_length=1,
    )

    def model_post_init(self, __context: Any) -> None:
        if not (self.symbol or self.asset_id):
            raise ValueError("Either symbol or asset_id is required")


def _order_payload_from_model(order: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(order, BaseModel):
        return order.model_dump(exclude_none=True)
    return {k: v for k, v in dict(order).items() if v is not None}


def _submit_order_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return anyio.from_thread.run(_submit_order_async, payload)
    except RuntimeError:
        return asyncio.run(_submit_order_async(payload))


def _prepare_order_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalised = deepcopy(payload)

    def _clean_leg(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            cleaned = {k: v for k, v in value.items() if v is not None}
            return cleaned or None
        return None

    take_profit = _clean_leg(normalised.get("take_profit"))
    if take_profit:
        normalised["take_profit"] = take_profit
    else:
        normalised.pop("take_profit", None)

    stop_loss = _clean_leg(normalised.get("stop_loss"))
    if stop_loss:
        normalised["stop_loss"] = stop_loss
    else:
        normalised.pop("stop_loss", None)

    order_type = str(normalised.get("type") or "").lower()
    if order_type != "trailing_stop":
        normalised.pop("trail_price", None)
        normalised.pop("trail_percent", None)
    else:
        if normalised.get("trail_price") in (None, 0):
            normalised.pop("trail_price", None)
        if normalised.get("trail_percent") in (None, 0):
            normalised.pop("trail_percent", None)

    if stop_loss:
        def _to_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        top_level_stop = _to_float(normalised.get("stop_price"))
        stop_loss_stop = _to_float(stop_loss.get("stop_price"))
        if order_type not in {"stop", "stop_limit"} or (
            top_level_stop is not None and stop_loss_stop is not None and abs(top_level_stop - stop_loss_stop) < 1e-9
        ):
            normalised.pop("stop_price", None)
    else:
        if normalised.get("order_class") in {"bracket", "oco", "oto"}:
            normalised.pop("stop_price", None)

    _enforce_ext_policy(normalised)
    normalised["client_order_id"] = _resolve_client_order_id(normalised.get("client_order_id"))
    return normalised

def _normalise_reason_code(value: Optional[str]) -> str:
    if not value:
        return "ORDER_REJECTED"
    cleaned = re.sub(r'[^A-Za-z0-9]+', '_', str(value)).strip('_')
    return cleaned.upper() or "ORDER_REJECTED"


def _order_reject_from_response(status: int, body: str) -> OrderReject:
    detail = body or "order rejected"
    context: Dict[str, Any] = {}
    source = None
    if body:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        else:
            if isinstance(data, dict):
                detail = str(data.get('detail') or data.get('message') or detail)
                source = data.get('reason_code') or data.get('code') or data.get('reason')
                context = {k: v for k, v in data.items() if k not in {'detail', 'message', 'reason_code', 'code', 'reason'}}
    reason_code = _normalise_reason_code(source)
    return OrderReject(reason_code=reason_code, detail=detail, status=status, context=context or None)



async def _submit_order_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _prepare_order_payload(payload)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "POST",
        "/v2/orders",
        json=prepared,
    )
    if status == 409:
        raise _order_reject_from_response(status, body)
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



def _resolve_order_plan_prompt(request_prompt: Optional[PromptTemplate]) -> Optional[Dict[str, Any]]:
    resolved: Dict[str, Any] = {}
    default_prompt = _ORDER_PLAN_PROMPT_DEFAULT or {}
    if default_prompt:
        resolved.update(default_prompt)
    user_override: Dict[str, Any] = {}
    if request_prompt is not None:
        user_override = request_prompt.model_dump(exclude_none=True)
    default_variables = default_prompt.get("variables") if isinstance(default_prompt, dict) else None
    user_variables = user_override.pop("variables", None)
    merged_variables: Dict[str, Any] = {}
    if isinstance(default_variables, dict):
        merged_variables.update(default_variables)
    if isinstance(user_variables, dict):
        merged_variables.update(user_variables)
    if merged_variables:
        resolved["variables"] = merged_variables
    elif "variables" in resolved:
        resolved.pop("variables", None)
    resolved.update(user_override)
    if not resolved.get("id"):
        return None
    return resolved



async def _call_order_plan_model(
    context: Dict[str, Any],
    prompt: Optional[Dict[str, Any]] = None,
) -> OrderPlanResponse:
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
    base_user_content = [
        {"type": "input_text", "text": "JSON schema (strict): " + schema_json},
        {
            "type": "input_text",
            "text": "Evaluate the proposed Alpaca orders and respond with the JSON schema.",
        },
        {"type": "input_text", "text": payload_text},
    ]

    for attempt in range(2):
        user_content = list(base_user_content)
        if attempt:
            user_content.insert(
                0,
                {
                    "type": "input_text",
                    "text": "Retry: the last response was invalid JSON. Return only valid JSON that matches the schema.",
                },
            )
        try:
            kwargs: Dict[str, Any] = {
                "model": OPENAI_MODEL,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                "max_output_tokens": 1800,
                "text_format": OrderPlanResponse,
            }
            if prompt:
                kwargs["prompt"] = prompt
            parsed = await client.responses.parse(**kwargs)
            break
        except ValidationError as exc:
            last_validation_error = exc
            if attempt == 1:
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenAI payload validation failed: {exc}",
                ) from exc
            continue
        except APIConnectionError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI connection error: {exc}") from exc
        except APIStatusError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc.status_code}") from exc
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

    if parsed is None:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI payload validation failed: {last_validation_error}",
        ) from last_validation_error

    return _coerce_order_plan_response(parsed)



def _coerce_order_plan_response(parsed: Any) -> OrderPlanResponse:
    if isinstance(parsed, OrderPlanResponse):
        return parsed
    candidate = getattr(parsed, "parsed", None)
    if isinstance(candidate, OrderPlanResponse):
        return candidate
    if hasattr(parsed, "output"):
        for item in getattr(parsed, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                candidate = getattr(content, "parsed", None)
                if isinstance(candidate, OrderPlanResponse):
                    return candidate
    try:
        return OrderPlanResponse.model_validate(parsed)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"OpenAI payload validation failed: {exc}") from exc


async def _iter_order_plan_text(stream: Any) -> AsyncGenerator[str, None]:
    async for event in stream:
        if getattr(event, "type", "") == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if delta:
                yield str(delta)


async def _stream_order_plan_model(
    context: Dict[str, Any],
    prompt: Optional[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, str], None]:
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
    user_content = [
        {"type": "input_text", "text": "JSON schema (strict): " + schema_json},
        {
            "type": "input_text",
            "text": "Evaluate the proposed Alpaca orders and respond with the JSON schema.",
        },
        {"type": "input_text", "text": payload_text},
    ]

    stream_kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": 1800,
        "text_format": OrderPlanResponse,
    }
    if prompt:
        stream_kwargs["prompt"] = prompt

    has_yielded = False

    try:
        async with client.responses.stream(**stream_kwargs) as stream:
            async for chunk in _iter_order_plan_text(stream):
                has_yielded = True
                yield {"event": "delta", "data": chunk}
            final = await stream.get_final_response()
    except ValidationError as exc:
        detail = f"OpenAI payload validation failed: {exc}"
        if has_yielded:
            yield {"event": "error", "data": detail}
            return
        raise HTTPException(status_code=502, detail=detail) from exc
    except APIConnectionError as exc:
        detail = f"OpenAI connection error: {exc}"
        if has_yielded:
            yield {"event": "error", "data": detail}
            return
        raise HTTPException(status_code=502, detail=detail) from exc
    except APIStatusError as exc:
        detail = f"OpenAI API error: {exc.status_code}"
        if has_yielded:
            yield {"event": "error", "data": detail}
            return
        raise HTTPException(status_code=502, detail=detail) from exc
    except OpenAIError as exc:
        detail = f"OpenAI error: {exc}"
        if has_yielded:
            yield {"event": "error", "data": detail}
            return
        raise HTTPException(status_code=502, detail=detail) from exc
    else:
        try:
            result = _coerce_order_plan_response(final)
        except HTTPException as exc:
            detail = str(exc.detail)
            if has_yielded:
                yield {"event": "error", "data": detail}
                return
            raise
        yield {"event": "result", "data": result.model_dump_json()}





@app.post(
    "/analysis/order-plan",
    response_model=OrderPlanResponse,
    summary="Review proposed Alpaca orders with GPT-5",
    description="Use GPT-5 Structured Output to review proposed orders and return a JSON execution plan with adjustments and risk notes.",
)
async def analyse_order_plan(
    payload: OrderPlanRequest,
    request: Request,
    stream: bool = Query(
        False,
        description="Stream the GPT analysis via server-sent events when true.",
    ),
) -> Union[OrderPlanResponse, EventSourceResponse]:
    _require_gateway_key_from_request(request)
    context = await _collect_order_plan_context(payload)
    prompt_payload = _resolve_order_plan_prompt(payload.prompt)
    if stream:
        return EventSourceResponse(
            _stream_order_plan_model(context, prompt_payload),
            media_type="text/event-stream",
        )
    result = await _call_order_plan_model(context, prompt_payload)
    return result


@app.get(
    "/metadata/prompts",
    response_model=Dict[str, PromptMetadata],
    summary="List reusable prompt templates",
    description="Surface the prompt template identifiers and versions used by the edge service.",
)
async def list_prompt_metadata() -> Dict[str, PromptMetadata]:
    defaults = _ORDER_PLAN_PROMPT_DEFAULT or {}
    prompt_id = defaults.get("id") or "order_plan.default"
    metadata = PromptMetadata(
        id=prompt_id,
        version=defaults.get("version"),
        variables=defaults.get("variables"),
    )
    return {"order_plan": metadata}



def _serialise_query_params(values: Any) -> Optional[Union[Dict[str, str], List[Tuple[str, str]]]]:
    """Preserve duplicate query keys when forwarding upstream."""
    if values is None:
        return None
    items: List[Tuple[str, str]] = []
    multi_items = getattr(values, 'multi_items', None)
    if callable(multi_items):
        items = [(str(k), str(v)) for k, v in multi_items()]
    elif isinstance(values, dict):
        items = [(str(k), str(v)) for k, v in values.items()]
    else:
        try:
            items = [(str(k), str(v)) for k, v in values]
        except Exception:
            items = []
    if not items:
        return None
    has_duplicates = len({k for k, _ in items}) != len(items)
    if has_duplicates:
        return items
    return {k: v for k, v in items}


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
@app.post("/v2/orders/sync", responses={409: {"model": OrderRejectResponse, "description": "Business rule rejection"}})
def order_create_sync(
    order: CreateOrder,
    request: Request,
):
    """Synchronously mirror `/v2/orders` in the current thread.

    Applies the limit guard (extended-hours must be DAY limit orders)
    and surfaces upstream HTTP 429 responses with their `Retry-After`.
    """
    _require_gateway_key_from_request(request)
    payload = _order_payload_from_model(order)
    return _submit_order_sync(payload)




@app.post("/v2/orders", responses={409: {"model": OrderRejectResponse, "description": "Business rule rejection"}})
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
async def analyse_order_plan(
    payload: OrderPlanRequest,
    request: Request,
    stream: bool = Query(
        False,
        description="Stream the GPT analysis via server-sent events when true.",
    ),
) -> Union[OrderPlanResponse, EventSourceResponse]:
    _require_gateway_key_from_request(request)
    context = await _collect_order_plan_context(payload)
    prompt_payload = _resolve_order_plan_prompt(payload.prompt)
    if stream:
        return EventSourceResponse(
            _stream_order_plan_model(context, prompt_payload),
            media_type="text/event-stream",
        )
    result = await _call_order_plan_model(context, prompt_payload)
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


@app.get("/v2/account/configurations")
async def account_configurations_get(request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/account/configurations",
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)

@app.patch("/v2/account/configurations")
async def account_configurations_patch(payload: AccountConfigurationUpdate, request: Request):
    _require_gateway_key_from_request(request)
    primary = payload.model_dump(exclude_none=True)
    extras = getattr(payload, 'model_extra', None) or {}
    body_payload = {**extras, **primary}
    if not body_payload:
        raise HTTPException(status_code=400, detail="At least one configuration field must be provided.")
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "PATCH",
        "/v2/account/configurations",
        json=body_payload,
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return _decode_json(headers, body)

@app.get("/v2/account/portfolio/history")
async def portfolio_history(request: Request):
    _require_gateway_key_from_request(request)
    params = _serialise_query_params(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/account/portfolio/history",
        params=params,
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/assets")
async def assets_list(request: Request):
    _require_gateway_key_from_request(request)
    params = _serialise_query_params(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/assets",
        params=params,
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/assets/{asset_id_or_symbol}")
async def assets_get(asset_id_or_symbol: str, request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/assets/{asset_id_or_symbol}",
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/calendar")
async def trading_calendar(request: Request):
    _require_gateway_key_from_request(request)
    params = _serialise_query_params(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/calendar",
        params=params,
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/clock")
async def market_clock(request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/clock",
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/corporate_actions/announcements")
async def corporate_actions_announcements(request: Request):
    _require_gateway_key_from_request(request)
    params = _serialise_query_params(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/corporate_actions/announcements",
        params=params,
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/corporate_actions/announcements/{announcement_id}")
async def corporate_actions_announcement_get(announcement_id: str, request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/corporate_actions/announcements/{announcement_id}",
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/options/contracts")
async def options_contracts(request: Request):
    _require_gateway_key_from_request(request)
    params = _serialise_query_params(request.query_params)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/options/contracts",
        params=params,
    )
    return _passthrough_json(status, headers, body)

@app.get("/v2/options/contracts/{contract_id}")
async def options_contract_get(contract_id: str, request: Request):
    _require_gateway_key_from_request(request)
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        f"/v2/options/contracts/{contract_id}",
    )
    return _passthrough_json(status, headers, body)






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



@app.get("/v2/orders:by_client_order_id")
async def orders_by_client_order_id(
    request: Request,
    client_order_id: str = Query(..., description="Client order identifier as documented in Alpaca Trading API (Oct 2025)."),
) -> Response:
    _require_gateway_key_from_request(request)
    params = {"client_order_id": client_order_id}
    status, headers, body = await _request_with_retry(
        _get_trading_http_client(),
        "GET",
        "/v2/orders:by_client_order_id",
        params=params,
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



@app.post("/v2/positions/{symbol_or_contract_id}/exercise")
async def positions_exercise(symbol_or_contract_id: str, request: Request, payload: Optional[Dict[str, Any]] = Body(None)):
    _require_gateway_key_from_request(request)
    body = payload or {}
    status, headers, response_body = await _request_with_retry(
        _get_trading_http_client(),
        "POST",
        f"/v2/positions/{symbol_or_contract_id}/exercise",
        json=body or None,
    )
    return _passthrough_json(status, headers, response_body)


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



@app.post("/v2/watchlists/{watchlist_id}")
async def watchlists_add_asset_v2(watchlist_id: str, entry: WatchlistEntryPatch, request: Request):
    payload = entry.model_dump(exclude_none=True)
    return await _proxy_alpaca_request(
        "POST",
        f"/v2/watchlists/{watchlist_id}",
        request,
        payload=payload,
    )


@app.delete("/v2/watchlists/{watchlist_id}/{symbol}")
async def watchlists_remove_asset_v2(watchlist_id: str, symbol: str, request: Request):
    return await _proxy_alpaca_request(
        "DELETE",
        f"/v2/watchlists/{watchlist_id}/{symbol}",
        request,
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


@app.get("/.well-known/openapi.json", include_in_schema=False)
def well_known_openapi() -> JSONResponse:
    return JSONResponse(app.openapi())


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

EXCLUDED_OPENAPI_OPERATIONS: Tuple[Tuple[str, str], ...] = (
    ("/v2/orders/sync", "post"),
    ("/v2/account/portfolio/history", "get"),
    ("/v2/corporate_actions/announcements/{announcement_id}", "get"),
    ("/v2/positions/{symbol_or_contract_id}/exercise", "post"),
    ("/v2/watchlists/{watchlist_id}", "post"),
    ("/v2/watchlists/{watchlist_id}", "delete"),
        ("/v2/watchlists/{watchlist_id}/{symbol}", "delete"),
    ("/v2/options/contracts", "get"),
    ("/v2/options/contracts/{contract_id}", "get"),
)


def _prune_openapi_operations(schema: Dict[str, Any]) -> None:
    paths = schema.get("paths", {})
    for path, method in EXCLUDED_OPENAPI_OPERATIONS:
        path_item = paths.get(path)
        if not isinstance(path_item, dict):
            continue
        path_item.pop(method, None)
        remaining = [name for name in path_item if not name.startswith("x-")]
        if not remaining:
            paths.pop(path, None)


def _build_openapi_schema(routes) -> Dict[str, Any]:
    schema = get_openapi(
        title="Alpaca Wrapper",
        version="1.1.0",
        routes=routes,
    )
    schema["openapi"] = "3.1.0"
    schema["jsonSchemaDialect"] = "https://json-schema.org/draft/2020-12/schema"

    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["EdgeApiKey"] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    for deprecated_scheme in ("ApiKeyAuth", "alpacaKey", "alpacaSecret"):
        security_schemes.pop(deprecated_scheme, None)
    schema["security"] = [{"EdgeApiKey": []}]

    info = schema.setdefault("info", {})
    info["description"] = dedent("""
        All endpoints require the `X-API-Key` header; the `api_key` query parameter is
        supported as a compatibility fallback. The service propagates Alpaca's rate
        limits as HTTP 429 responses and surfaces the upstream `Retry-After` header.
        When `extended_hours` is enabled the order payload must remain a DAY limit
        order, include `limit_price`, and omit advanced `order_class` values.
    """).strip()

    info.setdefault("license", {"name": "Proprietary", "url": "https://alpaca-py-production.up.railway.app/legal"})

    paths = schema.setdefault("paths", {})

    for path, method in (("/v2/orders", "post"),):
        operation = paths.get(path, {}).get(method)
        if isinstance(operation, dict):
            responses = operation.get("responses") or {}
            problem = responses.get("409")
            if isinstance(problem, dict):
                content = problem.setdefault("content", {})
                payload = content.pop("application/json", None)
                if payload is not None:
                    content["application/problem+json"] = payload
    for path_item in paths.values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            parameters = operation.get("parameters")
            if not parameters:
                continue
            filtered = [param for param in parameters if param.get("name") != "X-API-Key"]
            if filtered:
                operation["parameters"] = filtered
            else:
                operation.pop("parameters", None)

    consequential_operations = {
        ("/v2/orders", "post"),
        ("/v2/orders", "delete"),
        ("/v2/orders/{order_id}", "delete"),
        ("/v2/positions", "delete"),
        ("/v2/positions/{symbol}", "delete"),
        ("/v2/watchlists", "post"),
        ("/v2/watchlists/{watchlist_id}", "put"),
    }
    for path, method in consequential_operations:
        operation = paths.get(path, {}).get(method)
        if isinstance(operation, dict):
            operation["x-openai-isConsequential"] = True

    order_response_schema = {
        "type": "object",
        "additionalProperties": True,
        "required": ["id", "symbol", "status"],
        "properties": {
            "id": {
                "type": "string",
                "title": "Order ID",
                "description": "Unique Alpaca order identifier.",
            },
            "client_order_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Client Order ID",
            },
            "symbol": {"type": "string", "title": "Symbol"},
            "status": {
                "type": "string",
                "title": "Status",
                "description": "Current order status reported by Alpaca.",
            },
            "side": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Side",
            },
            "type": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Type",
            },
            "order_class": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Order Class",
            },
            "time_in_force": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Time In Force",
            },
            "qty": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Qty",
            },
            "notional": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Notional",
            },
            "limit_price": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Limit Price",
            },
            "stop_price": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Stop Price",
            },
            "filled_qty": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Filled Qty",
            },
            "submitted_at": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Submitted At",
            },
            "updated_at": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "title": "Updated At",
            },
            "extended_hours": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "title": "Extended Hours",
            },
            "legs": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                    {"type": "null"},
                ],
                "title": "Legs",
            },
        },
    }

    schemas = components.setdefault("schemas", {})
    schemas["OrderResponse"] = order_response_schema

    for path, method in (("/v2/orders", "post"), ("/v2/orders/sync", "post")):
        response = paths.get(path, {}).get(method, {}).get("responses", {}).get("200")
        if not isinstance(response, dict):
            continue
        content = response.setdefault("content", {}).setdefault("application/json", {})
        content["schema"] = {"$ref": "#/components/schemas/OrderResponse"}

    array_response_paths = (
        ("/v2/bars", "get"),
        ("/v2/trades", "get"),
        ("/v2/quotes", "get"),
    )
    for path_key, http_method in array_response_paths:
        array_schema = (
            paths
            .get(path_key, {})
            .get(http_method, {})
            .get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema")
        )
        if isinstance(array_schema, dict):
            items = array_schema.get("items")
            if isinstance(items, dict) and items.get("type") == "object":
                items.setdefault("additionalProperties", True)

    create_order_schema = schemas.get("CreateOrder")

    unauthorized_response = {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "description": "Error message explaining why the request is unauthorized."
                        }
                    },
                    "additionalProperties": True,
                }
            }
        },
    }

    rate_limited_response = {
        "description": "Rate limited",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "description": "Explanation of the rate limit condition."
                        }
                    },
                    "additionalProperties": True,
                }
            }
        },
    }

    four_xx_responses = {
        ("/healthz", "get"): {"429": rate_limited_response},
        ("/v2/orders", "get"): {"401": unauthorized_response},
        ("/v2/orders", "delete"): {"401": unauthorized_response},
        ("/v2/account", "get"): {"401": unauthorized_response},
        ("/v2/positions", "get"): {"401": unauthorized_response},
        ("/v2/watchlists", "get"): {"401": unauthorized_response},
    }

    for (path_key, http_method), extra_responses in four_xx_responses.items():
        operation = paths.get(path_key, {}).get(http_method)
        if not isinstance(operation, dict):
            continue
        responses = operation.setdefault("responses", {})
        for status_code, response_payload in extra_responses.items():
            responses.setdefault(status_code, response_payload)

    create_order_schema = schemas.get("CreateOrder")
    if isinstance(create_order_schema, dict):
        create_props = create_order_schema.setdefault("properties", {})

        side = create_props.get("side")
        if isinstance(side, dict):
            side["enum"] = ["buy", "sell"]

        order_type = create_props.get("type")
        if isinstance(order_type, dict):
            order_type["enum"] = [
                "market",
                "limit",
                "stop",
                "stop_limit",
                "trailing_stop",
            ]

        time_in_force = create_props.get("time_in_force")
        if isinstance(time_in_force, dict):
            time_in_force["enum"] = ["day", "gtc", "opg", "cls", "ioc", "fok"]

        order_class = create_props.get("order_class")
        order_class_enum = ["simple", "bracket", "oco", "oto"]
        if isinstance(order_class, dict):
            any_of = order_class.get("anyOf")
            if isinstance(any_of, list):
                for branch in any_of:
                    if isinstance(branch, dict) and branch.get("type") == "string":
                        branch["enum"] = order_class_enum
            else:
                order_class["enum"] = order_class_enum

        take_stop_keys = ("take_profit", "stop_loss")
        for key in take_stop_keys:
            prop = create_props.get(key)
            if not isinstance(prop, dict):
                continue
            any_of = prop.get("anyOf")
            if isinstance(any_of, list):
                for branch in any_of:
                    if isinstance(branch, dict) and branch.get("type") == "object":
                        branch.setdefault("additionalProperties", True)

        create_order_schema["oneOf"] = [
            {
                "required": ["qty"],
                "properties": {"qty": {"type": "number"}},
                "not": {"required": ["notional"]},
            },
            {
                "required": ["notional"],
                "properties": {"notional": {"type": "number"}},
                "not": {"required": ["qty"]},
            },
        ]

    _prune_openapi_operations(schema)
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



