import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

TTL_SEC_DEFAULT = int(os.getenv("ALPHA_QUOTE_TTL_SEC") or 10)
DRIFT_PCT_DEFAULT = float(os.getenv("ALPHA_DRIFT_PCT") or 0.005)  # 0.5%


def _mk_client() -> Optional[StockHistoricalDataClient]:
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not secret:
        return None
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret)


def latest_mid_and_age(symbol: str) -> Optional[Tuple[float, int, Dict[str, Any]]]:
    """
    Returns (mid_price, age_seconds, debug) or None if unavailable.
    Uses Alpaca Market Data v2 latest-quote API via alpaca-py.
    """
    client = _mk_client()
    if not client:
        return None
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    try:
        out = client.get_stock_latest_quote(req)
        q = out[symbol]
        bid = float(getattr(q, "bid_price"))
        ask = float(getattr(q, "ask_price"))
        mid = (bid + ask) / 2.0
        ts = getattr(q, "timestamp", None)
        if ts is None:
            return None
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        now = datetime.now(timezone.utc)
        age_s = int((now - ts).total_seconds())
        return mid, age_s, {"bid": bid, "ask": ask, "timestamp": ts.isoformat()}
    except Exception:
        return None


def evaluate_limit_guard(
    symbol: str,
    limit_price: float,
    ttl_sec: Optional[int] = None,
    drift_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Soft-fail evaluation for limit orders.
    Returns dict: { ok: bool, reason: 'stale'|'drift'|None, age: int|None, drift: float|None, debug: {...} }
    If quote unavailable, returns ok=True to avoid breaking flows.
    """
    ttl = int(ttl_sec or TTL_SEC_DEFAULT)
    dp = float(drift_pct or DRIFT_PCT_DEFAULT)

    res = latest_mid_and_age(symbol)
    if res is None:
        return {
            "ok": True,
            "reason": None,
            "age": None,
            "drift": None,
            "debug": {"unavailable": True},
        }

    mid, age_s, dbg = res
    if age_s > ttl:
        return {"ok": False, "reason": "stale", "age": age_s, "drift": None, "debug": dbg}

    drift = abs(mid - float(limit_price)) / float(limit_price)
    if drift > dp:
        return {"ok": False, "reason": "drift", "age": age_s, "drift": drift, "debug": dbg}

    return {"ok": True, "reason": None, "age": age_s, "drift": drift, "debug": dbg}
