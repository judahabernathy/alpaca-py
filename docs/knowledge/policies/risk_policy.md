# Risk policy

## Sizing
- qty = floor((equity * risk_pct) / abs(entry − stop)); risk_pct=0.02 default.

## Exposure + Halts
- Per‑name exposure cap: (position_value + order_value) ≤ 0.20 * equity.
- Daily loss halt: realized_pnl_day > −0.05 * equity_day_start required to place new entries.

## Sessions
- RTH: allow market/limit/stop/stop_limit; brackets permitted.
- EXT: enforce limit + day + extended=true; no bracket; no trailing entry.

## Freshness + Slippage
- TTL: quote/balances/positions must be fresh (≤10s). If stale, refresh Preflight.
- Slippage guard: |limit − last| / last ≤ 0.005.

## Earnings gate
- No new entries in the no‑trade window unless user explicitly approves where policy allows. Prefer watchlist.
