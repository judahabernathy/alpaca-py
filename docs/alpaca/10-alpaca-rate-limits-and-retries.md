# 10-alpaca-rate-limits-and-retries.md
version: 2025-09-08
status: canonical
scope: alpaca/rate-limits

## HTTP rate-limit signals
Read these response headers and pace requests:
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset  (Unix epoch when quota resets)

### 429 handling
- Treat as “Too Many Requests”.
- Sleep until X-RateLimit-Reset, then retry with jitter.
- Keep per-endpoint budgets.

## Market Data plan RPM (historical API)
- Basic: 200 requests/min
- Algo Trader Plus: 10,000 requests/min

## WebSocket limits
- Most subscriptions allow 1 active connection per endpoint. Second connection returns:

* * code: 406
* * msg: "connection limit exceeded"
*

## Backoff recipe
1) Track Remaining/Reset per token + host.
2) If Remaining <= 1, pause until Reset + random(50–200ms).
3) On 429, exponential backoff starting at Reset or 1s, max 60s.

## Monitor
- Log 429s, Reset deltas, and reconnect causes.

Sources:
- https://docs.alpaca.markets
- https://docs.alpaca.markets/docs/market-data-api
- https://docs.alpaca.markets/docs/streaming-market-data
