# 16-alpaca-errors-429.md
version: 2025-09-08
status: canonical
scope: alpaca/errors-429

## What 429 means
- HTTP 429 = rate limit exceeded. Respect per-endpoint quotas.

## Headers to read
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset  (Unix epoch when quota resets)

## Minimal handler
- If Remaining <= 1: wait until Reset + jitter.
- On 429: sleep(max(Reset-now, 1s)) with exponential backoff, cap 60s, then retry once.
- Never spam retries across multiple symbols in parallel without tokens.

### Example (pseudo)
```text
if resp.status==429:
  wait = max(resp.headers.Reset - now, 1)
  sleep(wait + rand(50..200ms))
  retry()
```

## WebSocket analog
- Connection limit errors return `{"T":"error","code":406,"msg":"connection limit exceeded"}`. Close the other session or wait before reconnecting.

## Logging
- Record endpoint, Remaining, Reset, retry count, and eventual success/fail per request id.

Sources:
- https://docs.alpaca.markets/docs/streaming-market-data
- https://docs.alpaca.markets/docs/market-data-api
