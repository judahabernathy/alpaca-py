# Error policy

## Kinds
- AUTH, VALIDATION, SESSION, RATE_LIMIT, RETRYABLE, UNSUPPORTED

## Retry rules
- 5xx: retry once with 200–500 ms jitter.
- 429: honor Retry‑After up to 5 s, then single retry.

## Circuit breaker
- After two consecutive failures on the same symbol: 30 s open; degrade to watchlist entry with appropriate reason.

## Failure payloads
- Always include `error.kind`, `remedy`, `next_step`.
