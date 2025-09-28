# Memory workflow (Journals)

## Store
- POST /params {version:'latest', source:'alpha-memory', blob{account, positions, watchlist, last_orders, risk_config, strategy_state}}.

## Load
- GET /params/latest?source=alpha-memory&version=latest.

## Policy
- Sync at session start; precedence = live Preflight > memory; update memory when diverging.
- Memory older than 30 days ⇒ stale; prompt refresh.
- No secrets in blobs.

## UI
- Annotate responses: “Memory: loaded|missing|stale (as of <ts>)”.
