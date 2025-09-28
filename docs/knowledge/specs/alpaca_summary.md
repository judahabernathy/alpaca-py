# Alpaca‑py summary

## Auth
- Header `x-api-key`

## Core
- POST /v2/orders (create; require client_order_id in caller)
- GET /v2/orders (list), GET /v2/orders/{order_id}, DELETE /v2/orders/{order_id}
- GET /v2/account
- GET /v2/positions, DELETE /v2/positions (close all), GET/DELETE /v2/positions/{symbol}
- Watchlists: GET/POST /v2/watchlists, GET/PUT/DELETE /v2/watchlists/{id}

## Policy overlay
- Allowed types: market, limit, stop, stop_limit
- TIF: day, gtc
- EXT: limit + day + extended=true; no bracket/trailing entries

## Idempotency
- Deduplicate by client_order_id client‑side.
