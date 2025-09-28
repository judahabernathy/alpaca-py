# 01-journals-urls-auth.md
version: 2025-09-08
status: canonical
scope: journals/urls-auth

## Base URLs
- Production: [https://turbo-broccoli-production.up.railway.app](https://turbo-broccoli-production.up.railway.app)
- Local development: [http://localhost:8000](http://localhost:8000)

## Endpoints summary
- **POST /journal** — Create a journal entry.
- **POST /journal/bulk** — Bulk insert multiple entries (payload has an `items` array).
- **GET /journal** — List entries; filter by symbol, strategy, start/end timestamps; limit results.
- **GET /journal/{id}** — Retrieve a single entry.
- **PATCH /journal/{id}** — Update fields on an entry.
- **DELETE /journal/{id}** — Remove an entry.
- **GET /stats** — Compute trades, win rate and average R per symbol/strategy.
- **POST /params** — Save a versioned parameter blob.
- **GET /params/latest** — Get the most recent parameter blob, optional source/version filters.
- **GET /health** — Liveness check with DB URL.
- **GET /readyz** — Readiness check; optional `detail` flag for DB info.
- **GET /openapi.yaml** — Download the OpenAPI spec.

## Authentication
- All endpoints except `/health`, `/readyz` and `/openapi.yaml` require an API key in the `X‑API‑Key` header.
- The expected key comes from the `JOURNAL_API_KEY` environment variable; invalid or missing keys return HTTP 401.

