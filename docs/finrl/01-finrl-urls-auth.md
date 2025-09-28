# 01-finrl-urls-auth.md
version: 2025-09-08
status: canonical
scope: finrl/urls-auth

## Base URLs
- Production: [https://finrl-actions-production.up.railway.app](https://finrl-actions-production.up.railway.app)
- Local development: [http://localhost:8000](http://localhost:8000)

## Endpoints
- **POST /train** — Train using journal entries and optionally save parameters.
- **GET /predict** — Retrieve the latest trained parameters.
- **GET /healthz** — Basic liveness probe. Returns `{ "status": "ok" }`.
- **GET /readyz** — Readiness probe that checks upstream dependencies and optionally returns details.

## Authentication
- All training and prediction endpoints require an API key via the `X-API-Key` header. Health probes do not require authentication.
- The upstream FinRL service base URL is configured via the environment variable `FINRL_BASE_URL` and defaults to `http://finrl:8000`.
- The FinRL signal provider uses `FINRL_PREDICT_URL` (default `http://finrl:8000/predict`) to fetch parameters from the upstream service.

Sources:
- OpenAPI server definitions for production and local environments
- Endpoint descriptions and security requirements
- Environment variable notes from settings file
