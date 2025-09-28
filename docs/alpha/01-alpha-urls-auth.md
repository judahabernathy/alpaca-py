# 01-alpha-urls-auth.md
version: 2025-09-08
status: canonical
scope: alpha/urls-auth

## Base URLs
- Production: https://alpha-classifier-production.up.railway.app
- Local development: http://localhost:8000

## Endpoints summary
- **POST /classify** — Classify a batch of tickers and return metrics and candidate tags.
- **GET /healthz** — Liveness check.

## Authentication
- All POST `/classify` requests require `X-API-Key` header matching the environment variable `ALPHA_CLASSIFIER_KEY`.
- If the key is missing or incorrect, the service returns HTTP 401.

### Example
```bash
curl -X POST https://alpha-classifier-production.up.railway.app/classify \
  -H "X-API-Key: <your_key>" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL","NVDA","TSLA"]}'
```
