# 02-alpha-env-vars.md
version: 2025-09-08
status: canonical
scope: alpha/env-vars

## Required variables

| Variable               | Description |
|-----------------------|-------------|
| **FINNHUB_TOKEN**      | Token for the Finnhub API. The service will not start without it. |
| **ALPHA_CLASSIFIER_KEY** | API key required in the `X-API-Key` header on classify requests. |

## Notes

- The classifier makes network calls to `https://finnhub.io/api/v1/stock/candle` using `FINNHUB_TOKEN`.
- Both variables must be set; otherwise the application raises an error at startup.
