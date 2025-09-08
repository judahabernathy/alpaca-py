# 01-finnhub-urls-auth.md
version: 2025-09-08
status: canonical
scope: finnhub/urls-auth

## Base URL
- [https://finnhub.io/api/v1](https://finnhub.io/api/v1)

## Authentication
- Supply your API key via the `X-Finnhub-Token` header or the `token` query parameter on every request.
- The service enforces a per‑account call cap of ~30 requests per second; requests above this threshold return HTTP 429 status.

## Plan categories
- **marketdata‑basic** – Provides real‑time quotes, historical candles, company news, symbol search, option chains and technical indicators (some indicators require premium).
- **fundamentals‑1** – Enables company profile, basic financial metrics, earnings history, earnings calendar, analyst recommendations, peers, insider transactions and some ownership data.
- **premium** – Unlocks news sentiment, social sentiment, advanced ownership reports and most technical indicators. Endpoints marked as premium will return `403 Access Denied` on lower plans.

## Notes
- All endpoints are read‑only (`GET`) and return JSON.
- Date ranges must be supplied in `YYYY‑MM‑DD` format; timestamps are Unix seconds.
- Always respect plan limits and handle `429` responses gracefully.

