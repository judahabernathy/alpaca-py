# 04-finnhub-news.md
version: 2025-09-08
status: canonical
scope: finnhub/news

## Company news
- **Endpoint**: `GET /company-news`
- **Params**:

  * `symbol` (required): Company ticker.
  * `from` (required): Start date `YYYY‑MM‑DD`.
  * `to` (required): End date `YYYY‑MM‑DD`.
- **Description**: Fetches the latest articles about a company. Each article includes `category`, `datetime` (Unix seconds), `headline`, `id`, `image`, `related` tickers, `source`, `summary` and `url`.

## News sentiment (premium)
- **Endpoint**: `GET /news-sentiment`
- **Params**: `symbol` (required).
- **Description**: Returns AI‑scored sentiment for recent news. The response includes:

  * `buzz` – Article counts and news ratio.
  * `companyNewsScore` – Composite sentiment score for the company.
  * `sectorAverageBullishPercent` and `sectorAverageNewsScore` – Sector benchmarks.
  * `sentiment` – Bearish and bullish percentages.
- **Plan**: This endpoint is premium. Users on marketdata‑basic or fundamentals‑1 plans will receive `403 Access Denied`.

## Social sentiment (premium)
- **Endpoint**: `GET /stock/social-sentiment`
- **Params**: `symbol` (required), `from` and `to` (optional).
- **Description**: Provides daily counts of Reddit and Twitter mentions, positive and negative scores and an overall sentiment score. Response fields include `mention`, `positiveMention`/`positiveScore`, `negativeMention`/`negativeScore`, `score`, `date`. This endpoint is premium and returns `403` on basic plans.

