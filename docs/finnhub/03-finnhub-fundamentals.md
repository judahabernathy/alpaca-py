# 03-finnhub-fundamentals.md
version: 2025-09-08
status: canonical
scope: finnhub/fundamentals

## Company profile
- **Endpoint**: `GET /stock/profile2`
- **Params**: At least one of `symbol`, `isin` or `cusip`.
- **Description**: Returns a company’s high‑level details such as name, exchange, IPO date, market capitalization, shares outstanding, country, currency, phone, website URL, logo and industry.

## Basic financial metrics
- **Endpoint**: `GET /stock/metric`
- **Params**:

  * `symbol` (required): Ticker.
  * `metric` (required): Metric category (`all`, `price`, `valuation`, `margin`). Use `all` to return the full set.
- **Description**: Returns a `metric` object containing selected ratios and statistics—such as P/E, 52‑week high/low and average trading volume—for the specified company.

## Earnings surprises
- **Endpoint**: `GET /stock/earnings`
- **Params**:

  * `symbol` (required): Company ticker.
  * `limit` (optional): Number of records to return.
- **Description**: Provides historical quarterly earnings results along with analysts’ estimates. Each record includes actual and estimated EPS, period, quarter, surprise and surprise percent.

## Earnings calendar
- **Endpoint**: `GET /calendar/earnings`
- **Params**:

  * `from` (required): Start date in `YYYY‑MM‑DD`.
  * `to` (required): End date in `YYYY‑MM‑DD`.
  * `symbol` (optional): Filter to a single ticker.
- **Description**: Returns upcoming earnings dates within the specified range. Response fields include `date`, `symbol`, `hour`, `quarter`, `year`, `epsActual`, `epsEstimate`, `revenueActual` and `revenueEstimate`.

## Recommendation trends
- **Endpoint**: `GET /stock/recommendation` (alias: `/stock/recommendation-trends`)
- **Params**: `symbol` (required).
- **Description**: Returns monthly counts of analyst ratings for the last several months. Each record contains `period` (YYYY‑MM) and counts of `strongBuy`, `buy`, `hold`, `sell` and `strongSell` recommendations.

## Peers
- **Endpoint**: `GET /stock/peers`
- **Params**:

  * `symbol` (required): Ticker.
  * `grouping` (optional): Grouping level (default `subIndustry`).
- **Description**: Returns an array of peer tickers operating in the same sector or industry.

