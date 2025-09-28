# 07-finnhub-insiders-ownership.md
version: 2025-09-08
status: canonical
scope: finnhub/insiders-ownership

## Insider transactions
- **Endpoint**: `GET /stock/insider-transactions`
- **Params**:

  * `symbol` (optional): Company ticker. If omitted, returns the latest filings across all companies.
  * `from` (optional): Start date `YYYY‑MM‑DD`.
  * `to` (optional): End date `YYYY‑MM‑DD`.
- **Description**: Returns Form 3/4/5 and related filings summarizing insider trades. Each record in the `data` array contains the insider’s `name`, number of `share`s held after the transaction, `change` from the previous period, `filingDate`, `transactionDate`, `transactionCode` and `transactionPrice`.

## Insider sentiment
- **Endpoint**: `GET /stock/insider-sentiment`
- **Params**:

  * `symbol` (required): Ticker.
  * `from` (required): Start date `YYYY‑MM‑DD`.
  * `to` (required): End date `YYYY‑MM‑DD`.
- **Description**: Provides monthly insider sentiment for a company. Each item in the `data` array includes `month`, `year`, `change` (net buying/selling), `mspr` (monthly share purchase ratio) and `symbol`.

## Ownership (premium)
- **Endpoint**: `GET /stock/ownership`
- **Params**:

  * `symbol` (required): Company ticker.
  * `limit` (optional): Maximum number of holders to return.
- **Description**: Lists shareholders and their holdings. Each record includes the investor’s `name`, number of `share`s, `change` (net buy/sell) and `filingDate`. Finnhub notes that full ownership data requires a premium plan; users on fundamentals‑1 may receive `403 Access Denied`.

