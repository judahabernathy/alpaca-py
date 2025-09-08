# 03-alpha-classify.md
version: 2025-09-08
status: canonical
scope: alpha/classify

## Endpoint: POST /classify

Classifies a list of symbols and returns technical metrics and candidate tags.

### Request
- **Header:** `X-API-Key` must match `ALPHA_CLASSIFIER_KEY`.
- **Body:** JSON object with properties:
  - `symbols` (list[string], required) — List of tickers to classify.
  - `rvolMode` (string, optional) — Reserved for future volume modes; defaults to `"20bar"`.
    - If no symbols are provided, the service returns HTTP 422.
    - If more than 50 symbols are provided, the service returns HTTP 413.

### Processing
- Symbols are deduplicated and upper-cased.
- The service fetches up to 200 daily candles and 10 days of 15-minute candles per symbol from Finnhub.
- It computes technical indicators and applies classification rules (see metrics doc).

### Response
Returns a JSON array of objects, each with fields:
- `symbol` – Ticker symbol.
- `close` – Last closing price.
- `sma50` – 50-day simple moving average.
- `atr` – 14-day average true range.
- `pir_pct` – Percent position in last 5-day range.
- `rvol15` – 15-minute relative volume (latest bar vs prior 20 bars).
- `candidates` – Comma-separated tags (see metrics doc).
- `confirm` – `"confirm_15m"` if `rvol15` ≥ 1.2, otherwise `"pending"`.

If an error occurs for a symbol (e.g., no data), the output includes default values and an `error:<message>` tag.

### Limits & Concurrency
- Maximum 50 symbols per request; excess returns HTTP 413.
- Classification uses a thread pool with up to 8 workers (min of 8 and number of symbols).
- The service retries Finnhub calls up to 4 times with exponential backoff on 429 responses.
