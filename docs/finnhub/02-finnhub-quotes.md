# 02-finnhub-quotes.md
version: 2025-09-08
status: canonical
scope: finnhub/quotes

## Real‑time quote
- **Endpoint**: `GET /quote`
- **Params**: `symbol` (required) – Ticker symbol.
- **Description**: Returns a snapshot of the most recent trade for a stock, including current price and daily range.
- **Response fields**:

  * `c` – Current price.
  * `d` – Change from previous close.
  * `dp` – Percent change from previous close.
  * `h` – High price of the day.
  * `l` – Low price of the day.
  * `o` – Open price of the day.
  * `pc` – Previous close price.
  * `t` – Unix timestamp (seconds).

## Historical candles
- **Endpoint**: `GET /stock/candle`
- **Params**:

  * `symbol` (required): Stock ticker.
  * `resolution` (required): Interval. Supported values: `1`, `5`, `15`, `30`, `60`, `D`, `W`, `M`.
  * `from` (required): Start time (Unix seconds).
  * `to` (required): End time (Unix seconds).
- **Description**: Fetches arrays of open‑high‑low‑close‑volume data at the specified resolution for the given period.
- **Response fields**:

  * `c` – Array of close prices.
  * `h` – Array of high prices.
  * `l` – Array of low prices.
  * `o` – Array of open prices.
  * `t` – Array of timestamps (seconds).
  * `v` – Array of volumes.
  * `s` – Status string (`"ok"` or `"no_data"`).

