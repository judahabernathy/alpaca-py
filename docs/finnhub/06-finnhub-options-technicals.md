# 06-finnhub-options-technicals.md
version: 2025-09-08
status: canonical
scope: finnhub/options-technicals

## Option chain
- **Endpoint**: `GET /stock/option-chain`
- **Params**:

  * `symbol` (required): Stock ticker.
  * `date` (optional): Expiration date `YYYY‑MM‑DD` to limit the chain.
- **Description**: Returns a snapshot of available option contracts for a symbol. The response contains a list of expiration dates with an `options` object for each; `options` contains `CALL` and `PUT` arrays. Each contract summary includes `symbol`, `type` (`CALL` or `PUT`), `strike`, `lastPrice` and may include additional fields. According to the action schema, this endpoint is part of the marketdata‑basic plan, although official Finnhub docs classify detailed option data as premium.

## Technical indicator series
- **Endpoint**: `GET /indicator`
- **Params**:

  * `symbol` (required): Ticker.
  * `resolution` (required): Interval (`1`, `5`, `15`, `30`, `60`, `D`, `W`, `M`).
  * `from` (required): Start time (Unix seconds).
  * `to` (required): End time (Unix seconds).
  * `indicator` (required): Indicator name (e.g. `sma`, `ema`, `rsi`, `macd`).
  * `timeperiod` (optional): Period parameter for certain indicators.
- **Description**: Computes a technical indicator over the specified period and returns a `series` of numbers alongside a matching array of `timestamps`. For example, a 3‑period simple moving average returns the moving average for each closing price.
- **Plan**: Finnhub labels this endpoint as premium; basic plans may receive `403` responses.

