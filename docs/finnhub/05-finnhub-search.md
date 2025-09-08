# 05-finnhub-search.md
version: 2025-09-08
status: canonical
scope: finnhub/search

## Symbol search
- **Endpoint**: `GET /search`
- **Params**:

  * `q` (required): Query string (company name, ticker, ISIN or CUSIP).
  * `exchange` (optional): Exchange code to restrict the results.
- **Description**: Performs a fuzzy search for securities. The response includes a `count` and a `result` array; each entry contains `symbol`, `displaySymbol`, `description` and `type`.

## Stock symbols
- **Endpoint**: `GET /stock/symbol`
- **Params**:

  * `exchange` (required): Exchange code (e.g. `US`).
  * `mic` (optional): Market identification code.
  * `securityType` (optional): Filter by security type.
  * `currency` (optional): Filter by currency.
- **Description**: Returns a catalogue of all instruments available on the specified exchange. Each item provides `symbol`, `displaySymbol`, `description`, `currency`, `figi`, `mic` and other metadata.

