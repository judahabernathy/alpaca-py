# 03-finrl-train.md
version: 2025-09-08
status: canonical
scope: finrl/train

## Endpoint
- **POST /train**

## Description
Accepts a list of trade journal entries and derives basic strategy parameters.  It computes the win rate and average R‑multiple across the journal, assigns conservative default multipliers, and optionally persists the derived parameter blob to an external journaling service.

## Request body
A JSON object containing:

- **journal_data** (array, required) – A list of trade entries.  Each entry may include:
  * `symbol` (string, required)
  * `side` (string, optional; e.g. "buy" or "sell")
  * `strategy` (string, optional) – strategy tag such as “Pullback” or “ORB”
  * `entry_time`, `exit_time` (ISO 8601 timestamps, optional)
  * `entry_price`, `exit_price` (floats, optional)
  * `stop_price`, `tp_price` (floats, optional)
  * `outcome` (string, optional; e.g. "win", "loss", "tp", "sl", "manual")
  * `R_multiple` (float, optional) – realized return in R units
  * `notes` (string, optional)
- **save_params** (boolean, default `true`) – If `true`, the derived parameters are saved to the external journal service.  Otherwise they are returned without persistence.
- **version** (string, optional) – Custom version identifier.  If omitted, a version string beginning with `journal:` and containing the current UTC timestamp is generated.

Unknown fields are rejected (`extra="forbid"`).  An empty `journal_data` list yields a `400` error.

## Derivation logic
For a journal of *n* entries, the service computes:

- **win_rate** – fraction of entries considered wins.  A win is determined by the `outcome` starting with "win" or a positive `R_multiple`.
- **avg_R** – arithmetic mean of all provided `R_multiple` values or a proxy based on the outcome field.
- **default_stop_R** – fixed at `1.0` as a conservative stop multiplier.
- **target_R** – fixed at `2.0` as a baseline take‑profit multiplier.
- **notes** – a constant string "Derived from journal_data" identifying how the parameters were obtained.

The version label defaults to `journal:<timestamp>` if none is provided.

## Persistence
If `save_params` is `true`, the service posts the derived parameters to `{JOURNAL_STORAGE_BASE_URL}/params` with JSON `{ "version", "source":"finrl", "blob": params }` and the `X‑API‑Key` header set to `JOURNAL_API_KEY`.  Any error in saving results in a `502` response.

## Response
On success, returns a JSON object:

- **version** – Version identifier used when saving the parameter blob.
- **params** – Object containing keys `win_rate`, `avg_R`, `default_stop_R`, `target_R`, `notes`.
- **saved** (boolean) – `true` if the blob was persisted to the journal service, `false` otherwise.

## Example
```bash
curl -X POST https://finrl-actions-production.up.railway.app/train \
  -H "X-API-Key: <your_key>" \
  -H "Content-Type: application/json" \
  -d '{"journal_data":[{"symbol":"AAPL","side":"buy","entry_price":163.04,"outcome":"win","R_multiple":2.0}], "save_params":true }'
```

Sources:
- Train endpoint definition and body schema
- Parameter derivation logic
- Persistence mechanism and response structure
