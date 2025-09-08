# 06-finrl-journal.md
version: 2025-09-08
status: canonical
scope: finrl/journal

## Purpose
Describe how the FinRL‑actions service persists and retrieves learned parameters via an external journaling service.  The journaling layer replaces the local Postgres storage used in earlier versions of the service.

## Saving parameters
When training completes and `save_params` is `true`, the service posts the derived parameter blob to the journaling service.  The target URL is constructed as:
``
{JOURNAL_STORAGE_BASE_URL}/params
``
The POST body includes:
- `version` – version identifier of the parameter set
- `source` – fixed string "finrl" to distinguish FinRL‑actions blobs
- `blob` – the parameter dictionary
The call uses an `X-API-Key` header containing `JOURNAL_API_KEY`.  Any failure to save raises a `502` error.

## Retrieving parameters
To fetch parameters, the service sends a GET request to either:
- `{JOURNAL_STORAGE_BASE_URL}/params/latest?source=finrl` to obtain the most recent parameters; or
- `{JOURNAL_STORAGE_BASE_URL}/params/{version}?source=finrl` to fetch a specific version.
The same `X-API-Key` header is included.  If the journaling service returns a 404, the FinRL‑actions service forwards a message indicating that no signals are available.

## Normalization
The journaling API may return keys such as `blob`, `params` or `data`.  The FinRL‑actions service normalizes the response to return a consistent object with `version` and `params` fields.

## Example
``
# Save parameters (performed internally during /train)
POST {JOURNAL_STORAGE_BASE_URL}/params
Headers: X-API-Key: <JOURNAL_API_KEY>
Body: {"version":"journal:2025-09-08T12:00:00Z","source":"finrl","blob":{"win_rate":0.5,"avg_R":0.8,...}}

# Retrieve latest parameters
GET {JOURNAL_STORAGE_BASE_URL}/params/latest?source=finrl
Headers: X-API-Key: <JOURNAL_API_KEY>

# Retrieve specific version
GET {JOURNAL_STORAGE_BASE_URL}/params/journal:2025-09-08T12:00:00Z?source=finrl
Headers: X-API-Key: <JOURNAL_API_KEY>
```

Sources:
- Journal save logic in the training endpoint
- Journal retrieval logic in the prediction endpoint
