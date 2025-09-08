# 04-finrl-predict.md
version: 2025-09-08
status: canonical
scope: finrl/predict

## Endpoint
- **GET /predict**

## Description
Fetch the most recently saved parameter blob from the journaling service.  Optionally specify a version identifier to retrieve a particular snapshot.  This endpoint allows clients to query tuned strategy parameters without re‑running training.

## Query parameters
- **version** (string, optional) – Specific version label to retrieve.  If omitted or set to `latest`, the most recent parameters are returned.

## Processing
The service makes a GET request to the journaling service:
- If a version is provided and not equal to `"latest"`, it queries `{JOURNAL_STORAGE_BASE_URL}/params/{version}` with `source=finrl` in the query string.
- Otherwise, it queries `{JOURNAL_STORAGE_BASE_URL}/params/latest` with `source=finrl`.
- The `X‑API‑Key` header is set to `JOURNAL_API_KEY`.
If the journal service returns a 404 response, the FinRL‑actions service responds with `{ "detail": "No signals available (train first or save parameters)" }`.  Other error conditions are surfaced as a `502` response.

## Response
A JSON object containing:
- **version** – The version identifier of the returned parameters.
- **params** – The saved parameter blob.  Keys such as `win_rate`, `avg_R`, `default_stop_R`, `target_R`, etc., depend on the training logic.

## Example
```bash
# Retrieve the latest parameters
curl -X GET -H "X-API-Key: <your_key>" \
  https://finrl-actions-production.up.railway.app/predict

# Retrieve a specific version
curl -X GET -H "X-API-Key: <your_key>" \
  https://finrl-actions-production.up.railway.app/predict?version=journal:2025-08-28T10:00:00Z
```

Sources:
- Predict endpoint logic and error handling
- Journal storage retrieval logic
