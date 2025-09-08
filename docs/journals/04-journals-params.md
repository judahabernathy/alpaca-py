# 04-journals-params.md
version: 2025-09-08
status: canonical
scope: journals/params

## Models

### ParamIn (request)
- **version** (string, required) – Human‑readable identifier.
- **source** (string, optional, default "finrl") – Origin of the parameters.
- **blob** (object, required) – Arbitrary JSON payload.

### ParamOut (response)
Extends `ParamIn` with `id` (int) and `created_at` (datetime).

## Endpoints

- **POST /params** – Saves a new parameter record and returns it with ID and creation time.
- **GET /params/latest** – Returns the most recent `ParamOut`. Optional query parameters: `source` and `version`. Returns 404 if no match or 500 if a DB error occurs.

### GET /openapi.yaml
Serves the OpenAPI 3.1 specification in YAML format. This endpoint is excluded from the API schema but can be fetched directly.

