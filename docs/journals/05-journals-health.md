# 05-journals-health.md
version: 2025-09-08
status: canonical
scope: journals/health

## Health endpoints

- **GET /health** – Returns `{ "ok": true, "db": "<DATABASE_URL>" }` indicating the service is running and the database connection string. No API key required.
- **GET /readyz** – Checks database connectivity. Without parameters returns `{ "ready": true }` on success. With `detail=true` it also returns the `db` field. Returns HTTP 503 with an error message if the DB is unreachable.
- **GET /openapi.yaml** – Provides the current OpenAPI specification as YAML. Useful for client generation and documentation.

