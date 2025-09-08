# 05-alpha-health.md
version: 2025-09-08
status: canonical
scope: alpha/health

## Endpoint: GET /healthz

Returns a simple liveness indicator:

```json
{ "ok": true }
```

This endpoint does not require authentication and can be used for monitoring.
