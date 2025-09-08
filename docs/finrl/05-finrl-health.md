# 05-finrl-health.md
version: 2025-09-08
status: canonical
scope: finrl/health

## Health endpoints
The FinRL‑actions service exposes health probes for monitoring and orchestration.

### GET /healthz
Returns a simple liveness indicator.  When the service is running, it responds with:
```json
{ "status": "ok" }
```
Authentication is not required.  Use this endpoint to verify that the service process is responding.

### GET /readyz
Readiness probe that checks upstream dependencies, such as the journaling service or the upstream FinRL engine.  An optional query parameter `detail` (boolean) can be supplied to include additional environment details in the response.  Possible responses:
- **200 OK** – The service and its dependencies are ready.
- **503 Service Unavailable** – One or more dependencies are unreachable.
Authentication is not required by default.  This endpoint is defined in the OpenAPI specification but may not yet be implemented in the current code base.

Sources:
- Liveness probe implementation
- OpenAPI readiness probe definition
