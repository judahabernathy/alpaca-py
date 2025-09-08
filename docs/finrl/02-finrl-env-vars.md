# 02-finrl-env-vars.md
version: 2025-09-08
status: canonical
scope: finrl/env-vars

## Purpose
Document the configuration knobs that must be set via environment variables when deploying the FinRL‑actions service.  Correct configuration ensures that the service can authenticate inbound requests, forward training and prediction calls to the upstream FinRL engine and journaling service, and segregate environments.

## Required variables
| Variable | Default | Description |
|---------|---------|-------------|
| **API_KEY** | — | API key required in the `X‑API‑Key` header for all endpoints except health probes.  Requests lacking this key are rejected. |
| **FINRL_BASE_URL** | `http://finrl:8000` | Base URL of the upstream FinRL service used for proxy routes.  Training and prediction requests are forwarded here. |
| **FINRL_API_KEY** | (unset) | Optional API key added to the `X‑API‑Key` header on outbound calls to the upstream FinRL service. |
| **ENV** | `default` | Label used to segregate watchlists or context.  Currently unused but reserved for future expansion. |
| **JOURNAL_STORAGE_BASE_URL** | `http://journal:8000` | Base URL of the external journaling service used to persist parameter blobs. |
| **JOURNAL_API_KEY** | — | API key for the journaling service.  Sent in the `X‑API‑Key` header when saving or retrieving parameters. |
| **FINRL_PREDICT_URL** | `http://finrl:8000/predict` | URL used by the FinRL signal provider to fetch the latest parameters from the upstream service.  Overrides `FINRL_BASE_URL` for synchronous calls. |

## Notes
- `DATABASE_URL` was removed — the service no longer uses a local Postgres database for parameter storage.  All persistence is delegated to the external journal service.
- Leave optional keys unset if the upstream FinRL service does not require authentication.

Sources:
- Environment variable definitions and defaults
- Removal of Postgres storage and external journaling service notes
