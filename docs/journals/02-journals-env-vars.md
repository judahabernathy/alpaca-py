# 02-journals-env-vars.md
version: 2025-09-08
status: canonical
scope: journals/env-vars

## Required variables

| Variable            | Default                              | Description |
|--------------------|--------------------------------------|-------------|
| **JOURNAL_API_KEY** | —                                    | API key required in the `X‑API‑Key` header for all protected endpoints. |
| **DB_ENGINE**       | `sqlite`                             | Database engine (`sqlite` or `duckdb`). |
| **DB_DIR**          | `$DB_DIR` or `$RAILWAY_VOLUME_MOUNT_PATH` or `/app/data` | Directory where the database file lives. |
| **DB_FILE**         | `journals.db` (sqlite) or `journals.duckdb` (duckdb) | Name of the DB file. |
| **PUBLIC_URL**      | `https://turbo-broccoli-production.up.railway.app` | Absolute server URL used in OpenAPI spec. |

## Notes
- The API key is validated against `JOURNAL_API_KEY` for every protected route.
- `DB_ENGINE`, `DB_DIR` and `DB_FILE` construct the `DATABASE_URL` at runtime.
- Setting `PUBLIC_URL` ensures that the OpenAPI spec shows the correct base path.

