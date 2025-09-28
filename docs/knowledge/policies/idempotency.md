# Idempotency

- Require client_order_id on create. If missing: alpha-YYYYMMDDHHmmss-rand6.
- Pre‑create check: list orders and return existing by client_order_id if present (no duplicate submit).
- Journal dedupe: exactly one Journal per client_order_id.
