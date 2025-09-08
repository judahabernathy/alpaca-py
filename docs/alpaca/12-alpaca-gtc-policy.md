# 12-alpaca-gtc-policy.md
version: 2025-09-08
status: canonical
scope: alpaca/gtc

## GTC aging
- Alpaca auto-cancels GTC orders 90 days after creation.
- See `expires_at` on order objects.
- Cancel job runs on the `expires_at` date at 4:15 pm ET; order may show `pending_cancel` until venue confirms.

## What to do
- Read `expires_at` when placing orders.
- Re-issue long-lived GTC instructions before expiration if still desired.
- Journal the new client_order_id when re-placing.

## Affects
- Applies to single, bracket, OCO, and stop orders placed with GTC.

Sources:
- https://docs.alpaca.markets/docs/orders-at-alpaca
