# 13-alpaca-idempotency.md
version: 2025-09-08
status: canonical
scope: alpaca/idempotency

## Use client_order_id
- Provide `client_order_id` (≤128 chars) on POST /v2/orders.
- Retrieve by client id:

* * GET /v2/orders:by_client_order_id?client_order_id=...
*

## Why
- Prevent duplicate entries on retries.
- Correlate fills and updates across services.

## Pattern
1) Generate UUIDv4 per intent.
2) POST order with `client_order_id`.
3) If timeout or 5xx, GET by client id before retrying.

### PowerShell 7
```powershell
$cid=[guid]::NewGuid().ToString()
$body=@{symbol="AAPL";side="buy";qty="1";type="market";time_in_force="day";client_order_id=$cid} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "https://paper-api.alpaca.markets/v2/orders" -Headers @{ "APCA-API-KEY-ID"="$env:ALPACA_KEY"; "APCA-API-SECRET-KEY"="$env:ALPACA_SECRET" } -Body $body -ContentType "application/json"
# verify
Invoke-RestMethod -Method Get -Uri "https://paper-api.alpaca.markets/v2/orders:by_client_order_id?client_order_id=$cid" -Headers @{ "APCA-API-KEY-ID"="$env:ALPACA_KEY"; "APCA-API-SECRET-KEY"="$env:ALPACA_SECRET" }
```

## Replace caveat
- A successful replace response does not guarantee the original wasn’t filled first; confirm via `trade_updates` stream.

Sources:
- https://docs.alpaca.markets/reference/getorderbyclientorderid
- https://docs.alpaca.markets/docs/working-with-orders
- https://docs.alpaca.markets/reference/replaceorderforaccount-1
