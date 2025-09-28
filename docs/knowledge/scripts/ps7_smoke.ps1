# PowerShell 7 smoke tests (Paper)

# 1) Account snapshot
Invoke-RestMethod "$env:ALPACA_API_BASE_URL/v2/account" -Headers @{ 'x-api-key' = $env:ALPACA_API_KEY }

# 2) Create Paper order (client-side idempotency)
$cid = "alpha-$(Get-Date -UFormat %Y%m%d%H%M%S)-$([System.Guid]::NewGuid().ToString('N').Substring(0,6))"
$body = @{
  symbol = "AAPL"
  side = "buy"
  type = "limit"
  time_in_force = "day"
  qty = 1
  limit_price = 10
  client_order_id = $cid
} | ConvertTo-Json
Invoke-RestMethod "$env:ALPACA_API_BASE_URL/v2/orders" -Method Post -ContentType "application/json" -Body $body -Headers @{ 'x-api-key' = $env:ALPACA_API_KEY }

# 3) List orders; confirm idempotency by client_order_id
Invoke-RestMethod "$env:ALPACA_API_BASE_URL/v2/orders" -Headers @{ 'x-api-key' = $env:ALPACA_API_KEY }
