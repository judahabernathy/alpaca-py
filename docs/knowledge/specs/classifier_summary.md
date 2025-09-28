# Alpha Classifier summary

## Auth
- Header `X-API-Key`

## Endpoints
- /healthz (GET), /classify (POST)

## Limits
- symbols 1..50; rvolMode default '20bar'

## Output
- ClassifyItem {symbol, close, sma50, atr, pir_pct, rvol15, candidates, confirm}
