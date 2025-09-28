# 09-alpaca-urls-auth.md
version: 2025-09-08
status: canonical
scope: alpaca/urls-auth

## REST base URLs
- Live Trading: [https://api.alpaca.markets](https://api.alpaca.markets)
- Paper Trading: [https://paper-api.alpaca.markets](https://paper-api.alpaca.markets)
- Market Data (HTTP): [https://data.alpaca.markets](https://data.alpaca.markets)

## Auth (HTTP)
Send headers on every private request:
- APCA-API-KEY-ID: <key_id>
- APCA-API-SECRET-KEY: <secret_key>

### Quick test
GET https://{api|paper-api}.alpaca.markets/v2/account

## Trading WebSocket (account & order updates)
- wss://api.alpaca.markets/stream  (live)
- wss://paper-api.alpaca.markets/stream  (paper)

Authenticate after connect:
```json
{"action":"auth","key":"<KEY_ID>","secret":"<SECRET>"}
```

Subscribe:
```json
{"action":"listen","data":{"streams":["trade_updates"]}}
```

## Market Data WebSocket
URL schema:
- wss://stream.data.alpaca.markets/{version}/{feed}
- Feeds: v2/sip, v2/iex, v2/delayed_sip, v1beta1/boats, v1beta1/overnight

Auth options:
- HTTP headers: APCA-API-KEY-ID / APCA-API-SECRET-KEY
- Or message:
```json
{"action":"auth","key":"<KEY_ID>","secret":"<SECRET>"}
```

Subscribe example:
```json
{"action":"subscribe","trades":["AAPL"],"quotes":["AAPL"],"bars":["AAPL"]}
```

## Notes
- Paper and live use different domains and different credentials. Verify before sending orders.

Sources:
- https://docs.alpaca.markets/docs/authentication-1
- https://docs.alpaca.markets/docs/websocket-streaming
- https://docs.alpaca.markets/docs/streaming-market-data
- https://docs.alpaca.markets/docs/real-time-stock-pricing-data
