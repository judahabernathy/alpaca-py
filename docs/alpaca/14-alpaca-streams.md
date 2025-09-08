# 14-alpaca-streams.md
version: 2025-09-08
status: canonical
scope: alpaca/streams

## Trading stream (orders, fills, account)
- Endpoints: wss://{api|paper-api}.alpaca.markets/stream
- Auth message:
```json
{"action":"auth","key":"<KEY_ID>","secret":"<SECRET>"}
```
- Listen:
```json
{"action":"listen","data":{"streams":["trade_updates"]}}
```
- Common event values: new, partial_fill, fill, canceled, expired, done_for_day, replaced.
- Less common: accepted, rejected, pending_new, stopped, pending_cancel, pending_replace, calculated, suspended, order_replace_rejected, order_cancel_rejected.

## Market Data stream
- Endpoint schema: wss://stream.data.alpaca.markets/{version}/{feed}
- Feeds: v2/sip, v2/iex, v2/delayed_sip, v1beta1/boats, v1beta1/overnight
- Auth: headers or message {"action":"auth","key":"...","secret":"..."} within 10s.
- Subscribe / unsubscribe:
```json
{"action":"subscribe","trades":["AAPL"],"quotes":["AAPL"],"bars":["*"]}
```
```json
{"action":"unsubscribe","quotes":["AAPL"]}
```
- Test stream: wss://stream.data.alpaca.markets/v2/test  (use symbol FAKEPACA)

## Message format
- Array of JSON objects. Control messages have `T` in {success, error, subscription}.

## Errors (data WS)
- 401 not authenticated, 402 auth failed, 404 auth timeout, 405 symbol limit exceeded, 406 connection limit exceeded, 407 slow client, 409 insufficient subscription, 410 invalid subscribe action.

## Notes
- Paper trading stream may use binary frames. Handle JSON and MessagePack.

Sources:
- https://docs.alpaca.markets/docs/websocket-streaming
- https://docs.alpaca.markets/docs/streaming-market-data
- https://docs.alpaca.markets/docs/real-time-stock-pricing-data
