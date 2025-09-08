# 11-alpaca-orders-quickref.md
version: 2025-09-08
status: canonical
scope: alpaca/orders

## Core fields
symbol, side, qty|notional, type, time_in_force, limit_price?, stop_price?, extended_hours?, client_order_id?

## Equities (whole shares)
- Types: market, limit, stop, stop_limit
- TIF: day, gtc (ioc/fok/opg/cls *by arrangement*)
- Sub-penny rule for limits: ≥$1 → 2 decimals; <$1 → 4 decimals

## Equities (fractional)
- TIF: day only
- Types allowed with day: market, limit, stop, stop_limit

## Extended-hours (equities)
- Allowed only with limit + time_in_force=day + extended_hours=true
- Brackets not supported in extended hours

## Crypto
- Types: market, limit, stop_limit
- TIF: gtc, ioc

## Options
- Types: market, limit
- TIF: day only
- No extended hours

## OTC assets
- Types: market, limit, stop, stop_limit
- TIF: day, gtc

## Bracket / OCO / OTO
- bracket: order_class="bracket", take_profit.limit_price, stop_loss.stop_price
- oco: order_class="oco"
- Extended hours: not supported
- TIF: day or gtc

## Trailing stop
- type="trailing_stop", trail_percent|trail_price
- Does not trigger outside regular hours
- TIF: day or gtc

### Example (equity bracket)
```json
{
  "symbol":"AAPL","side":"buy","qty":"10","type":"limit","time_in_force":"day","limit_price":"195",
  "order_class":"bracket",
  "take_profit":{"limit_price":"199"},
  "stop_loss":{"stop_price":"191"}
}
```

Sources:
- https://docs.alpaca.markets/docs/orders-at-alpaca
- https://docs.alpaca.markets/reference/createorderforaccount
- https://docs.alpaca.markets/docs/options-trading-overview
