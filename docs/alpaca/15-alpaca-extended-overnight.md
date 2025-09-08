# 15-alpaca-extended-overnight.md
version: 2025-09-08
status: canonical
scope: alpaca/extended

## Sessions (equities)
- Overnight: 8:00 pm – 4:00 am ET, Sun–Fri
- Pre-market: 4:00 am – 9:30 am ET, Mon–Fri
- After-hours: 4:00 pm – 8:00 pm ET, Mon–Fri

## How to place extended-hours orders
- Equities only.
- Must be: type=limit, time_in_force=day, extended_hours=true.

## Not supported in extended hours
- Bracket/OCO/OTO legs.
- Trailing stop triggers.

## Fractional during extended
- Limit orders in extended hours supported for fractional per platform updates.

## Overnight tradability flags
- Assets API exposes fields to identify overnight eligibility (e.g., `overnight_tradable`, `overnight_halted`) when overnight trading is enabled.

## Market data for overnight
- WebSocket feeds include `v1beta1/boats` and `v1beta1/overnight` for overnight prints.

Sources:
- https://docs.alpaca.markets/docs/extended-hours-trading
- https://docs.alpaca.markets/docs/real-time-stock-pricing-data
