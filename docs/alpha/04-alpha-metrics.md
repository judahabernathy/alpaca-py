# 04-alpha-metrics.md
version: 2025-09-08
status: canonical
scope: alpha/metrics

## Technical indicators

The classifier computes several metrics from historical price and volume data:

- **SMA50** – 50-day simple moving average of closing prices.
- **ATR14** – 14-period average true range computed from daily highs, lows and closes.
- **PIR (position-in-range)** – Position of last daily close in the range of the last 5 days, expressed as a percentage.
- **RVOL15** – Relative volume of the most recent 15-minute bar compared to the average of the previous 20 bars.
- **Bar position** – Position of the current daily close within today’s high-low range.

## Classification rules

Based on these indicators, the classifier assigns one or more tags:

- **Pullback Long** – Close above SMA50, PIR ≤ 25%, bar position ≥ 75%.
- **Breakout Long** – Close equals or exceeds the highest high of the last 20 days.
- **Breakdown Short** – No previous tags, close is below both SMA50 and the minimum low of the last 20 days.
- **Range** – PIR ≤ 15% or ≥ 85%.

The output field `confirm` indicates whether the 15-minute relative volume is strong (`"confirm_15m"`) or requires further confirmation (`"pending"`).
