# Watchlist workflow (batch)

## Batch mode
- Inputs: 15–50 symbols. Classify all; run light RL on all.

## Groups
- Actionable now | Watchlist added | Discarded.

## Auto‑add reasons
- ClassifierGateFail, EarningsWindow, SessionClosed, Halted, StrategyNearMiss, ExposureLimit.

## Record
- WatchlistEntry {strategy, symbol, reason, added_ts}; journal each add.

## TTL
- Do not auto‑watchlist on transient quote TTL failures; refresh first.
