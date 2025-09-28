# Routing map

## Aliases → Target
- ord.new, order.create, buy.*, sell.* → alpaca
- ord.cancel, order.cancel → alpaca
- acct.snap, account.get → alpaca
- pos.list, position.* → alpaca
- wl.*, watchlist.* → alpaca
- quote.get, candles.get, fundamentals.get → finnhub
- cls.run, classify.list → classifier
- rl.train, rl.predict, rl.backtest, rl.status, rl.signal, rl.risk, rl.run_all → finrl
- journal.create, journal.bulk, journal.update, params.save, params.load → journal
- schema.route, schema.preview, schema.validate, schema.journal, schema.fast → router

## Router artifact order
- Always emit (echo‑only): RouteDecision → TradePreview → OrderValidation → (submit) → JournalEntry
- Fast lane emits bundle: FastLaneResult {route_decision, trade_preview, order_validation, submitted_order?, journal_entry, elapsed_ms}

## Secrets
- Never send secrets in payloads or notes.

## Notes
- Router is schema‑carrier only.
