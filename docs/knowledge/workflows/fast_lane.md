# Fast lane workflow

## Trigger
- “Confirm and submit”

## Atomic steps
1) Preflight (freshness ≤10s) → compute size in TradePreview.
2) Validate order intent (EXT policy, exposure, halt, slippage).
3) Submit to Alpaca; enforce idempotency.
4) Auto‑journal (LIVE only).

## Output
- FastLaneResult {route_decision, trade_preview, order_validation, submitted_order?, journal_entry, elapsed_ms}

## Flags
- dry_run:true executes steps 1–2 only, returns bundle with submitted_order=null.

## Notes
- No background work; synchronous single action.
