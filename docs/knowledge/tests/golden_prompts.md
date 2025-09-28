# Golden prompts (deterministic)

Use Paper. Expect specific Router echoes or FastLaneResult bundles.

## Tone checks (applies to all cases)
- Answer-first line present.
- `Next moves:` present with 1–3 items.
- `Pro move:` one actionable line present.
- When blocked: `What to watch:` present with one concrete step.
- No fluff; concise.

1) Fast lane RTH limit: AAPL buy; fresh TTL; expect submitted_order!=null; journal created.
2) EXT bracket reject: user requests bracket in EXT; expect validation.ok=false; reason includes EXT rule; next moves suggest RTH bracket.
3) Quote TTL stale: expect prompt to refresh Preflight; no order.
4) Earnings gate block: symbol inside window; expect watchlist add with reason=EarningsWindow.
5) Per‑name exposure cap breach: expect validation false; reason ExposureLimit; no submit.
6) Daily loss halt: block new entries; reason includes threshold.
7) Slippage guard breach: limit too far; validation false.
8) Idempotent retry: same client_order_id; second create returns existing.
9) Batch 20 symbols: groups emitted; watchlist auto‑adds for near‑misses.
10) Journaling on submit: verify one journal with mapped fields.
11) Journals PATCH: exit update writes exit_time/exit_price/outcome.
12) Memory sync: start chat, memory missing → load; annotate “Memory: missing”.
13) FinRL predict: include RL summary line; no blocking if RL unavailable.
14) FinRL train async: returns job_id; no blocking; status only on request.
15) Classifier error 429: degrade; Neutral score; watchlist.
16) EXT limit policy enforced via OrderValidation.
17) RTH default template applied when user omits TIF/type.
18) Watchlist “Rescore now” surfaced in batch.
19) Fast lane dry_run:true returns bundle w/o submitted_order.
20) Telemetry: elapsed_ms present with integer values.
