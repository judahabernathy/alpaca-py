# Classifier adapter

## API
- POST /classify {symbols[], rvolMode:'20bar'} → ClassifyItem[].

## Mapping to internal AlphaClassifierOutput
- features.trend = close / sma50 − 1
- features.rvol = rvol15
- features.pir = pir_pct / 100
- features.atr = atr
- features.confirm = confirm
- label = Sell if candidates contains “Breakdown Short”; else Buy if contains “Breakout Long” or “Pullback Long”; else Neutral
- score = clamp(0,1, 0.5 + 0.6*trend + 0.2*(rvol − 1) + 0.1*(1 − abs(pir_pct − 50)/50))
- If label=Buy and score ≥ 0.85 → label=StrongBuy

## Gate
- Proceed only if label ∈ {StrongBuy, Buy} AND score ≥ 0.70.

## Errors
- 401/429/5xx → return Neutral, score=0.50; add error.kind and remedy; watchlist as needed.

## Batching
- Chunk to ≤50 per call.
