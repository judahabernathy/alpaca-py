# FinRL adapter

## Endpoints (OpenAPI v1.3.11 only)
- POST /train, GET /predict, POST /backtest, GET /status/{job_id}, GET /signal, GET /risk, POST /run_all, GET /healthz. /order exists but is a reject stub.

## Usage
- Default: remote predict/backtest for fast metrics; local train only on explicit request; consent line required.
- Jobs: /train and /backtest return job_id; do not block; check /status/{job_id} only on request.

## Evidence mapping
- finrl_signal: {symbol, version, side, signal, confidence?}
- finrl_metrics: {total_reward, sharpe, win_rate, trade_steps, params?}

## Presentation
- Human line: “RL: {side, conf} · Sharpe {x} · DD {y}”.

## Watchlist EV
- Weight RL with Classifier, Liquidity, Trigger, Portfolio, Exposure (weights live in tests or code).

## Limits
- RL never overrides TTL, sizing, session, or risk gates.
