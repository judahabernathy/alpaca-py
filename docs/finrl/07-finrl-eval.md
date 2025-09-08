# 07-finrl-eval.md
version: 2025-09-08
status: canonical
scope: finrl/evaluation

## Purpose
Provide a quick reference for evaluation utilities included with the FinRL‑actions repository.  These functions assist in assessing strategy performance and overfitting risk but are not directly exposed via API endpoints.

## Metrics

### Sharpe ratio
`sharpe(returns: np.ndarray, rf: float = 0.0) -> float`

Computes the annualized Sharpe ratio of a series of returns with respect to a risk‑free rate `rf`.  It subtracts the risk‑free rate from the returns, then divides the mean by the sample standard deviation and scales by √252 (trading days per year).  If the standard deviation is zero, the function returns 0.

### Deflated Sharpe ratio
`deflated_sharpe(sr: float, n_trials: int, skew: float = 0.0, kurt: float = 3.0) -> float`

Estimates a confidence adjustment for a Sharpe ratio given multiple trials.  The approximation penalizes the Sharpe ratio by `√(log(n_trials))` and maps the result to a 0–1 confidence via `tanh`.  When `n_trials` ≤ 1, it returns a baseline confidence based solely on the ratio.

### Probability of Backtest Overfitting (PBO)
`pbo_from_ranks(in_sample_ranks, out_sample_ranks) -> float`

Given arrays of in‑sample and out‑of‑sample ranks, computes the fraction of cases where the best in‑sample model performs worse than the median out‑of‑sample performance.  Higher values indicate a greater risk of overfitting.

## Notes
These functions live in the `ml/eval/metrics.py` module and are intended for offline analysis.  They are not part of the FinRL‑actions API but may inform strategy research.

Sources:
- Evaluation metric implementations
