FinRL (Reinforcement Learning) Parameters 🎛️

This file outlines how to use the FinRL tool for training or using a reinforcement learning trading agent, including required parameters, process flow, and usage guidelines. FinRL allows the assistant to develop AI trading strategies based on historical data and then apply them to live or future data.

Train-Predict Workflow

Using FinRL involves two main stages: training an agent on historical market data, and then using the trained agent to make predictions/trading decisions (often called the “trade” phase in examples). Typically, the workflow is:

Training Phase: Define an environment and let the RL algorithm learn from historical data. Parameters needed:

Tickers/Assets: Which instruments to trade (e.g., a list of stock symbols or an index). Could be single or multiple if doing portfolio training.

Data Period: Date range for historical data to train on (e.g., 2015-01-01 to 2020-01-01).

Frequency: Timeframe of trading (daily, hourly, 15-min bars, etc.). FinRL supports various frequencies but one must set resolution accordingly.

Indicators/States: What features to include (technical indicators like MACD, RSI, moving averages, etc., or fundamental features if available). FinRL often has a default state space including prices and tech indicators.

Algorithm: Choice of RL algorithm (e.g., DDPG, PPO, SAC, etc.). FinRL has multiple; if user doesn’t specify, use a default (like PPO or DDPG which is common for continuous action space).

Episodes & Hyperparams: Number of training episodes, learning rate, etc. Defaults can be used, but user can tweak for advanced usage.

Reward function: Usually defined by environment – often something like change in portfolio value, risk-adjusted returns, etc. FinRL has preset envs for maximizing returns.

The assistant should confirm these settings with the user before kicking off training, as training can take time and compute.

Once parameters are set, the assistant initiates training. This might take some time (in reality), but for our context, assume we either simulate quick results or abstract the time. The result of training is a trained agent (policy network) stored as a model.

Testing/Validation: (Optional) Often we test the agent on a validation set (a period after training data) to see how it would have done. This gives performance metrics (returns, Sharpe ratio, drawdown, etc.). FinRL’s pipeline often includes a “test.py” to simulate the trained model on new data. The assistant can report these metrics to user: e.g., “On validation (2021), the agent achieved 15% return vs S&P 10%, Sharpe 1.2, max drawdown 5%.” If performance is poor, that’s a flag.

Prediction/Trading Phase: Using the trained agent on live or forward data. This can mean:

Running the model on the latest market state to get an action (e.g., “Buy AAPL” or “allocate 30% to GOOG, 20% to cash, etc.” depending on design). FinRL’s trade.py essentially does a backtest on a subsequent period, but one could step it forward in real-time as well.

The assistant can query the agent given current market observations to see what it “recommends.” For instance, after training a single-stock trading agent, ask it: given today’s state (tech indicators values), would it hold, buy, or sell?

In a multi-asset environment, the action might be rebalancing portfolio weights.

The user might ask for these predictions or to actually deploy them in trading. Possibly the assistant would translate agent’s action into suggestions: “The RL agent suggests buying 50 more shares of X and selling Y” or “maintain current positions, no trade now.”

File/Code Management: FinRL likely uses config files or code to set up the above. The assistant should handle that behind scenes (maybe it uses the GitHub FinRL library code installed). For the user, we hide the code and just present outcomes.

Parameter Details

Environment Config: FinRL has presets like StockTradingEnv for single/multiple stocks, etc. Key parameters:

state_space: what data points form a state (like recent prices, indicators).

action_space: often continuous (like allocation % to each asset, or [-1,1] representing short to long position size). Or discrete (buy/hold/sell).

tech_indicator_list: list of technical indicators to include (ATR, SMA, RSI, etc.).

if_use_tech_indicator: boolean.

if_use_vix: etc., if including volatility index for context.

Possibly turbulence_threshold for risk management (some FinRL examples avoid trading in highly turbulent times).

Algorithm: Common ones in FinRL:

PPO (Proximal Policy Optimization) – good default for many cases.

DDPG (Deep Deterministic Policy Gradient) – often used in FinRL examples.

A2C, SAC, TD3 – others available. If user has preference or mention, choose accordingly.

The assistant might default to one like A2C or PPO if not told.

Training duration: FinRL training can be time-consuming. The assistant should set expectations: “Training a model on 5 years of data might take a few minutes.” If in our environment it’s quick, fine. If slow, maybe we simulate progress or ask to proceed.

Model output: after training, a model file can be saved. The assistant might refer to it like “trained model saved as Agent_XYZ.pkl” (if needed). But user might not care about file, just results.

Performance Metrics: key ones:

Cumulative Return (total return over test period).

Sharpe Ratio (return vs volatility).

Max Drawdown (largest peak-to-trough drop).

Win rate (if applicable).

If multiple assets, maybe a breakdown of performance vs benchmark.

The assistant should present these in a readable format, maybe a small table or bullet points: “Backtest Results: Return = 18%, Sharpe = 1.1, Max Drawdown = -8%.” Also possibly mention if it beat something: “This outperformed SPY which returned 10% in same period.”

Usage Policy

Due to computational intensity and complexity, FinRL usage is typically done sparingly and purposefully:

When to use: Perhaps when user explicitly requests strategy development, backtesting or if they ask “Can you have an AI figure out how to trade this?” The assistant might propose training an RL model if it makes sense. Or if user connected some custom environment. It’s not for everyday quick advice due to overhead.

Resource limits: If training on very long histories or many assets, it could be slow or memory heavy. The assistant should possibly limit by default (like train on last 3 years daily data for 1-5 assets) unless user wants more. Also, not do it concurrently with a bunch of other tasks to avoid overload.

Permission: It might be wise to confirm with user before kicking off heavy training: “Shall I train a reinforcement learning model on this data? It may take a couple of minutes.” This avoids surprise delays.

No Overuse: The assistant shouldn’t retrain models constantly on every minor query due to time. Ideally, if user is experimenting, maybe reuse a model or keep one updated rather than new every time. Possibly store the model or results in Journal memory for reuse. For now, probably one-off as needed.

Interpretation: Remind user that these models optimize for historical patterns; results aren’t guaranteed in the future. There’s overfitting risk.

Possibly caution: “The agent did well in backtest, but markets can change. Use its signals with caution and perhaps combine with other analysis.”

No Black Box Blind Trust: The assistant should not just say “The RL says do X, so do X.” It should still contextualize or even combine with Alpha Classifier or user’s strategy logic. But RL could come up with unconventional strategies (like lots of small trades). The assistant needs to translate that to actionable insight.

Limitations: FinRL is mostly for research/backtest. Real-time deployment might need continuous monitoring which is beyond a quick Q&A. If user expects the agent to auto-trade live continuously, that’s out-of-scope for the assistant’s session which is interactive. We can do one-step recommendations though.

Policy compliance: Ensure it doesn’t break any rules (shouldn’t, it’s just analysis). Also do not produce the raw code unless user specifically requests. Focus on results and strategy derived.

Example Dialogue:
User: “Can we use AI to create a trading strategy for AAPL?”
Assistant: “Yes. We could train a reinforcement learning agent on Apple’s historical data. For example, using daily price data from 2010-2020 to learn when to buy/sell. Shall I proceed with training such a model?”
User: “Okay, do it.”
Assistant: [Trains model] -> “The RL agent has been trained. In backtesting from 2021-2023, it achieved a 25% total return, vs 20% for simply holding AAPL, with lower volatility (Sharpe 1.0). It tended to buy after pullbacks and sell after modest rises, effectively swing trading. For instance, it often bought on 5% dips and sold after 5-7% rebounds. Currently, according to the model’s latest evaluation, it would hold (no trade) since AAPL is in the middle of a recent range. If a pullback to around $x occurred, it might buy.”

This shows how to summarize without overloading details of the training.

In summary, FinRL is a powerful but heavy tool. The assistant uses it for deeper strategy tasks, explains the outcomes clearly (with metrics and possibly example actions), and respects the time/complexity by confirming usage and not overdoing it.
