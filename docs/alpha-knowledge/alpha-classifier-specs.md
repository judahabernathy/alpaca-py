Alpha Classifier Specs 🤖

This file details the Alpha Classifier tool, including what inputs it expects, what outputs it produces, and how to interpret those outputs. The Alpha Classifier is a machine-learning model integrated into the assistant’s workflow for evaluating trade setups or market conditions quantitatively.

Purpose

The Alpha Classifier’s job is to analyze a given trading scenario or data set and output an alpha score or category that indicates the potential for abnormal returns (“alpha”) in that scenario. In simpler terms, it’s trying to gauge how good a trade or idea might be. This helps the assistant back up its recommendations with an objective model’s perspective.

It’s called a classifier because it may bucket the outcome (e.g., “Strong Buy, Weak Buy, Neutral, Weak Sell, Strong Sell”) or provide a numeric score along a range which corresponds to such categories.

Input Specification

The classifier can take in various forms of input depending on context:

Technical Features: It might ingest recent price data, indicators, volumes, volatility metrics, etc. For instance, it could use values like RSI, moving average slopes, volume surge info, etc., pertaining to the asset in question. The assistant will assemble such data points from Finnhub or internal calculations when invoking the classifier.

Fundamental or News Features: If the scenario involves news or fundamentals (like an earnings report sentiment, or a sector rotation), the classifier might also accept summarized sentiment or fundamental metrics. For example, it could take as input “earnings beat by 10%, stock gapped +5%” or a sentiment score of recent news headlines.

Trade Context: If evaluating a specific trade setup, context such as entry price, stop, target might be input so the classifier knows the risk-reward. Or if comparing multiple stocks, their relative strengths can be input to pick best.

In practice for the assistant, a lot of this is abstracted – the assistant just calls the classifier with whatever features it has on hand about the situation. For user transparency, we can say it considers technical and possibly fundamental factors of the asset.

Example Input Case: User asks “How strong is the setup for XYZ?” The assistant gathers that XYZ has RSI 72, just broke out of a range, RVOL 2.5, moderate earnings growth, etc. These are fed to the classifier.

Note: The user doesn’t directly see this input; it’s internal. But we ensure to explain output with relevant factors.

Output Specification

The Alpha Classifier returns either a numeric score or a category label (or both).

Alpha Score: Typically a float between 0 and 1 (or 0 to 100 if percentile). Higher means more bullish (higher expected alpha), lower means more bearish (negative expected alpha). For instance: 0.9 might mean a strong buy signal, 0.5 neutral, 0.1 strong sell. Sometimes it could output negative values if convention set that negative = short favorability, but likely 0-1 scaled where <0.5 implies underperformance expectation.

Classification Label: It may map the score into human-friendly classes. e.g., Score >0.8 = “Strong Long”, 0.6-0.8 “Moderate Long”, 0.4-0.6 “Neutral/Hold”, 0.2-0.4 “Moderate Short”, <0.2 “Strong Short”. This is an example mapping. The actual model may have, say, 5 classes or a continuous score.

We will assume a 5-tier interpretation: Strong Buy, Buy, Hold, Sell, Strong Sell corresponding to very bullish to very bearish outlooks. Or similarly worded categories.

Additional outputs: The classifier might also provide some sub-scores or rationale features (if it’s an explainable model). For instance, it might highlight which factors influenced it (like “momentum +0.3, fundamentals +0.1, volatility -0.05, overall score 0.35” – just hypothetical). If such detail is available, the assistant can use it to explain why the score is what it is. If not, the assistant infers based on known inputs (like “Classifier likely liked the strong momentum and volume”).

Score Meaning and Guidance

We interpret the classifier’s output for the user in plain language, also citing the numeric for precision if needed:

Strong Buy (Score ~0.8-1.0): The model sees very high positive alpha potential. This suggests the trade or asset has a high probability of outperforming / a very favorable setup. In practice, the assistant will say something like “The AI classifier rates this as a Strong Buy (score 0.85) – indicating a high-confidence bullish signal.”

Buy (Score ~0.6-0.8): Moderately positive. A good setup, albeit not top-tier, but still expected to yield above-average returns. The assistant might say “moderately bullish”. If user is looking for longs, it’s a yes, if being picky maybe only go for strong buys.

Hold/Neutral (Score ~0.4-0.6): The model expects average or uncertain outcome – no clear edge. Could mean mixed signals. “The classifier is neutral on this – score 0.5, suggesting limited edge (it might perform in line with general market or it’s too uncertain).”

Sell (Score ~0.2-0.4): Moderately negative. Suggests likely underperformance or a poor setup. If user was considering a long, that’s a caution sign; if considering short, maybe a mild opportunity. “Classifier sees bearish bias – perhaps best to avoid or consider short, score 0.3.”

Strong Sell (Score ~0.0-0.2): The model is strongly negative on it. Possibly a failing company or very weak technicals. “It’s giving a strong sell signal (score 0.1), implying high confidence that this will underperform or drop.”

Exact numeric cutoffs can be refined, but those ranges are conceptually how we map it. The assistant shouldn’t present the raw score as gospel but as one input. Also, mention if the difference is small. E.g., if something is score 0.55 vs another 0.50, not a big difference – both neutralish. But 0.85 vs 0.4 is clearly different.

Confidence & Reliability: The assistant may note that the classifier is a tool, not guaranteed. E.g., “The classifier’s confidence is high, but it’s based on historical patterns – always wise to also consider current fundamentals/news.” If score is near 0.5, basically it’s saying no edge – which itself is information (maybe skip trade).

Integration into Workflow

The assistant will call the classifier typically when the user asks for an analysis like “Should I buy this?” or “How does this setup look?” or even on its own to cross-check its human analysis.

If the classifier output conflicts with other analysis, the assistant might say “Our technical analysis looked good, but the AI model isn’t as convinced (score only 0.45). This discrepancy suggests caution.” Or vice versa. It helps add a second opinion.

The classifier might also be used to rank a list of candidates. E.g., user says “Which of these 5 stocks is best?” The assistant can get scores for all and say “The classifier favors stock A (0.78) over others (next highest was 0.6), indicating A has the best projected alpha.”

When not to use blindly: If a situation has factors the classifier might not have seen (e.g., an unprecedented news or it’s a penny stock with unusual conditions), the assistant should note the limitations. “The classifier output might not fully account for the current meme-stock frenzy, so take it with salt.”

Example Interpretation to User

User: “I’m thinking of going long ACME Corp. What’s the outlook?”
Assistant: “Based on the current data, the Alpha Classifier gives ACME a score of 0.82, which falls in a ‘Strong Buy’ range. That indicates the setup is quite favorable
investopedia.com
. This high score likely reflects ACME’s strong upward momentum (it recently broke out to 52-week highs on high volume) and solid fundamentals (earnings growth last quarter). In other words, the AI model has high confidence in ACME’s upside potential. Of course, no model is 100% — but this is a positive sign.”

Another:
User: “How about Beta Inc? It’s been dropping; do we see a rebound?”
Assistant: “The classifier score for Beta Inc. is only 0.25, a ‘Sell’ signal. It suggests continued weakness — likely because Beta’s technical trend is still down (it’s trading below key moving averages) and there’s no sign of reversal momentum yet. In model terms, that’s a pretty bearish outlook, so catching a falling knife here might be risky.”

Finally:
If user asks to explain classifier:
Assistant can say it looks at dozens of indicators and patterns learned from historical data to output that score, in a user-friendly manner: “It’s basically summarizing a bunch of factors into one number for how good or bad this looks.”

Remember, the assistant should never just say “classifier says buy so do it.” It should incorporate it as one piece of advice, albeit important. Always pair with context like why it might be saying that if possible (momentum, fundamentals, etc.). That makes the recommendation more transparent and trustworthy.
