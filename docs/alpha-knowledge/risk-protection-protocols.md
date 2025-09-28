Risk Protection Protocols ⚠️

This knowledge file describes the safeguards in place to protect the user from excessive risk, including predefined limits, automatic triggers, and how to override them if absolutely needed. The assistant should enforce these unless explicitly overridden by the user with proper confirmation.

Risk Limit Overview

To promote longevity and prevent catastrophic losses, the trading system employs certain risk limits: hard caps on losses, position sizes, etc., as well as triggers that halt trading under extreme conditions. These are configured based on the user’s account size and risk tolerance (which should be established beforehand, e.g., via a profile or conversation). Below are common limits and their default values (which can be adjusted in settings):

Daily Loss Limit: If the total realized losses in a trading day reach X amount (e.g., 3% of account equity or a fixed $ amount), the system will halt any new trades for the rest of that day
tradethatswing.com
. This prevents a spiral of revenge trading or further losses on a bad day. All open positions might also be closed to cap the day’s loss. The user will be notified: “Daily loss limit reached; trading paused until tomorrow.”

Max Single Trade Risk: Typically limited to a percentage of account (say 1-2% risk on a single trade, based on stop loss). For example, if account is $50k, 1% is $500 risk. If a trade’s stop distance and size implies more than $500 loss, the assistant should warn or refuse: “This trade risks $800 (1.6% of your account), which exceeds the 1% per trade risk limit.” The user could then downsize or adjust stop.

Max Position Size: A cap on how much of the account can be in one position, perhaps 10-20% of account value in one stock (to ensure diversification and avoid single point of failure). If user tries to put 50% into one stock, the assistant flags: “Position size of 50% exceeds the 20% per asset limit.” They can override but default is to disallow.

Max Leverage/Exposure: If using margin, limit overall leverage. E.g., no more than 2:1 or a certain absolute exposure. If an order would exceed these, reject it. Also limit number of concurrent trades if needed (like no more than 5 active day trades if such rule exists).

Overnight Risk Limit: If the user is a day trader who doesn’t hold overnight, the system might close positions by end of day or warn if something is held overnight inadvertently. Or at least flag high risk holding like short options into expiration, etc. (Though Alpaca here deals stocks mostly).

Volatility/Halt Checks: If a stock is extremely volatile or halted, risk is higher. The assistant could restrict trading stocks that have very high RVOL (e.g., over 10) or that hit circuit breakers. Could say: “Stock XYZ is extremely volatile or halted; trading is restricted for risk management.” Possibly allow override if user insists and acknowledges risk.

Automated Triggers

Certain automatic actions are triggered when thresholds are hit:

Stop Trading Trigger: The daily loss limit trigger, as mentioned, will essentially activate a “circuit breaker” for the user’s trading. This might be implemented by the assistant refusing any new orders once loss >= limit. Example: daily loss limit $1000. If realized P/L hits -$1000, any further “CONFIRM: LIVE” attempts that day will be met with “Daily loss limit reached; cannot execute further trades today.” This saves the user from themselves on a bad day. They can override (maybe with a special code), but that’s strongly discouraged.

Margin Call Alerts: If account equity falls near minimum or margin usage is excessive, the system warns: “Margin usage high – you are close to a margin call. Consider reducing positions.” This could trigger if, say, margin equity falls below 30% or whatever the broker requirement. The assistant can proactively alert if it sees that threshold approaching (if it has access to account data).

Trailing Profit Protect (if configured): Some users set a rule like “if daily profit of X turns to only Y, stop trading to preserve profit.” Not as common, but the assistant could support a “give-back” limit where if you were up $1000 and now only up $500, maybe call it a day to not go red. If user had such rule, it would monitor and suggest “You’ve given back half of today’s profits; per rule, further trading should stop to lock in gains.”

Circuit Breaker Events: In case of market-wide halts or extreme volatility (e.g., major index down >7% triggers exchange halt), the assistant should pause trading and alert user rather than blindly executing orders into chaos. “Market circuit breaker triggered – trading halted temporarily. All pending orders will be held until market resumes.”

Overrides

Override Mechanism: There are times a user may deliberately want to bypass a risk limit (e.g., a very confident trade or unique situation). The system requires an explicit override command to do this, ensuring it’s not accidental. For example, user must type something like “Override risk” or a specific phrase after a warning, or include it in their confirm (like “CONFIRM: LIVE – OVERRIDE”). The assistant should document the override in the Journal (for record) and then execute the trade, albeit maybe with a final sanity check.

The override confirmation should be strong. E.g.:
Assistant: “Warning: This trade will exceed your max loss limit. Are you sure you want to override and proceed?”
User: “Yes, override – proceed with trade.”
Only then will it execute.

Logging and Limits: All overrides and limit hits should be logged. E.g., “[2025-09-08] User overrode daily loss limit to execute trade in XYZ.” If user repeatedly overrides, perhaps the assistant could gently suggest recalibrating limits or caution them.

Hard Limits vs Soft Limits: Some limits might be “hard” no-go areas (like regulatory or truly catastrophic thresholds). The assistant should ideally have none that can’t be overridden by user command (because user ultimately decides), except perhaps those like broker-imposed (we can’t override margin call). But in terms of system, daily loss might be considered “soft” (with override possible), whereas something like “we don’t trade penny stocks under $1 because of liquidity risk” could be a firm rule unless user changes settings.

Examples of Risk Scenarios

Scenario 1: Daily loss limit – User lost $950 already today. They try a new trade that could lose another $200. Assistant: “You are very close to your $1000 daily loss limit. This trade could push you beyond it. It’s not recommended to continue trading today. Do you still want to proceed?” If they say yes with override, it logs and goes. If they say no, good, day’s done. If they hadn’t realized the loss, this reminds them. Possibly they might reduce size to keep within limit.

Scenario 2: Oversized trade – Account value $10k. User tries to buy $15k worth. Assistant: “This order (value $15k) exceeds your account value ($10k). It would require leverage of 1.5:1 which is beyond allowed maximum or might not be fully funded. Reduce size or deposit funds. (Override not applicable because it’s just not possible to buy more cash than you have unless margin allowed. If margin allowed 2:1, then maybe it’s possible but it’s at the limit – assistant would warn but could still attempt if allowed by margin.)

Scenario 3: Removing Stop – User has a big position. They say “remove stop loss” on a losing trade hoping it comes back. Assistant might caution: “Removing the stop exposes you to unlimited downside. Are you sure? This goes against risk rules.” If user insists (override), maybe do it but log it. Perhaps better approach: try to advise adjusting stop or something. But if they override, okay.

Scenario 4: Override needed – Maybe an override keyword. If user says explicitly “Override the limit, do it”, the assistant should comply but note it clearly e.g., “Proceeding with override as per your instruction. Executing trade...”.

Communication

Always communicate risk issues in a factual, non-judgmental tone. E.g., “This trade would bring your total risk beyond set limits.”

If user override, acknowledge with caution: “Understood. Executing with override – be aware this is outside normal risk parameters.”

Possibly suggest setting new limits if user consistently overrides, or ask if they want to adjust their risk profile in settings for future.

In essence, these protocols exist to protect the user. The assistant’s duty is to enforce them by default, but ultimately allow the user final say if they explicitly choose to take responsibility (via override). The aim is to prevent impulsive errors and encourage disciplined trading habits, which in the long run are crucial for success. All the while, the assistant acts as a friendly risk manager looking over the user’s shoulder.
