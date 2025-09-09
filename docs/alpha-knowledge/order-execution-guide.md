Order Execution Guide 💱

This knowledge file outlines how to format and handle trade orders, including advanced order types (brackets, OCO, trailing stops), price drift considerations, and common rejection scenarios. The assistant uses these rules to ensure trades are executed or previewed correctly and safely.

Bracket Orders

What they are: A Bracket Order is essentially a “three-in-one” order setup: it consists of an initial entry order (to buy or sell), paired with two exit orders – one to take profit at a specified target, and one to stop loss at a specified level
quantifiedstrategies.com
. The two exit orders are OCO (One-Cancels-Other; see below), so that if one executes, the other is automatically canceled. Bracket orders help automate trade management by bracketing the position with an upper profit exit and a lower protective exit.

Syntax & Usage:
When placing a bracket order through the assistant (particularly via Alpaca if supported), the user (or assistant’s preview) should specify:

Entry: side (buy/sell), quantity, and entry price (or market). E.g., “Buy 100 shares of XYZ @ $50.00”.

Take Profit: the price at which to sell if favorable (for a buy order) or buy back if favorable (for a short). E.g., “Take-profit at $55.00”.

Stop Loss: the price at which to exit if the trade goes against you. E.g., “Stop-loss at $48.00”.

The assistant might format it as: “Buy 100 XYZ @ $50, with a stop at $48 and target $55 (bracket order).” This indicates the bracket clearly. Some platforms accept a single bracket order command including all three; others require sending the main order and then attaching exits. The assistant will handle appropriately via the Alpaca API (which does allow bracket orders by providing take_profit and stop_loss parameters in the order).

Example: “Buy 10 AAPL at $150, target $157, stop $145 (bracket).” This means a limit buy at 150. If filled, an OCO pair is created: a sell limit at 157 (profit) and a sell stop at 145 (loss). If the price hits 157 first, the profit order sells the shares and the 145 stop is canceled. If price falls to 145 first, the stop triggers and the profit order is canceled.

Benefits:

Automatically protects downside and secures upside without needing constant monitoring.

Defines your risk (stop distance) and reward (target) upfront, making risk/reward calculation clear.

No emotion – the orders execute as planned, which helps discipline.

Considerations:

Ensure the stop and target are at logical levels (e.g., beyond noise for stop, before major resistance for target).

The initial order must fill for the bracket exits to be active. If the entry doesn’t execute (e.g., a limit that never fills), the bracket doesn’t activate. Typically bracket orders are not placed unless entry fills (some APIs simulate the whole bracket as one, handling partial fills accordingly).

Partial fills: If only part of the entry fills, bracket exit order quantities usually adjust or only protect the filled portion. The assistant should confirm how Alpaca handles partial bracket fills (usually, yes – if 50 out of 100 shares fill, the bracket exits will be for 50 shares).

Syntax nuances: In plain language, we’ll just describe it clearly. The assistant doesn’t expect the user to know code-like syntax, we interpret their intent. E.g., if user says “Buy 50 ABC at 10, sell at 12, stop 9”, we treat that as a bracket order request. In previews, always label each part to avoid confusion.

OCO Orders

What it is: One-Cancels-the-Other (OCO) is a pair of orders linked such that if one order executes, the other is automatically canceled. Bracket orders utilize an OCO for their exits, but OCO can also be used in other contexts, like two alternate entries or exits.

Use cases:

Take Profit vs. Stop Loss: As above, where one exit will make the other irrelevant.

Straddle Entries: For instance, placing two entry orders around the current price – e.g. a buy-stop above resistance and a sell-stop below support (a breakout up or break down scenario). You only want one to execute – whichever happens first cancels the other, to avoid being double-positioned.

Multiple Targets: Suppose you hold a position and want to exit either if it hits a certain high or falls to a certain low. You place two sell orders – one high, one low – as OCO. Whichever hits closes the trade, the other is scrapped.

How to specify: When the user describes two orders that are linked mutually exclusively, the assistant will interpret that as OCO. For example: “If I’m filled long at $50, set a sell limit $55 OCO with a stop $48.” That is a bracket scenario. Or a user might say: “I want to set a buy order at $20 and a sell short at $18 on XYZ; whichever triggers, cancel the other.” That’s an OCO entry pair for a breakout of a range $18-$20. The assistant in a preview could write: “OCO Entry Orders: Buy stop 100 XYZ @ $20.00, and Sell stop 100 XYZ @ $18.00 – whichever fills first cancels the other.”

On Alpaca or similar, this might involve placing an OCO group. If direct OCO isn’t supported, the assistant must simulate by monitoring (but Alpaca does support bracket which is a subset of OCO usage).

Important: The orders in an OCO pair must be of opposite effect on the position (one closes or opens in opposite directions). You wouldn’t OCO two orders that together would compound risk – the whole point is only one should ever execute. So the assistant will ensure the logic is correct (e.g., don’t OCO two buy orders unless one is meant as alternative entry and you don’t want both).

Example of OCO outside bracket: You own 100 shares of XYZ at $100. You’d be happy to sell at $110 for profit, but if it falls to $95 you want out to stop further loss. You place an OCO: Sell 100 @ $110 (limit) and Sell 100 @ $95 (stop). If price rockets to 110, you sell profit, and the stop order at 95 is automatically canceled. If instead it plunges to 95, the stop triggers a sale, and the 110 order is canceled. This is effectively the manual creation of a bracket’s exit.

Trailing Stop Orders

What it is: A Trailing Stop order is a type of stop order that dynamically moves with the price. Instead of a fixed stop price, it’s set a certain distance (in price or percentage) from the current price and “trails” it. For a long position, a trailing stop will move up as the price goes up, but stay put if the price goes down. For a short, it will move down as price falls (locking more profit), but not move if price rises.

For example, a 5% trailing stop on a long stock initially placed when stock is $100 would sit at $95 (5% below). If the stock rises to $110, the trailing stop would move up to $104.50 (still 5% below that new high). If the stock then starts to fall, the stop doesn’t move down – it stays at $104.50. If price falls back to $104.50 or below, the stop triggers a sell. Essentially, it locks in a $4.50 profit in that case.

In dollar terms, one might say “trailing stop $2” meaning always 2 dollars behind the peak price reached. E.g., stock at $50, stop at $48. Stock goes to $55, stop ratchets to $53. Drop to $53 triggers it.

Syntax & usage via assistant:
Users might say “place a trailing stop 5% below market” or “trail by $1”. The assistant will interpret that accordingly. In an order preview, it may be phrased: “Set a trailing stop at 5%.” However, trailing stop orders need some specifics: you typically submit a trailing stop order with either a trail amount (like $ or %) and optionally a not-to-exceed limit if using trailing stop limit.

For simplicity, if not specified as limit, we assume trailing stop market order: it will sell at market when triggered by trailing price. If user says “trailing stop limit”, then we’d need a limit offset too. If not, a normal trailing stop suffices.

Alpaca format: Alpaca’s API allows something like order_type="trailing_stop", trail_percent=5 or trail_price=$X. The assistant will map user’s instruction to that.

Behavior:

Only moves in one direction. For longs: move upward with new highs, never down. For shorts: move downward with new lows, never up.

It’s executed server-side by broker once set; you don’t need to manage it manually (the broker tracks the high price and adjusts).

When to use:

To protect profits without setting a fixed exit. E.g., “I want to ride this trend as long as it goes up, but if it pulls back more than 5% from its peak, sell me out.” A trailing stop does exactly that.

Volatile stocks: trailing by a percentage accounts for volatility scale. If stock doubles, your trailing stop distance also increases in absolute terms, since 5% of a larger price is more points. That can prevent getting stopped out too early and capture more profit.

As a compromise between not having a stop and taking profit too early. It lets winners run but defines how much give-back is allowed.

Example: You buy at $100. Instead of targeting $120 or something, you set a 10% trailing stop. Price goes to $130 over two months (nice run!). The trailing stop would’ve ratcheted from $90 (initially) up to $117 (10% off the $130 high). Now if price starts dropping and hits $117, your stop triggers and you sell, locking in ~17% gain. If it kept going to $150, the stop would climb to $135 (still 10% off) and so on.

Caveats:

Trailing stops can be whipsawed in choppy markets. If your trail is too tight for the volatility, normal fluctuations hit it and close your position prematurely. Setting the trail distance appropriately (often a bit more than average volatility or a technical swing amount) is key. E.g., trailing 2% on a stock that fluctuates 3% daily likely will trigger often.

Overnight gaps: A trailing stop doesn’t guarantee the exact trail distance execution if a sudden gap occurs. E.g., if stock closed at $100, stop trailing at $95, but next morning opens at $90, you’ll be stopped at $90 (gap through it). So risk can exceed trail distance in gaps.

Not all platforms allow trailing stops on all securities or outside regular hours. The assistant should note if any limitation (Alpaca supports trailing for regular hours, I believe).

Complexity: If user already has a stop and wants to modify it to trailing, we need to cancel the fixed stop and replace with trailing. The assistant should clarify if needed.

Price Drift Policy

What it is: The “price drift” policy addresses what to do if the market price moves significantly between the time a trade idea is generated (or preview presented) and the time of execution. Essentially, it’s a tolerance for slippage or movement beyond which the original plan might be invalid. The assistant should alert or reconfirm if price drifts beyond a set threshold before execution.

Default threshold: Let’s say 0.5% or a certain number of ticks as a default, unless user specifies. For example, if the assistant prepared a preview “Buy at $100” and user confirms a minute later, but now price is $103 (3% higher), that’s a big drift. Executing at $103 vs planned $100 significantly alters the trade’s profile (higher entry, worse reward/risk). According to policy, the assistant should not blindly execute; it should pause and ask the user to confirm the new price or adjust the order.

If the drift is small, like from $100 to $100.20, that’s likely fine (0.2%). Could execute with maybe a note. But define threshold: maybe >0.5% or > some absolute amount like $0.50 (depending on stock price). Perhaps user can set tolerance too (“only buy up to $101” etc.).

How assistant applies it:

During Live Confirmation: When user says “CONFIRM: LIVE”, the assistant will quickly re-fetch the latest price (via Finnhub or Alpaca data) for the asset. Compare to preview price. If difference > tolerance, respond with something like: “The current price of XYZ has moved to $103 from the planned $100 (+3%). This exceeds our allowed drift. Would you like to proceed at market anyway, adjust the order price, or cancel?” This ensures user is aware and can make an informed decision.

The user might then say “Okay place it anyway” or “No, set a limit at $100” or “Cancel”. The assistant then follows that instruction.

For Stop/Target orders: If placing a stop/limit some time after initial discussion, similar logic – e.g., if user asked to set a stop at $50 but now stock is $45 due to quick drop, that stop might be useless (since price already way below). The assistant should alert that “Price has already moved past the intended stop level – order not placed. You may need a new plan.”

Rationale: This policy is to prevent executing trades at prices far different from what the user saw or intended, which could lead to unexpected loss or missed the good entry. It’s a safety net akin to a reconfirmation threshold.

Example scenario: The assistant shows a preview at 9:31 AM: Buy 100 ABC @ $50, stop $48, target $55. The user is slow to respond; at 9:35 they say confirm. But ABC released news at 9:33 and it spiked to $52. That’s +4%. The assistant, seeing this, doesn’t just send a market order at $52. It warns: “ABC is now $52.00 (4% above the planned $50 entry). This is beyond the slippage limit. I recommend adjusting or canceling. Shall I still place the order (perhaps as a limit), or would you prefer to wait?”. This gives user a chance to re-evaluate. Maybe they decide not to chase it. Or they say “Place a limit at $50.50 in case it pulls back.” The assistant then places a limit order instead of a market.

If user unreachable or it’s an automated context: In a fully automated mode (not our scope but imagine), one might either cancel or adjust strategy (some systems convert a chase beyond drift into a limit at original or skip trade). Since here we always have user to confirm, we lean on user input.

The assistant’s compliance: by default, always dry-run, so price drift likely addressed at confirm time. It should also be mindful if the user delays confirmation by hours, the context could change a lot (maybe news came out). The assistant can even proactively mention “This preview was based on price $X at time Y, if much time passes, it may no longer be valid.”

Order Rejection Rules

Sometimes an order cannot or should not be executed. The assistant should check for these conditions and handle them gracefully by rejecting (or not placing) the order and explaining why. Common rejection scenarios include:

Outside Trading Hours: User tries to place a regular trade when the market is closed. For instance, 8pm EST for U.S. stocks (unless they have extended hours trading on). Alpaca by default executes in market hours (unless specified extended). The assistant will inform: “Market is currently closed; the order will be queued for next open.” Or if that’s not desired, ask user if they want to place an extended hours order (if supported) or cancel. If user specifically says it’s okay (and broker supports), we mark the order as extended_hours. Otherwise, typically don’t place a market order overnight.

Insufficient Buying Power: If the user’s account doesn’t have enough cash or margin to cover the trade. Alpaca’s API would return an error. The assistant should catch that and tell user: “Order rejected: insufficient funds to buy 100 shares of XYZ.” Possibly suggest alternatives (like reduce size). The assistant would know approximate buying power if it queried account – in preview it might not check, but on execution if error returns, handle it.

Invalid Symbol or Asset: If user enters a ticker that doesn’t exist or is not tradable (typo, delisted, etc.), the API will error. The assistant: “The symbol ‘ABCD’ is not recognized or not tradable. Please check the ticker.”.

Order Parameters Error: E.g., negative quantity, or a stop price above limit price in a stop-limit order, etc. The assistant should validate basic logic: if user says “sell at 60 stop, 55 target” but they are short (sell order so these are reversed), or they said “buy with stop 52, target 50” which makes no sense (stop below target for a buy – that would mean guaranteed loss). It should clarify likely confusion: maybe they mixed up stop/target. The assistant can ask or correct politely.

Risk Limit Violations: If there are pre-defined risk rules (from Risk Protection file) – e.g., this trade would exceed daily loss limit or position limit. The assistant should refuse to execute unless override is given. E.g., “This trade risks $5,000 which exceeds your $2,000 risk limit – order not placed.” If user insists with an override command, then proceed but log that override.

Duplicate Orders / Position Conflict: If user already has a position or order and this new one conflicts (like they’re long 100 shares and place another sell short 100 – which would flip them, or they accidentally send same order twice). The assistant should detect if possible. E.g., if they say “Buy 100 XYZ” and a minute later again “Buy 100 XYZ” (maybe not realizing first was filled), we might ask “You already have an order/position in XYZ, do you intend to increase it?” If it seems accidental, clarify. If intentional, proceed.

Minimums and Increments: Some brokers have min price for stocks (e.g., can’t trade stocks under $1 on margin or at all). Or options not allowed here (Alpaca doesn’t do options). If user tries to trade something not allowed, reject: “Order not allowed: [Reason].”

Technical Issues: If Alpaca API fails (network issue, etc.), the assistant should tell user it couldn’t place order due to technical error and not assume it went through. It should encourage trying again later or checking account.

Account Restrictions: If account is flagged (like pattern day trading rules if under equity requirement), or trading halted instrument, etc. For instance, if stock is halted, the order won’t execute – assistant: “Trading in XYZ is currently halted, order cannot be executed at this time.”

In all cases of rejection, the assistant’s tone: informative, not scolding. Provide the reason and possible next step. E.g., insufficient funds → maybe suggest reducing quantity or depositing more. If outside hours → either queue or wait. Risk limit → require override or modify trade.

Logging: It should log in Journal any serious rejection especially if user needs to be aware later (like if they ask “why didn’t my trade execute?” the log notes we didn’t place it due to X).

Example dialogue:
User: “Buy 1000 XYZ at market now.” (User’s account only has $500, XYZ is $10, needs $10k).
Assistant: “Order not placed – buying 1000 XYZ (~$10,000) exceeds your available funds. You could reduce quantity or deposit more capital.”

Another:
User: “Sell 50 ABC now” (but user holds 0 ABC and no short selling available).
Assistant: “Cannot place sell order – you don’t have ABC shares to sell (and account not enabled for short selling).”

One more:
User: “Short 5 GME at $4” (typo, probably meant $40 since GME is $40, $4 would be bizarre low). Assistant might catch “$4 is far from current price $40, are you sure?” If not a typo, maybe they think it’ll drop massively? But probably a mistake. Clarify rather than just accept a wildly off limit that won’t fill.

In summary: The assistant double-checks orders for feasibility and safety. If something triggers a rule above, it stops and informs the user instead of blindly sending to broker. This ensures more robust, error-free execution.
