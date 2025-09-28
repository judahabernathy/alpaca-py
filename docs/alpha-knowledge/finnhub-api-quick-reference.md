Finnhub API Endpoints Quick Reference 🔍

This reference maps common data tasks to the relevant Finnhub API endpoints and required parameters. The assistant uses these endpoints to retrieve financial data when asked. All endpoints require the API token (handled internally), so the user only needs to specify the query, not the token.

Below are typical use-cases and how to fulfill them with Finnhub:

Get Current Stock Price (Quote):
Endpoint: /quote?symbol={ticker}
Description: Returns real-time quote data for a stock ticker. The JSON includes fields: c (current price), h (high of day), l (low of day), o (open price), pc (previous close), and t (timestamp).
Usage: Use when user asks for the latest price or daily change of a stock.
Example: For AAPL, GET .../quote?symbol=AAPL -> yields c:150.0, o:148.0, h:151.0, l:147.5, pc:149.0 etc. The assistant would parse and say: “AAPL is $150.00, up 0.7% from yesterday’s $149.00 close.” (Percent change can be derived: (c - pc)/pc *100%).

Get Historical Prices (Candles):
Endpoint: /stock/candle?symbol={ticker}&resolution={res}&from={unix_time_start}&to={unix_time_end}
Description: Fetches historical OHLCV data (Open, High, Low, Close, Volume) for the given time resolution. resolution can be 1,5,15,30,60 (minutes), D (daily), W (weekly), M (monthly). The time range is specified in Unix timestamps (seconds since epoch).
Usage: For charts, technical analysis, or backtesting data.
Example: To get daily candles for 2023 YTD for GOOGL: set from=1672531200 (2023-01-01), to=1704067199 (2023-12-31). GET /stock/candle?symbol=GOOGL&resolution=D&from=1672531200&to=1704067199. The response lists arrays for t (time), o, h, l, c, v. The assistant can then calculate trends, or feed into indicators.

Company News (Recent):
Endpoint: /company-news?symbol={ticker}&from={YYYY-MM-DD}&to={YYYY-MM-DD}
Description: Retrieves news articles related to the company in the given date range. Each article includes headline, summary, date, source, etc.
Usage: Use when user asks "Any recent news on X?" or for catalyst analysis.
Example: User: “What’s new with TSLA?” -> The assistant fetches last week’s news: from=2025-09-01&to=2025-09-08. It then might summarize: “Several news items: e.g., a new model launch rumor (source CNBC), and an analyst upgrade by Morgan Stanley
stockstotrade.com
.” The assistant should filter/summarize rather than listing raw JSON.

General Market News:
Endpoint: /news?category={category}
Description: Provides latest market news in a category such as general, forex, crypto, merger.
Usage: If user asks for market news or what's driving markets today.
Example: “What’s the market news?” -> GET /news?category=general. Then pick a few top headlines: “Stocks rally as inflation cools; Fed hints at pause – Reuters.” and so on.

Fundamental Data (Company Profile / Metrics):
Profiles: /stock/profile2?symbol={ticker} returns basic info like company name, industry, CEO, etc. Useful if user asks “What does this company do?”
Metrics: /stock/metric?symbol={ticker}&metric=all gives a host of fundamental metrics (PE ratio, EBITDA, margins, etc)
interactivebrokers.com
.
Usage: For fundamental analysis questions like “What’s the P/E of GOOG?” or “How’s AAPL’s financial health?” The assistant can fetch metrics and quote some key ones
columbia.edu
: e.g., “AAPL P/E is 28, profit margin 22%
interactivebrokers.com
.”
Possibly narrow metrics: Finnhub allows metric=all or specific metric groups like metric=price etc. Usually all covers most.

Earnings Data:
Latest Earnings (Surprises): /stock/earnings?symbol={ticker} gives recent earnings (actual vs estimate)
stackoverflow.com
. Use if asked “Did XYZ beat earnings?”
Example: If user asks about last earnings of NFLX: GET /stock/earnings?symbol=NFLX might yield an array of recent quarters with actual, estimate. The assistant can say: “Last quarter, NFLX reported $3.20 EPS vs $3.00 expected – a beat
dlthub.com
.”
Earnings Calendar: There’s also /calendar/earnings?from=...&to=... for upcoming earnings in a date range, or specifically earnings date for a ticker via profile2 or other endpoints. But for single ticker, profile2 includes ipo and sometimes next earnings date under nextEarningsDate if available. If user asks “When does XYZ report earnings?”, we might use alternative approach (Finnhub had an endpoint like /stock/earnings? with future flag? Or /calendar I can search via symbol filtering). Quick hack: If we only have historical, maybe mention “expected in about X date” if known schedule.

Economic Data / Other Assets:
Finnhub can also fetch forex rates, crypto, indices, etc:

Forex: /forex/rates?base=USD or use symbol like “OANDA:EUR_USD” in /quote or /candle.

Crypto: symbols like “BINANCE:BTCUSDT” in quote/candle endpoints for crypto prices.

Macroeconomic indicators: Finnhub has endpoints (e.g., /economic-data?symbol=...) for things like CPI, unemployment, etc. If needed and user asks macro question.

Indices: ^GSPC (S&P 500) or ^NDX possibly via symbol in quote (I think Finnhub uses e.g. ^GSPC for S&P, or might require ^ encoded).

For simplicity, if user asks “What’s S&P 500 doing?”, we try /quote?symbol=^GSPC (if supported) or use index futures (like “US500” might be in forex).

Search Symbols:
Endpoint: /search?q={query}
Description: If user provides a company name or partial ticker and we need to find the actual symbol.
Usage: The assistant might use this when user says “quote Tesla” but didn’t give ticker TSLA. It will search “Tesla” -> find TSLA. Or if ambiguous name, it might find multiple.
Example: search?q=Tesla returns TSLA and maybe other similar names. The assistant picks TSLA (the major one) and proceeds. If multiple and unclear, might ask user which.

Analyst Recommendations:
Endpoint: /stock/recommendation?symbol={ticker}
Description: Provides analyst consensus like number of Buy, Hold, Sell ratings.
Usage: If user asks “What do analysts say about XYZ?”
Example: It might show e.g. 10 Buy, 5 Hold, 1 Sell. The assistant can summarize: “Analyst consensus is bullish: of 16 analysts, 10 Buy, 5 Hold, 1 Sell.”

Insider Transactions, etc.: Finnhub has some specialty endpoints (/stock/insider-transactions, /stock/splits, /corporate-news, etc). Use as needed if user specifically asks for those (rare).

Parameter Mapping Quick Table:
TaskEndpointRequired ParamsNotes
Current price & daily change/quotesymbolReturns c (current), pc (prev close), etc. Use to compute % change.
Intraday/Historical prices/stock/candlesymbol, resolution, from, tools/Use for charts, indicators. Need Unix time range.
Company recent news/company-newssymbol, from, to (YYYY-MM-DD)Limit ~last 30 days max. For older news beyond that, Finnhub might not return or limited.
Market news (general, crypto, etc.)/newscategorycategory=general for broad market; other categories for specific domains.
Company profile info/stock/profile2symbolBasic company info like industry, name, CEO, exchange.
Key financial metrics/stock/metricsymbol, metricmetric=all for full set (incl P/E, P/B, margins, etc).
Earnings surprises/history/stock/earningssymbolLast 4 quarters of actual vs estimate and dates.
Upcoming earnings date/calendar/earningsfrom, tools/Filter results for the symbol of interest. Alternatively use profile2’s earningsCalendar.
Symbol lookup (by name)/searchq (company name or partial)Use when user gives name instead of ticker.
Analyst recommendations/stock/recommendationsymbolReturns trends of analyst ratings.
Forex/Crypto rates/forex/rates or /quoteFor forex: base currency/forex/rates gives multiple pairs. Or treat like stock: e.g. symbol=“BINANCE:BTCUSDT” in quote.
Macro indicators/economic endpointsTypically an indicator symbole.g. /economic-data?symbol=US:CPI for US CPI. (Finnhub has specific codes for each).

The assistant doesn’t need to show endpoint names to user, but should cite sources appropriately when giving data (in our environment, we often transform into some output, so citing Finnhub might be like
interactivebrokers.com
if from known docs, but in general we might not have a direct line to cite for a dynamic query. We could say “according to Finnhub data” as a source mention if required by UI).

Important: The assistant should handle date to Unix conversion for candles, ensure correct resolution (e.g., for >1 year daily data, resolution D is fine; for intraday 1-min maybe last 30 days limit). It should also check if data returned is empty (e.g., if market closed for that date or symbol invalid).

Example Use in Assistant Response:
User: “What’s the price of Amazon stock and how’s it today?”
Assistant: (calls /quote?symbol=AMZN) -> gets say current 134.50, prev close 130.00 ->
Reply: “Amazon (AMZN) is trading at $134.50 right now, about +3.5% higher than yesterday’s close of $130
columbia.edu
. (It opened at $132 and climbed on above-average volume.)”

User: “Show me Tesla’s chart for the last month.”
Assistant: (calls /stock/candle? symbol=TSLA, resolution=D, from=some 1 month ago timestamp, to=now) then it might generate or describe:
“The past month TSLA ranged roughly $240 to $290. It’s currently near $265. After a mid-month dip, it rebounded; overall up ~5% over the month. (Chart not shown in text, but description given.)”

User: “Any news on MSFT?”
Assistant: (calls /company-news? symbol=MSFT, from=2025-09-01, to=2025-09-08)
Then summarizes top 1-2 headlines:
“Yes – recent news for Microsoft includes a report that they are investing in AI chip development (source: Bloomberg, Sept 5)
stockstotrade.com
, and an announcement of a dividend increase on Sept 2. No negative news of note in the past week.”

This reference helps ensure we use correct endpoints and parameters to gather info and respond accurately.
