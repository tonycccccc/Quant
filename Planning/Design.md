# 📄 System Specification & Execution Plan: AI-Driven Tech Wave Sniper Trader (Alpaca Edition)

## 1. Executive Summary & Design Philosophy
This document specifies the architecture for an autonomous, paper-trading system designed to execute mid-term swing trading strategies (2 days to 3 weeks holding periods) targeting 8 high-volume software and hardware giants in the tech sector.

### Core Investment Philosophy
> **"Decouple Research from Execution: Pre-clear fundamentals at dawn, shoot with sub-millisecond technical triggers during market hours, and enforce ironclad server-side risk management."**

The system bypasses complex, fragile multi-agent environments in favor of a **linear, decoupled, and asynchronous event-driven architecture** that enforces strict capital protection:
1. **What to Trade**: A strictly managed 8-stock watchlist focused entirely on high-alpha tech giants. Watch list will be provided to you in Methodology/Rules/Watchlist.md.
2. **When to Research**: Every morning at 9:30 AM EST via a Cron Job. Perplexity AI (`sonar-reasoning`) acts as a digital analyst to check macro catalysts, news, and earnings, granting or denying a `daily_clearance` permit. The research output should be dumped to the Metholody/Research/{company_name}.md file. Files should be compressed and summarized accordingly.
3. **When to Execute**: In real-time when the conditions in Methodology/Rules/Strategy.md are met. If a stock has a clearance permit, entry is instantaneous.
4. **How to Manage**: Absolute mathematical enforcement of position sizing via Alpaca Server-Side Bracket Orders (One-Triggers-OCO). Zero options leverage. Please refer to Methodology/Rules/Portfolio.md for the management policy.

How the system works?
1. Pre-market Research Layer: the research agent should start iterating the watchlist, research news about each company through Perplexity API, and calls LLMs to judge if this company is eligible for trading today (set SQLite daily_clearance).
2. Research Layer: Once this layer is called, the trading agent should play as a professional wall-street trader and use LLM's excellent reasoning and analysis capability to do deep fundamental vetting and rank the companys in the watchlist, favoring the companies that show 'strong buy' signal. Then use technical analysis to calculate the entry point, profit cash-out point and loss end point for each company, append to the rank list.
3. Execution Layer: Once trading agent gives research and entry point for each company. Execution layer will rank 8 companies with buy confidence. The system fetches historical data directly via Alpaca's Historical Market Data API every 30 minutes, calculates technical overlays specified by Strategy.md, and instantly judges structural breakouts. If the buy condition is confirmed, the execution layer should send real orders to Alpaca immediately.
4. Tracking Layer: All transactions (buy & sell) need to be logged, along with other technical stats at the point of buy and sell, so our trading agent/research layer can keep optimizing the trading strategy. Each transaction also needs to be send to discord webhook so the account owner can be notified.

Implementation roadmap:

Phase 0: Pre-Market Clearance Engine (The Sniper Prep)
Objective: Build a python script that when main program is launched, starts to run market analysis and research, saving decisions to SQLite to bypass intraday LLM latency.

Claude Prompt:

Please build Phase 0 of our automated system: The Pre-Market Clearance Engine. It must iterate through our 8-stock watchlist table in SQLite. For each stock, it must call the Perplexity API (sonar-reasoning) to search for news, sentiment, macro catalysts, and earnings dates over the past 48 hours. Pass that parsed text to an LLM (OpenAI/Anthropic) to reason and suggest if the market sentiments shows bullish or bearish signal for the company. This layer should enforce specific strategy rules: if earnings are within 3 days or news reveals structural damage, output daily_clearance = 0. Otherwise, output daily_clearance = 1. Update the SQLite rows with the clearance integer and a 2-sentence summary. Conclude by sending a beautiful "Daily Battle Plan" grid embed to the Discord #-trade-alert channel.

Phase 1: Intraday 30-Minute Polling & Local Analytics Engine
Goal: Build the core time-based daemon loop that fetches historical bars from Alpaca Data API, computes professional stats natively in Python, detects signals, and call LLMs to judge if a buy condition is confirmed.

Claude Prompt:

Please build Phase 1 of our system. Create a python interface that when triggered, uses the official alpaca-py HistoricalDataClient to pull the day's intraday 30-minute bars for all 8 stocks in our universe. Compute the technical stats specified in Strategy.md, calls LLMs to judge if current position is a good entry point and determine if a buy signal is confimed. When a signal hits, verify if the ticker has daily_clearance == 1 in our SQLite database, and pass approved alerts directly to our internal execution pipe while logging skipped events. 
<!-- Eventual goal: Please build Phase 1 of our system. Create an asynchronous execution scheduler using APScheduler or a persistent asyncio sleep loop that executes exactly every 30 minutes during US Equity Market hours (9:30 AM - 4:00 PM EST). When triggered, it must use the official alpaca-py HistoricalDataClient to pull the day's intraday 30-minute bars for all 8 stocks in our universe. Compute the technical stats specified in Strategy.md, calls LLMs to judge if current position is a good entry point and determine if a buy signal is confimed. When a signal hits, verify if the ticker has daily_clearance == 1 in our SQLite database, and pass approved alerts directly to our internal execution pipe while logging skipped events. -->

Phase 2: Alpaca Execution & Server-Side Bracket Configuration
Goal: Wire up the official alpaca-py trading module to deploy risk-mitigated Bracket Orders automatically.

Claude Prompt:

Please implement Phase 2 of our system. Create a robust PortfolioManager class that connects to Alpaca via the official alpaca-py library utilizing the TradingClient initialized for Paper Trading (paper=True). When an approved technical signal is passed from our Phase 1 polling worker, the manager must fetch the current liquidation value from the Alpaca account, execute our 1.5% Risk-Per-Trade formula to compute precise entry shares, and verify we have less than 5 open positions. If validation clears, submit a structured OrderClass.BRACKET order to Alpaca: a Primary Market Order to buy, a nested StopLossRequest at exactly -3.5% of the entry close price, and a nested TakeProfitRequest at +7% of the entry close price. Write the resulting Alpaca Parent Order ID, shares, and timestamps into the SQLite active_positions table safely. Please send the trade confirmation alert immediately to the Discord #-trade-alert channel. Meanwhile, please log all technical stats we fetched from Alpaca and we computed ourselves to the Logs/trading_stats.csv. 

Phase 3: Trailing Maintenance & Markdown Status Projection
Goal: Build continuous asset state maintenance

Claude Prompt:

Please implement Phase 3 of our system. Integrate an async listener using Alpaca's TradingStream WebSocket client to monitor trade update streams in real-time. When a bracket leg order triggers a 'fill' status (meaning either our automatic hard stop or +7% profit target was hit on Alpaca's servers), update the corresponding SQLite record status. Please always maintain our portfolio status through a dedicated portfolio_status.md file. Please match all gains/loss with the technical stats we logged in Logs/trading_stats.csv and asked the trading agent to reflect on the transactions and add new rules to Strategy.md if needed. 

Design Ideas:
LLM is a decision-support layer, not a decision-maker. All trade execution is mostly rule-driven and statistically deterministic.