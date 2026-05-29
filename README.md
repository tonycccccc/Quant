# Phase 0 — no API keys needed
python Code/main.py phase0 --dry-run

# Phase 0 — live (needs ANTHROPIC_API_KEY + PERPLEXITY_API_KEY in .env)
python Code/main.py phase0

# Phase 1 — uses real Alpaca historical data, skips market-hours check
python Code/main.py phase1 --test --dry-run

# Phase 2 — injects a real-data signal for any ticker (dry-run by default)
python Code/main.py phase2 --ticker NVDA
python Code/main.py phase2 --ticker TSLA --live   # submits real paper order

# Phase 3 — starts the WebSocket listener (blocks until Ctrl+C)
python Code/main.py phase3

# Portfolio status
python Code/main.py status

# Full pipeline
python Code/main.py run --test --dry-run