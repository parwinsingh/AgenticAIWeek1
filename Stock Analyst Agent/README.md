# Stock Analyst Agent

An AI-powered Streamlit app that tracks and analyzes a personal US stock
portfolio from a transaction-history CSV. Live/historical prices come from
`yfinance`; portfolio Q&A and insights are powered by OpenAI.

## Project structure

```
app.py                     # Streamlit entry point (page config, sidebar, tabs)
utils/
  data_processing.py       # CSV ingestion + yfinance price fetching (cached)
  portfolio_math.py        # FIFO cost basis, lifetime metrics, XIRR
  llm_agent.py              # OpenAI client, AI summary, streaming chat
components/
  tab_upload.py             # Tab 1: Data Upload
  tab_portfolio.py          # Tab 2: Consolidated Portfolio View
  tab_performance.py        # Tab 3: Historical Performance
  tab_chat.py                # Tab 4: AI Analyst (Chat)
sample_transactions.csv     # Example CSV to try the app with
```

## Setup (uv only)

```bash
cd "Stock Analyst Agent"
uv sync                      # creates .venv and installs all dependencies
cp env.demo .env             # then edit .env and add your OPENAI_API_KEY
uv run streamlit run app.py
```

You can also paste your OpenAI API key directly into the sidebar at runtime
instead of using a `.env` file — get a key at https://platform.openai.com.

## CSV format (Tab 1)

Required headers, one row per transaction:

| ticker | date       | transaction_type | quantity | price  |
|--------|------------|-------------------|----------|--------|
| AAPL   | 2023-01-15 | Buy               | 10       | 135.21 |
| AAPL   | 2024-01-20 | Sell              | 4        | 193.00 |

`sample_transactions.csv` in this folder (also downloadable from Tab 1) is
ready to use for a quick test.

## Notes

- Cost basis (Tab 2) is computed with **FIFO** lot matching: sells consume the
  oldest open buy lots first, and the average cost basis shown is the
  weighted-average price of whatever lots remain open.
- **XIRR** (Tab 3) treats every Buy as a negative cash flow, every Sell as a
  positive cash flow, and the current portfolio value as a final positive
  cash flow as of today, then solves for the annualized rate that zeroes the
  NPV of all of them.
- Live and historical prices are cached (5 min / 30 min TTL) to stay within
  yfinance rate limits.
