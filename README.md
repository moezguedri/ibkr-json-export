# 📊 IBKR to JSON Exporter  
### Advanced multi-source market data exporter for ChatGPT-powered investment analysis

This project provides a robust Python script that connects to
Interactive Brokers (IBKR) via ib_insync, retrieves market data, and
exports it into a clean JSON format.

---

## 🔍 Overview

This script connects to **Interactive Brokers (IBKR)** and **Yahoo Finance** to generate rich, structured **JSON files** designed for **ChatGPT-assisted portfolio analysis**.

Each JSON contains **everything needed** for deep investment insights:

- 📈 5-year daily historical data  
- 🕒 6-month 1-hour intraday data  
- 🧮 Fundamentals (valuation, growth, cashflow, margins)  
- 🧑‍💼 Analyst sentiment & price targets  
- 🗓 Earnings timestamps & growth metrics  
- ⚡ Implied volatility  
- 📉 Short interest  
- 🏦 Insider & institutional ownership  
- 🎯 Your open limit orders  
- 💼 Your current position  
- 👤 Your investor profile  
- 🏭 Peer companies for comparison  

This JSON becomes a **complete offline knowledge package** that ChatGPT can analyze like a real financial analyst.

---

## 🧩 Data Sources

### 🟦 IBKR (via `ib_insync`)
Used for broker-level precision market data:

- 5 years of daily OHLCV  
- 6 months of intraday OHLCV (1h)  
- Your open BUY orders  
- Your current positions  
- Optional raw XML fundamentals (if available)

### 🟩 Yahoo Finance (via `yfinance`)
Used for all fundamentals and advanced metrics:

- PE, Forward PE, PEG, PS, PB  
- Dividends  
- Cashflows  
- Revenue & earnings growth  
- Analyst recommendations & price targets  
- Short interest  
- Implied volatility  
- Insider/institutional ownership  
- Earnings timestamps  

Yahoo Finance works **even with IBKR paper accounts**.

---

## 🎯 Why This JSON Format?

It enables ChatGPT to perform **complete investment analysis** without needing internet access.

Each JSON file includes:

| Section | Purpose |
|--------|----------|
| `price_daily_5y` | Long-term trend & volatility |
| `intraday_1h_6m` | Recent momentum |
| `fundamentals.yahoo` | Core business metrics |
| `fundamentals.ibkr` | Raw XML snapshot (optional) |
| `your_position` | Your entry, position size |
| `open_orders` | Pending buy entries |
| `peer_symbols` | Comparable competitors |
| `investor_profile` | Your constraints & preferences |
| `meta_description` | Documentation baked into the file |

This format allows ChatGPT to answer questions like:

- “🧠 How should I manage this position for 6 months?”  
- “💸 Should I add more at current levels?”  
- “📉 Where are realistic buy-the-dip zones?”  
- “⚔️ How does this stock compare to its peers?”

---

## 🛠 Prerequisites

Install dependencies:

```bash
pip install ib_insync yfinance pandas tzdata
py -c "import ib_insync; print('OK')"
py -c "from zoneinfo import ZoneInfo; print(ZoneInfo('US/Eastern'))"
```

## Usage

``` bash
python ibkr_to_json.py
```

Outputs JSON files into `ibkr_json_output/`.
