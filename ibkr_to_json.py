#!/usr/bin/env python3
"""
IBKR → JSON exporter for ChatGPT analysis.

- Connects to IBKR TWS / Gateway with ib_insync
- For each symbol:
    * 5 years of daily OHLCV (IBKR)
    * 6 months of 1-hour OHLCV (IBKR)
    * Fundamentals from Yahoo Finance (primary)
    * Fundamentals from IBKR (best-effort, may fail)
    * Your current position (shares, avg cost) from IBKR
    * Your open BUY orders for that symbol from IBKR
    * Your investor profile (configured in this script)
    * A static peer list for basic comparison context
- Outputs one JSON file per symbol.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from ib_insync import IB, Stock, Contract, BarData
import yfinance as yf

# ===================== USER CONFIGURATION ===================== #

# IBKR connection parameters
IB_HOST = "127.0.0.1"
IB_PORT = 7496          # TWS port (live or paper)
IB_CLIENT_ID = 1        # Change if you use multiple clients

# Symbols to export: (symbol, exchange, currency)
SYMBOLS = [
    ("NVDA", "SMART", "USD"),
    # ("MSFT", "SMART", "USD"),
    # ("AMZN", "SMART", "USD"),
    # ("META", "SMART", "USD"),
    # ("GOOGL", "SMART", "USD"),
    # ("COST", "SMART", "USD"),
    # ("TSM", "SMART", "USD"),
]

# Output directory for JSON files
OUTPUT_DIR = "ibkr_json_output"

# Historical data settings
DAILY_5Y_DURATION = "5 Y"
DAILY_5Y_BAR_SIZE = "1 day"

INTRADAY_6M_DURATION = "6 M"
INTRADAY_6M_BAR_SIZE = "1 hour"

WHAT_TO_SHOW = "TRADES"   # For stocks this is usually fine


# === INVESTOR PROFILE CONFIGURATION ==========================================
# Edit this dictionary to reflect YOUR personal investor profile.
# This profile will be embedded into every JSON file so that ChatGPT
# can adapt its analysis to your situation and preferences.
INVESTOR_PROFILE = {
    "risk_tolerance": "medium",           # 'low' | 'medium' | 'high'
    "investment_horizon_years": 5,        # Typical horizon in years (e.g. 3, 5, 10)
    "monthly_investment_usd": 500.0,      # How much you plan to invest per month
    "max_single_position_pct": 0.25,      # Max % of portfolio in a single stock (e.g. 0.25 = 25%)
    "preferred_style": "growth",          # 'income' | 'balanced' | 'growth'
    "notes": (
        "Edit this section to reflect your personal profile. "
        "This helps ChatGPT reason about allocation, risk, and position sizing."
    ),
}

# === PEER GROUPS (for basic comparison context) ===============================
# These are static peer lists per symbol. They are purely informational and
# can be extended or modified as you wish.
PEERS_BY_SYMBOL: Dict[str, List[str]] = {
    "NVDA": ["AMD", "AVGO", "TSM"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "AMZN": ["MSFT", "GOOGL", "META"],
    "META": ["GOOGL", "AMZN", "SNAP"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "COST": ["WMT", "TGT"],
    "TSM": ["INTC", "ASML", "AMD"],
}

# ============================================================================ #


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def connect_ib() -> IB:
    ib = IB()
    print(f"Connecting to IBKR at {IB_HOST}:{IB_PORT} (clientId={IB_CLIENT_ID})...")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")
    print("Connected to IBKR.")
    return ib


def qualify_contract(ib: IB, symbol: str, exchange: str, currency: str) -> Contract:
    contract = Stock(symbol, exchange, currency)
    contracts = ib.qualifyContracts(contract)
    if not contracts:
        raise RuntimeError(f"Could not qualify contract for {symbol} ({exchange}, {currency}).")
    qualified = contracts[0]
    print(f"Qualified contract: {qualified}")
    return qualified


def bars_to_daily_list(bars: List[BarData]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for b in bars:
        if isinstance(b.date, str):
            # 'YYYYMMDD' or 'YYYYMMDD  HH:MM:SS'
            date_str = b.date.split(" ")[0]
            dt = datetime.strptime(date_str, "%Y%m%d")
        else:
            dt = b.date
        date_iso = dt.strftime("%Y-%m-%d")
        result.append(
            {
                "date": date_iso,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume) if b.volume is not None else None,
            }
        )
    result.sort(key=lambda x: x["date"])
    return result


def bars_to_intraday_list(bars: List[BarData]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for b in bars:
        if isinstance(b.date, str):
            parts = b.date.split()
            if len(parts) == 2:
                dt = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S")
            else:
                dt = datetime.strptime(b.date, "%Y%m%d")
        else:
            dt = b.date
        datetime_str = dt.strftime("%Y-%m-%d %H:%M")
        result.append(
            {
                "datetime": datetime_str,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume) if b.volume is not None else None,
            }
        )
    result.sort(key=lambda x: x["datetime"])
    return result


def fetch_historical(
    ib: IB,
    contract: Contract,
    duration: str,
    bar_size: str,
    what_to_show: str = WHAT_TO_SHOW,
    use_rth: bool = False,
) -> List[BarData]:
    print(f"Requesting historical data: duration={duration}, barSize={bar_size}, whatToShow={what_to_show}")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    print(f"Received {len(bars)} bars.")
    return bars


# ---------- Fundamentals via Yahoo Finance (primary) ---------- #

def fetch_yahoo_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Fetch key fundamentals using yfinance.
    This does NOT require any IBKR fundamentals subscription.
    Includes: valuation, margins, growth, basic earnings info, risk signals.
    """
    print(f"Fetching fundamentals from Yahoo Finance for {symbol} ...")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
    except Exception as e:
        print(f"Error fetching Yahoo fundamentals for {symbol}: {e}")
        return {
            "source": "yahoo",
            "note": f"Failed to fetch fundamentals from Yahoo Finance: {e}",
        }

    def get_float(key: str) -> Optional[float]:
        v = info.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    fundamentals = {
        "source": "yahoo",
        "note": "Fundamentals fetched from Yahoo Finance via yfinance. Fields may be null if not available.",
        # Valuation
        "market_cap": get_float("marketCap"),
        "pe": get_float("trailingPE"),
        "forward_pe": get_float("forwardPE"),
        "peg": get_float("pegRatio"),
        "ps": get_float("priceToSalesTrailing12Months"),
        "pb": get_float("priceToBook"),
        "dividend_yield": get_float("dividendYield"),
        # Earnings & growth
        "earnings_timestamp": get_float("earningsTimestamp"),
        "earnings_timestamp_start": get_float("earningsTimestampStart"),
        "earnings_timestamp_end": get_float("earningsTimestampEnd"),
        "earnings_quarterly_growth": get_float("earningsQuarterlyGrowth"),
        "revenue_growth": get_float("revenueGrowth"),
        # Margins & profitability
        "revenue_ttm": get_float("totalRevenue"),
        "net_income_ttm": get_float("netIncomeToCommon"),
        "gross_margin": get_float("grossMargins"),
        "operating_margin": get_float("operatingMargins"),
        "profit_margin": get_float("profitMargins"),
        "roe": get_float("returnOnEquity"),
        # Balance sheet / leverage
        "debt_to_equity": get_float("debtToEquity"),
        "current_ratio": get_float("currentRatio"),
        "quick_ratio": get_float("quickRatio"),
        # Cash flow
        "free_cashflow": get_float("freeCashflow"),
        "operating_cashflow": get_float("operatingCashflow"),
        # Analyst consensus
        "analyst_recommendation": info.get("recommendationKey"),
        "target_mean_price": get_float("targetMeanPrice"),
        "target_high_price": get_float("targetHighPrice"),
        "target_low_price": get_float("targetLowPrice"),
        "number_of_analyst_opinions": get_float("numberOfAnalystOpinions"),
        # Risk signals / sentiment
        "short_percent_of_float": get_float("shortPercentOfFloat"),
        "shares_short": get_float("sharesShort"),
        "shares_short_prior_month": get_float("sharesShortPriorMonth"),
        "implied_volatility": get_float("impliedVolatility"),
        "held_percent_insiders": get_float("heldPercentInsiders"),
        "held_percent_institutions": get_float("heldPercentInstitutions"),
    }
    return fundamentals


# ---------- Fundamentals via IBKR (best-effort) ---------- #

def fetch_ibkr_fundamentals(ib: IB, contract: Contract) -> Dict[str, Any]:
    """
    Keep a best-effort call to IBKR fundamentals.
    - Tries multiple reportType values.
    - If all fail (no subscription, paper account, etc.), returns a placeholder
      with an explanatory note.
    """
    report_types = [
        "ReportSnapshot",
        "ReportsFinSummary",
        "ReportsFinStatements",
        "ReportRatios",
    ]
    last_error: Optional[str] = None

    for rt in report_types:
        try:
            print(f"Requesting IBKR fundamental data with reportType='{rt}' ...")
            fd = ib.reqFundamentalData(contract, reportType=rt)
            if not fd or not getattr(fd, "reportData", None):
                print(f"No IBKR fundamental data returned for reportType='{rt}'.")
                last_error = f"No data for reportType={rt}"
                continue
            raw_xml = fd.reportData
            return {
                "source": "ibkr",
                "note": (
                    "Raw fundamental XML snapshot stored in 'raw_snapshot'. "
                    f"Fetched using reportType='{rt}'. Parsing can be added later."
                ),
                "report_type_used": rt,
                "raw_snapshot": raw_xml,
            }
        except Exception as e:
            err_msg = f"Error fetching IBKR fundamentals with reportType='{rt}': {e}"
            print(err_msg)
            last_error = err_msg

    return {
        "source": "ibkr",
        "note": (
            "No IBKR fundamental data available for this symbol or fundamentals "
            "subscription missing / paper account. Last error: " + (last_error or "None")
        ),
        "report_type_used": None,
        "raw_snapshot": None,
    }


# ---------- Positions & open orders from IBKR ---------- #

def get_position_info(ib: IB, contract: Contract) -> Optional[Dict[str, Any]]:
    """
    Return your current position for this contract (if any):
    shares, avg cost, account.
    """
    try:
        positions = ib.positions()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return None

    for p in positions:
        if p.contract.conId == contract.conId:
            shares = float(p.position)
            return {
                "account": p.account,
                "shares": int(shares),
                "avg_cost": float(p.avgCost),
            }
    return None


def get_open_orders_for_contract(ib: IB, contract: Contract) -> List[Dict[str, Any]]:
    """
    Return a list of open BUY orders for this contract.

    Requires TWS settings:
      - 'Download open orders on connection' checked
      - and either Master API client ID = IB_CLIENT_ID, or clientId=0
    """
    orders: List[Dict[str, Any]] = []
    try:
        ib.reqAllOpenOrders()
        ib.sleep(1.0)
        trades = ib.openTrades()
    except Exception as e:
        print(f"Error fetching open orders: {e}")
        return orders

    for t in trades:
        c = t.contract
        ord_ = t.order
        st = t.orderStatus

        if c.conId != contract.conId:
            continue
        if ord_.action != "BUY":
            continue
        if st.status not in ("Submitted", "PreSubmitted", "PendingSubmit"):
            continue

        orders.append(
            {
                "action": ord_.action,
                "order_type": ord_.orderType,
                "time_in_force": ord_.tif,
                "total_quantity": float(ord_.totalQuantity),
                "limit_price": float(getattr(ord_, "lmtPrice", 0)) or None,
                "status": st.status,
                "filled": float(st.filled),
                "remaining": float(st.remaining),
            }
        )
    return orders


# ---------- Investor profile & meta ---------- #

def build_investor_profile() -> Dict[str, Any]:
    """
    Returns a copy of the global INVESTOR_PROFILE.
    """
    return dict(INVESTOR_PROFILE)


def build_meta_description() -> Dict[str, Any]:
    return {
        "version": "1.4",
        "description": "Standard data format for stock analysis via ChatGPT.",
        "notes": [
            "All fields are optional except 'symbol'.",
            "Time series arrays must be sorted in ascending date/time order.",
            "Prices are in the currency specified in 'currency'.",
            "intraday_1h_6m = 1-hour candles over approximately the last 6 months.",
            "generated_at is an ISO 8601 timestamp (with seconds) when this JSON was generated.",
            "Fundamentals are fetched primarily from Yahoo Finance via yfinance, "
            "with an additional best-effort raw snapshot from IBKR.",
            "The 'investor_profile' section is filled from the configuration in this script.",
            "If there is any inconsistency between JSON and the text you provide to ChatGPT, "
            "the text values take precedence.",
        ],
        "sections": {
            "symbol": "Ticker of the stock (e.g. NVDA, MSFT).",
            "currency": "Quotation currency (USD, EUR, etc.).",
            "generated_at": "ISO 8601 timestamp (with seconds) when this JSON was generated.",
            "price_daily_5y": "Daily OHLCV history over about 5 years.",
            "intraday_1h_6m": "1-hour OHLCV intraday data over about 6 recent months.",
            "fundamentals": (
                "Key financial indicators, with sub-sections 'yahoo' (rich numeric fields) "
                "and 'ibkr' (raw XML snapshot if available)."
            ),
            "your_position": "Your personal position data (entry, current price, size, horizon).",
            "open_orders": "Your open BUY orders for this symbol (if any).",
            "investor_profile": "Your investor profile (risk, horizon, monthly investment, etc.).",
            "peer_symbols": "Static list of comparable tickers for context.",
        },
    }


def build_json_document(
    symbol: str,
    currency: str,
    daily_5y: List[Dict[str, Any]],
    intraday_6m: List[Dict[str, Any]],
    fundamentals_yahoo: Dict[str, Any],
    fundamentals_ibkr: Dict[str, Any],
    position_info: Optional[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    investor_profile: Dict[str, Any],
    peer_symbols: List[str],
) -> Dict[str, Any]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    last_close = daily_5y[-1]["close"] if daily_5y else None

    if position_info is not None:
        entry_price = position_info.get("avg_cost")
        shares = position_info.get("shares")
    else:
        entry_price = None
        shares = None

    your_position = {
        "entry_price": entry_price,
        "current_price": last_close,
        "shares": shares,
        "horizon": None,
        "notes": (
            "Auto-filled from IBKR. You can override these values in your message "
            "to ChatGPT if something changed or if you use a different reference price."
        ),
    }

    fundamentals = {
        "primary_source": "yahoo",
        "yahoo": fundamentals_yahoo,
        "ibkr": fundamentals_ibkr,
    }

    doc: Dict[str, Any] = {
        "meta_description": build_meta_description(),
        "symbol": symbol,
        "currency": currency,
        "generated_at": generated_at,
        "price_daily_5y": daily_5y,
        "intraday_1h_6m": intraday_6m,
        "fundamentals": fundamentals,
        "your_position": your_position,
        "open_orders": open_orders,
        "investor_profile": investor_profile,
        "peer_symbols": peer_symbols,
    }
    return doc


# ---------- Export logic ---------- #

def export_symbol(ib: IB, symbol: str, exchange: str, currency: str, output_dir: str) -> None:
    print(f"\n=== Processing {symbol} ({exchange}, {currency}) ===")
    contract = qualify_contract(ib, symbol, exchange, currency)

    # 5 years daily
    daily_bars = fetch_historical(
        ib,
        contract,
        duration=DAILY_5Y_DURATION,
        bar_size=DAILY_5Y_BAR_SIZE,
        what_to_show=WHAT_TO_SHOW,
        use_rth=False,
    )
    daily_list = bars_to_daily_list(daily_bars)

    # 6 months 1-hour (robust wrapper)
    try:
        intraday_bars = fetch_historical(
            ib,
            contract,
            duration=INTRADAY_6M_DURATION,
            bar_size=INTRADAY_6M_BAR_SIZE,
            what_to_show=WHAT_TO_SHOW,
            use_rth=False,
        )
        intraday_list = bars_to_intraday_list(intraday_bars)
    except Exception as e:
        print(f"Failed to fetch or parse intraday 1h data for {symbol}: {e}")
        intraday_list = []

    fundamentals_yahoo = fetch_yahoo_fundamentals(symbol)
    fundamentals_ibkr = fetch_ibkr_fundamentals(ib, contract)
    position_info = get_position_info(ib, contract)
    open_orders = get_open_orders_for_contract(ib, contract)
    investor_profile = build_investor_profile()
    peer_symbols = PEERS_BY_SYMBOL.get(symbol, [])

    doc = build_json_document(
        symbol,
        currency,
        daily_list,
        intraday_list,
        fundamentals_yahoo,
        fundamentals_ibkr,
        position_info,
        open_orders,
        investor_profile,
        peer_symbols,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # minute precision in filename
    ensure_output_dir(output_dir)
    filename = os.path.join(output_dir, f"{symbol}_ibkr_data_{timestamp}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"JSON exported to: {filename}")


def main():
    ensure_output_dir(OUTPUT_DIR)
    ib = connect_ib()
    try:
        for symbol, exchange, currency in SYMBOLS:
            try:
                export_symbol(ib, symbol, exchange, currency, OUTPUT_DIR)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
    finally:
        print("Disconnecting from IBKR...")
        ib.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
