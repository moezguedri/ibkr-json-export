#!/usr/bin/env python3
"""
IBKR → One JSON snapshot (portfolio + symbols) for ChatGPT analysis.

Based on user's original script, modified to:
- Generate ONE JSON file containing:
    * portfolio: global positions / weights / largest holding / account values
    * symbols: market data per symbol (daily 5y, intraday 6m, fundamentals, orders)
- Add per-symbol decision-ready derived metrics:
    * last_close
    * high_12m (252 trading days, close-based)
    * drawdown_12m_close_pct
    * ma200 (200-day simple moving average of close)
    * below_ma200 (bool)

Notes:
- Market history remains available (5y daily, 6m intraday) for future strategy refinement.
- Decisions should rely on symbols[*].core_metrics + portfolio.derived.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from ib_insync import IB, Stock, Contract, BarData
import yfinance as yf


# ===================== USER CONFIGURATION ===================== #

# IBKR connection parameters
IB_HOST = "127.0.0.1"
IB_PORT = 4001          # TWS/Gateway port (live or paper)
IB_CLIENT_ID = 1        # Change if you use multiple clients

# Symbols to export: (symbol, exchange, currency)
SYMBOLS: List[Tuple[str, str, str]] = [
    ("NVDA", "SMART", "USD"),
    ("MSFT", "SMART", "USD"),
    ("AMZN", "SMART", "USD"),
    ("META", "SMART", "USD"),
    ("GOOGL", "SMART", "USD"),
    # ("COST", "SMART", "USD"),
    # ("TSM", "SMART", "USD"),
]

# Output directory for JSON file
OUTPUT_DIR = "ibkr_json_output"

# Historical data settings (kept as extras)
DAILY_5Y_DURATION = "5 Y"
DAILY_5Y_BAR_SIZE = "1 day"

INTRADAY_6M_DURATION = "6 M"
INTRADAY_6M_BAR_SIZE = "1 hour"

WHAT_TO_SHOW = "TRADES"   # For stocks this is usually fine

# === INVESTOR PROFILE / RULES CONFIGURATION ==================================
# Keep your rules/profile here. This will live under portfolio.rules and portfolio.investor_profile.
INVESTOR_PROFILE = {
    "risk_tolerance": "medium",           # 'low' | 'medium' | 'high'
    "investment_horizon_years": 5,
    "monthly_investment_usd": 4000.0,     # <-- set to your real monthly DCA
    "preferred_style": "growth",
    "notes": "Investor profile embedded for analysis context."
}

RULES = {
    # Example fields; keep/edit as you wish
    "max_single_position_pct": 0.20,          # cap 20%
    "nvda_forbidden_if_largest": True,        # your lock
    "dca_day_of_month": 15,

    # Opportunistic rule (example placeholders – keep aligned with your finalized rules)
    "opportunistic": {
        "dd_step1_pct": -15,                  # drawdown <= -15% => step1
        "dd_step2_pct": -20,                  # drawdown <= -20% => step2
        "ma200_filter": {
            "MSFT": 1.00,
            "GOOGL": 1.02,
            "AMZN": 1.03,
            "META": 1.00,
            "NVDA": 1.00
        }
    }
}

# === PEER GROUPS (informational) =============================================
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
        "market_cap": get_float("marketCap"),
        "pe": get_float("trailingPE"),
        "forward_pe": get_float("forwardPE"),
        "peg": get_float("pegRatio"),
        "ps": get_float("priceToSalesTrailing12Months"),
        "pb": get_float("priceToBook"),
        "dividend_yield": get_float("dividendYield"),
        "earnings_timestamp": get_float("earningsTimestamp"),
        "earnings_timestamp_start": get_float("earningsTimestampStart"),
        "earnings_timestamp_end": get_float("earningsTimestampEnd"),
        "earnings_quarterly_growth": get_float("earningsQuarterlyGrowth"),
        "revenue_growth": get_float("revenueGrowth"),
        "revenue_ttm": get_float("totalRevenue"),
        "net_income_ttm": get_float("netIncomeToCommon"),
        "gross_margin": get_float("grossMargins"),
        "operating_margin": get_float("operatingMargins"),
        "profit_margin": get_float("profitMargins"),
        "roe": get_float("returnOnEquity"),
        "debt_to_equity": get_float("debtToEquity"),
        "current_ratio": get_float("currentRatio"),
        "quick_ratio": get_float("quickRatio"),
        "free_cashflow": get_float("freeCashflow"),
        "operating_cashflow": get_float("operatingCashflow"),
        "analyst_recommendation": info.get("recommendationKey"),
        "target_mean_price": get_float("targetMeanPrice"),
        "target_high_price": get_float("targetHighPrice"),
        "target_low_price": get_float("targetLowPrice"),
        "number_of_analyst_opinions": get_float("numberOfAnalystOpinions"),
        "short_percent_of_float": get_float("shortPercentOfFloat"),
        "shares_short": get_float("sharesShort"),
        "shares_short_prior_month": get_float("sharesShortPriorMonth"),
        "implied_volatility": get_float("impliedVolatility"),
        "held_percent_insiders": get_float("heldPercentInsiders"),
        "held_percent_institutions": get_float("heldPercentInstitutions"),
    }
    return fundamentals


# ---------- IBKR fundamentals (best-effort placeholder retained) ---------- #

def fetch_ibkr_fundamentals(ib: IB, contract: Contract) -> Dict[str, Any]:
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


# ---------- Derived metrics (decision core) ---------- #

def compute_core_metrics(daily_prices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decision-ready metrics derived from daily close:
    - last_close
    - high_12m (252 trading days, close-based)
    - drawdown_12m_close_pct = (last_close / high_12m - 1) * 100
    - ma200 (200-day simple moving average of close)
    - below_ma200 (bool)
    """
    closes = [float(d["close"]) for d in daily_prices if d.get("close") is not None]
    if not closes:
        return {"note": "No daily close data available."}

    last_close = closes[-1]

    # 12m high approximated as last 252 trading sessions
    window_12m = closes[-252:] if len(closes) >= 252 else closes
    high_12m = max(window_12m) if window_12m else None

    drawdown_12m_close_pct = None
    if high_12m and high_12m > 0:
        drawdown_12m_close_pct = (last_close / high_12m - 1.0) * 100.0

    ma200 = None
    if len(closes) >= 200:
        ma200 = sum(closes[-200:]) / 200.0

    below_ma200 = None
    if ma200 is not None:
        below_ma200 = last_close < ma200

    out = {
        "last_close": round(last_close, 6),
        "high_12m": round(high_12m, 6) if high_12m is not None else None,
        "drawdown_12m_close_pct": round(drawdown_12m_close_pct, 2) if drawdown_12m_close_pct is not None else None,
        "ma200": round(ma200, 6) if ma200 is not None else None,
        "below_ma200": below_ma200,
        "convention": {
            "high_12m_basis": "close",
            "high_12m_window_sessions": 252,
            "ma200_basis": "close",
            "ma200_window_sessions": 200
        }
    }
    return out


# ---------- Positions & open orders from IBKR ---------- #

def get_position_info(ib: IB, contract: Contract) -> Optional[Dict[str, Any]]:
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
                "shares": float(shares),
                "avg_cost": float(p.avgCost),
            }
    return None


def get_open_orders_for_contract(ib: IB, contract: Contract) -> List[Dict[str, Any]]:
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


# ---------- Portfolio snapshot (global) ---------- #

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fetch_account_summary(ib: IB) -> Dict[str, Any]:
    """
    Returns a dict keyed by tag (NetLiquidation, TotalCashValue, etc.).
    If multiple accounts exist, we keep the first account's values + include the full raw list.
    """
    try:
        summary = ib.accountSummary()
    except Exception as e:
        return {"note": f"Failed to fetch account summary: {e}", "raw": []}

    raw = []
    for s in summary:
        raw.append({
            "account": s.account,
            "tag": s.tag,
            "value": s.value,
            "currency": s.currency
        })

    # Pick a primary account (first encountered)
    primary_account = raw[0]["account"] if raw else None

    # Build a tag->value map for the primary account (base currency lines often currency='BASE')
    primary_map: Dict[str, Any] = {}
    if primary_account:
        for r in raw:
            if r["account"] == primary_account:
                key = f'{r["tag"]}:{r["currency"]}'
                primary_map[key] = r["value"]

    def get_tag(tag: str) -> Optional[float]:
        # prefer BASE
        v = primary_map.get(f"{tag}:BASE")
        if v is None:
            # fallback any currency
            for k, vv in primary_map.items():
                if k.startswith(tag + ":"):
                    v = vv
                    break
        return _safe_float(v)

    return {
        "primary_account": primary_account,
        "net_liquidation": get_tag("NetLiquidation"),
        "total_cash_value": get_tag("TotalCashValue"),
        "gross_position_value": get_tag("GrossPositionValue"),
        "available_funds": get_tag("AvailableFunds"),
        "excess_liquidity": get_tag("ExcessLiquidity"),
        "raw": raw
    }


def fetch_portfolio_positions(ib: IB) -> List[Dict[str, Any]]:
    """
    Uses ib.portfolio() which returns portfolio items (market value, position, etc.).
    """
    try:
        items = ib.portfolio()
    except Exception as e:
        print(f"Failed to fetch ib.portfolio(): {e}")
        return []

    out: List[Dict[str, Any]] = []
    for it in items:
        c = it.contract
        # Keep mostly stocks; IBKR may return other asset classes as well.
        out.append({
            "account": it.account,
            "symbol": getattr(c, "symbol", None),
            "secType": getattr(c, "secType", None),
            "currency": getattr(c, "currency", None),
            "conId": getattr(c, "conId", None),
            "position": _safe_float(it.position),
            "marketPrice": _safe_float(it.marketPrice),
            "marketValue": _safe_float(it.marketValue),
            "averageCost": _safe_float(it.averageCost),
            "unrealizedPNL": _safe_float(it.unrealizedPNL),
            "realizedPNL": _safe_float(it.realizedPNL),
        })
    return out


def build_portfolio_section(
    ib: IB,
    tracked_symbols: List[str],
    rules: Dict[str, Any],
    investor_profile: Dict[str, Any]
) -> Dict[str, Any]:
    acct = fetch_account_summary(ib)
    positions_all = fetch_portfolio_positions(ib)

    # Filter stock positions with symbol present
    stock_positions = [p for p in positions_all if p.get("secType") == "STK" and p.get("symbol")]

    # Largest holding across ALL stock positions (by absolute marketValue)
    largest_holding = None
    largest_value = None
    for p in stock_positions:
        mv = p.get("marketValue")
        if mv is None:
            continue
        if largest_value is None or abs(mv) > abs(largest_value):
            largest_value = mv
            largest_holding = p.get("symbol")

    # Build tracked positions dict (only symbols in SYMBOLS list)
    tracked_positions: Dict[str, Any] = {}
    for p in stock_positions:
        sym = p["symbol"]
        if sym in tracked_symbols:
            tracked_positions[sym] = {
                "shares": p.get("position"),
                "market_value": p.get("marketValue"),
                "avg_cost": p.get("averageCost"),
                "currency": p.get("currency"),
                "account": p.get("account"),
            }

    # Compute weights: prefer NetLiquidation if available; otherwise sum of tracked market values
    denom = acct.get("net_liquidation")
    if denom is None or denom == 0:
        denom = sum(v.get("market_value") or 0 for v in tracked_positions.values())
        denom = denom if denom != 0 else None

    weights_pct: Dict[str, Optional[float]] = {}
    if denom:
        for sym, v in tracked_positions.items():
            mv = v.get("market_value")
            if mv is None:
                weights_pct[sym] = None
            else:
                weights_pct[sym] = round((mv / denom) * 100.0, 4)

    return {
        "account_summary": acct,
        "positions_all": positions_all,  # full raw list (can be large; keep for audit)
        "tracked_positions": tracked_positions,
        "derived": {
            "tracked_weights_pct": weights_pct,
            "largest_holding_stock_symbol": largest_holding,
        },
        "rules": dict(rules),
        "investor_profile": dict(investor_profile),
    }


# ---------- Meta ---------- #

def build_meta() -> Dict[str, Any]:
    return {
        "version": "2.0",
        "description": "One-shot portfolio + symbols snapshot for DCA/opportunistic decision analysis.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "symbols[*].core_metrics contains decision-ready metrics derived from daily closes.",
            "symbols[*].price_daily_5y and symbols[*].intraday_1h_6m are optional extras for future strategy refinement.",
            "portfolio.derived.tracked_weights_pct uses accountSummary.net_liquidation when available; otherwise falls back to tracked market values sum.",
        ],
    }


# ---------- Symbol export (returns dict instead of writing file) ---------- #

def build_symbol_block(ib: IB, symbol: str, exchange: str, currency: str) -> Dict[str, Any]:
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

    # 6 months 1-hour
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
    peer_symbols = PEERS_BY_SYMBOL.get(symbol, [])

    core_metrics = compute_core_metrics(daily_list)

    # Last close for convenience
    last_close = daily_list[-1]["close"] if daily_list else None

    symbol_block: Dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "currency": currency,
        "last_close": last_close,
        "core_metrics": core_metrics,
        "price_daily_5y": daily_list,
        "intraday_1h_6m": intraday_list,
        "fundamentals": {
            "primary_source": "yahoo",
            "yahoo": fundamentals_yahoo,
            "ibkr": fundamentals_ibkr,
        },
        "your_position": {
            "entry_price": position_info.get("avg_cost") if position_info else None,
            "shares": position_info.get("shares") if position_info else None,
            "account": position_info.get("account") if position_info else None,
        },
        "open_orders": open_orders,
        "peer_symbols": peer_symbols,
    }
    return symbol_block


def main():
    ensure_output_dir(OUTPUT_DIR)

    tracked_symbols = [s for (s, _, _) in SYMBOLS]
    meta = build_meta()

    ib = connect_ib()
    try:
        symbols_data: Dict[str, Any] = {}
        for symbol, exchange, currency in SYMBOLS:
            try:
                symbols_data[symbol] = build_symbol_block(ib, symbol, exchange, currency)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                symbols_data[symbol] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "currency": currency,
                    "error": str(e),
                }

        portfolio_section = build_portfolio_section(
            ib=ib,
            tracked_symbols=tracked_symbols,
            rules=RULES,
            investor_profile=INVESTOR_PROFILE,
        )

        final_doc: Dict[str, Any] = {
            "asof": datetime.now().isoformat(timespec="seconds"),
            "meta": meta,
            "portfolio": portfolio_section,
            "symbols": symbols_data,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(OUTPUT_DIR, f"portfolio_ibkr_snapshot_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2)

        print(f"\n✅ Snapshot exported to: {filename}")

    finally:
        print("Disconnecting from IBKR...")
        ib.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
