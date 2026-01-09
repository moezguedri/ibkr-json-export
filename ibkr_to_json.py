#!/usr/bin/env python3
"""
IBKR → One JSON snapshot (portfolio + symbols), redesigned.

FULL mode (default):
- Daily history (5y by default) via yfinance
- Intraday history: 1h bars over ~6 months (intraday_1h_6m) via yfinance
- Optional fundamentals (yfinance + IBKR)
- Optional open orders

LIGHT mode (--light):
- Minimal snapshot for the monthly decision engine (fast & small)
- Daily history shortened (last ~260 bars)
- NO intraday, NO fundamentals, NO open orders

Stable snapshot contract for downstream consumers:
- portfolio.derived.tracked_weights_pct
- portfolio.derived.largest_holding_stock_symbol
- symbols[SYM].core_metrics:
    last_close
    high_12m_close
    drawdown_12m_close_pct
    ma200_close
    price_vs_ma200_pct
    below_ma200

Machine-friendly output:
- Prints: SNAPSHOT_PATH=<path>

Where to change things:
- Defaults are in the USER CONFIGURATION section below.
- You can override most defaults via CLI flags.

Requirements:
- Python 3.10+
- ib_insync
- yfinance
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf
from ib_insync import IB, Stock


# ===================== USER CONFIGURATION ===================== #
DEFAULT_TRACKED_SYMBOLS = ["NVDA", "MSFT", "AMZN", "META", "GOOGL"]
DEFAULT_IGNORED_SYMBOLS = ["COST"]

# IBKR connection defaults (TWS / IB Gateway)
IB_HOST = "127.0.0.1"
IB_PORT = 4001
IB_CLIENT_ID = 7

DEFAULT_OUTPUT_DIR = "snapshots"

# History windows
FULL_DAILY_YEARS = 5
LIGHT_DAILY_YEARS = 2
LIGHT_DAILY_MAX_ROWS = 260

# Intraday (FULL mode)
INTRADAY_INTERVAL = "1h"
INTRADAY_PERIOD = "6mo"  # about 6 months
INTRADAY_MAX_ROWS = None  # keep all returned rows

# Core metrics windows (close-based)
WINDOW_HIGH_12M_TRADING_DAYS = 252
WINDOW_MA200_DAYS = 200

# FULL-mode enrichments
FETCH_OPEN_ORDERS_IN_FULL = True
FETCH_YFINANCE_FUNDAMENTALS_IN_FULL = True
FETCH_IBKR_FUNDAMENTALS_IN_FULL = True  # can be huge (XML)

# Performance
YFINANCE_THREADS = True
# ============================================================= #


@dataclass
class ModeConfig:
    mode: str  # "full" or "light"
    daily_years: int
    daily_max_rows: Optional[int]
    include_intraday: bool
    include_fundamentals: bool
    include_open_orders: bool


def parse_csv_symbols(csv: str) -> List[str]:
    return [s.strip().upper() for s in csv.split(",") if s.strip()]


def iso_now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    """
    Convert scalars to float safely.

    Fixes pandas FutureWarning: float(Series([single])) is deprecated.
    - If x looks like a pandas Series (has .iloc), use x.iloc[0].
    - If x is a numpy scalar, float(x) works.
    """
    try:
        if x is None:
            return None

        # pandas Series or similar
        if hasattr(x, "iloc") and not isinstance(x, (str, bytes)):
            try:
                # Handle empty series
                if hasattr(x, "empty") and x.empty:
                    return None
                return float(x.iloc[0])
            except Exception:
                pass

        return float(x)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a single IBKR snapshot JSON (portfolio + symbols). FULL by default."
    )
    p.add_argument("--light", action="store_true", help="Generate a LIGHT snapshot (faster, smaller).")
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write snapshots (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_TRACKED_SYMBOLS),
        help="Comma-separated tracked symbols (default from script).",
    )
    p.add_argument(
        "--ignore",
        default=",".join(DEFAULT_IGNORED_SYMBOLS),
        help="Comma-separated ignored symbols (default from script).",
    )
    p.add_argument("--host", default=IB_HOST, help="IBKR API host (default from script).")
    p.add_argument("--port", type=int, default=IB_PORT, help="IBKR API port (default from script).")
    p.add_argument("--client-id", type=int, default=IB_CLIENT_ID, help="IBKR API clientId (default from script).")
    p.add_argument(
        "--no-ibkr-fundamentals",
        action="store_true",
        help="In FULL mode, do not request IBKR fundamentals (reduces file size).",
    )
    p.add_argument(
        "--no-yf-fundamentals",
        action="store_true",
        help="In FULL mode, do not fetch yfinance fundamentals (reduces runtime).",
    )
    p.add_argument(
        "--no-open-orders",
        action="store_true",
        help="In FULL mode, do not include open orders (reduces runtime).",
    )
    p.add_argument(
        "--no-intraday",
        action="store_true",
        help="In FULL mode, do not include intraday_1h_6m history (reduces runtime).",
    )
    return p.parse_args()


def build_mode_config(args: argparse.Namespace) -> ModeConfig:
    if args.light:
        return ModeConfig(
            mode="light",
            daily_years=LIGHT_DAILY_YEARS,
            daily_max_rows=LIGHT_DAILY_MAX_ROWS,
            include_intraday=False,
            include_fundamentals=False,
            include_open_orders=False,
        )
    include_fundamentals = not (args.no_ibkr_fundamentals and args.no_yf_fundamentals)
    include_open_orders = FETCH_OPEN_ORDERS_IN_FULL and (not args.no_open_orders)
    include_intraday = (not args.no_intraday)
    return ModeConfig(
        mode="full",
        daily_years=FULL_DAILY_YEARS,
        daily_max_rows=None,
        include_intraday=include_intraday,
        include_fundamentals=include_fundamentals,
        include_open_orders=include_open_orders,
    )


def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def fetch_portfolio(ib: IB) -> Dict[str, Any]:
    positions = ib.portfolio()
    positions_all: List[Dict[str, Any]] = []
    for p in positions:
        c = p.contract
        positions_all.append(
            {
                "symbol": getattr(c, "symbol", None),
                "secType": getattr(c, "secType", None),
                "currency": getattr(c, "currency", None),
                "exchange": getattr(c, "exchange", None),
                "position": safe_float(p.position),
                "marketPrice": safe_float(p.marketPrice),
                "marketValue": safe_float(p.marketValue),
                "averageCost": safe_float(p.averageCost),
                "unrealizedPNL": safe_float(p.unrealizedPNL),
                "realizedPNL": safe_float(p.realizedPNL),
            }
        )

    account_summary_items = ib.accountSummary()
    account_summary: Dict[str, Any] = {}
    for item in account_summary_items:
        key = f"{item.tag}_{item.currency}" if item.currency else item.tag
        account_summary[key] = item.value

    return {"account_summary": account_summary, "positions_all": positions_all}


def derive_tracked(portfolio: Dict[str, Any], tracked_symbols: List[str]) -> Dict[str, Any]:
    tracked_set = set(tracked_symbols)
    positions_all = portfolio.get("positions_all", [])

    tracked_positions = [p for p in positions_all if (p.get("symbol") or "").upper() in tracked_set]
    tracked_market_values = [safe_float(p.get("marketValue")) for p in tracked_positions]
    tracked_market_values = [v for v in tracked_market_values if v is not None]
    total_tracked_mv = sum(tracked_market_values) if tracked_market_values else 0.0

    tracked_weights_pct: Dict[str, float] = {}
    largest_symbol: Optional[str] = None
    largest_mv = -1.0

    for p in tracked_positions:
        sym = (p.get("symbol") or "").upper()
        mv = safe_float(p.get("marketValue")) or 0.0
        tracked_weights_pct[sym] = (mv / total_tracked_mv * 100.0) if total_tracked_mv > 0 else 0.0
        if mv > largest_mv:
            largest_mv = mv
            largest_symbol = sym

    ignored_positions_detected = sorted(
        {
            (p.get("symbol") or "").upper()
            for p in positions_all
            if (p.get("symbol") is not None) and ((p.get("symbol") or "").upper() not in tracked_set)
        }
    )

    return {
        "positions_tracked": tracked_positions,
        "derived": {
            "tracked_total_market_value": safe_float(total_tracked_mv),
            "tracked_weights_pct": tracked_weights_pct,
            "largest_holding_stock_symbol": largest_symbol,
            "ignored_positions_detected": ignored_positions_detected,
        },
    }


def fetch_open_orders(ib: IB) -> List[Dict[str, Any]]:
    orders = ib.openOrders()
    trades = ib.openTrades()
    trade_by_order_id = {t.order.orderId: t for t in trades if hasattr(t, "order")}

    out: List[Dict[str, Any]] = []
    for o in orders:
        t = trade_by_order_id.get(o.orderId)
        contract = getattr(t, "contract", None) if t else None
        out.append(
            {
                "orderId": o.orderId,
                "clientId": o.clientId,
                "action": o.action,
                "totalQuantity": safe_float(o.totalQuantity),
                "orderType": o.orderType,
                "lmtPrice": safe_float(getattr(o, "lmtPrice", None)),
                "auxPrice": safe_float(getattr(o, "auxPrice", None)),
                "tif": getattr(o, "tif", None),
                "symbol": getattr(contract, "symbol", None) if contract else None,
                "secType": getattr(contract, "secType", None) if contract else None,
                "currency": getattr(contract, "currency", None) if contract else None,
                "exchange": getattr(contract, "exchange", None) if contract else None,
                "status": getattr(t, "orderStatus", None).status if t and getattr(t, "orderStatus", None) else None,
            }
        )
    return out


def yf_download_daily(symbols: List[str], years: int) -> Dict[str, Any]:
    period = f"{years}y"
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=YFINANCE_THREADS,
        progress=False,
    )

    out: Dict[str, Any] = {}
    if len(symbols) == 1:
        out[symbols[0]] = data
        return out

    for sym in symbols:
        try:
            out[sym] = data[sym].dropna(how="all")
        except Exception:
            out[sym] = None
    return out


def yf_download_intraday_1h(symbol: str) -> Any:
    """
    yfinance intraday (1h). period=6mo usually supported.
    """
    try:
        df = yf.download(
            tickers=symbol,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return df.dropna(how="all") if df is not None else None
    except Exception:
        return None


def get_row_value(row: Any, field: str) -> Any:
    """
    Robustly fetch a value from a pandas Series row.
    Handles both:
      - flat columns: "Open", "High", ...
      - MultiIndex columns: ("Open", "MSFT") etc., where row["Open"] returns a sub-Series.
    """
    try:
        v = row.get(field)
        # If v is a Series (single element), safe_float will handle it.
        return v
    except Exception:
        return None


def df_to_records(df: Any, max_rows: Optional[int], is_intraday: bool) -> List[Dict[str, Any]]:
    if df is None:
        return []
    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        key = "datetime" if is_intraday else "date"
        ts_str = idx.strftime("%Y-%m-%d %H:%M:%S") if is_intraday else idx.strftime("%Y-%m-%d")
        records.append(
            {
                key: ts_str,
                "open": safe_float(get_row_value(row, "Open")),
                "high": safe_float(get_row_value(row, "High")),
                "low": safe_float(get_row_value(row, "Low")),
                "close": safe_float(get_row_value(row, "Close")),
                "volume": safe_float(get_row_value(row, "Volume")),
            }
        )
    return records


def compute_core_metrics(daily_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    closes = [r["close"] for r in daily_records if r.get("close") is not None]
    if not closes:
        return {
            "last_close": None,
            "high_12m_close": None,
            "drawdown_12m_close_pct": None,
            "ma200_close": None,
            "price_vs_ma200_pct": None,
            "below_ma200": None,
        }

    last_close = closes[-1]

    high_12m_close = None
    drawdown_12m_close_pct = None
    if len(closes) >= WINDOW_HIGH_12M_TRADING_DAYS:
        last_252 = closes[-WINDOW_HIGH_12M_TRADING_DAYS :]
        high_12m_close = max(last_252) if last_252 else None
        if high_12m_close and high_12m_close != 0:
            drawdown_12m_close_pct = (last_close / high_12m_close - 1.0) * 100.0

    ma200_close = None
    if len(closes) >= WINDOW_MA200_DAYS:
        last_200 = closes[-WINDOW_MA200_DAYS :]
        ma200_close = sum(last_200) / len(last_200) if last_200 else None

    below_ma200 = None
    price_vs_ma200_pct = None
    if ma200_close and ma200_close != 0:
        below_ma200 = last_close < ma200_close
        price_vs_ma200_pct = (last_close / ma200_close - 1.0) * 100.0

    return {
        "last_close": safe_float(last_close),
        "high_12m_close": safe_float(high_12m_close),
        "drawdown_12m_close_pct": safe_float(drawdown_12m_close_pct),
        "ma200_close": safe_float(ma200_close),
        "price_vs_ma200_pct": safe_float(price_vs_ma200_pct),
        "below_ma200": below_ma200,
    }


def fetch_yf_fundamentals(symbol: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        keys = [
            "shortName",
            "longName",
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "beta",
            "dividendYield",
            "earningsQuarterlyGrowth",
            "revenueGrowth",
            "profitMargins",
            "grossMargins",
            "operatingMargins",
            "freeCashflow",
            "totalCash",
            "totalDebt",
            "currency",
        ]
        curated = {k: info.get(k) for k in keys if k in info}
        return {"curated_info": curated}
    except Exception as e:
        return {"error": str(e)}


def fetch_ibkr_fundamentals(ib: IB, symbol: str) -> Dict[str, Any]:
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)
    out: Dict[str, Any] = {}
    try:
        xml = ib.reqFundamentalData(contract, reportType="ReportSnapshot")
        out["report_type"] = "ReportSnapshot"
        out["raw_snapshot_xml"] = xml
    except Exception as e:
        out["error"] = str(e)
    return out


def build_meta(mode_cfg: ModeConfig, tracked: List[str], ignored: List[str], portfolio: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at_utc": iso_now_utc(),
        "mode": mode_cfg.mode,
        "tracked_symbols": tracked,
        "ignored_symbols_config": ignored,
        "portfolio_ignored_positions_detected": portfolio.get("derived", {}).get("ignored_positions_detected", []),
        "convention": {
            "price_basis": "close",
            "high_12m_window_trading_days": WINDOW_HIGH_12M_TRADING_DAYS,
            "ma_name": "SMA",
            "ma_period_days": WINDOW_MA200_DAYS,
        },
        "history_windows": {
            "daily_years": mode_cfg.daily_years,
            "daily_max_rows": mode_cfg.daily_max_rows,
            "intraday": {
                "enabled": mode_cfg.include_intraday,
                "interval": INTRADAY_INTERVAL if mode_cfg.include_intraday else None,
                "period": INTRADAY_PERIOD if mode_cfg.include_intraday else None,
            },
        },
    }


def main() -> int:
    args = parse_args()
    mode_cfg = build_mode_config(args)

    tracked_symbols = parse_csv_symbols(args.symbols)
    ignored_symbols = parse_csv_symbols(args.ignore)

    ensure_dir(args.output_dir)

    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        portfolio_base = fetch_portfolio(ib)
        portfolio_derived = derive_tracked(portfolio_base, tracked_symbols)
        portfolio = {**portfolio_base, **portfolio_derived}

        open_orders: List[Dict[str, Any]] = []
        if mode_cfg.include_open_orders:
            try:
                open_orders = fetch_open_orders(ib)
            except Exception:
                open_orders = []

        daily_dfs = yf_download_daily(tracked_symbols, years=mode_cfg.daily_years)

        symbols_doc: Dict[str, Any] = {}
        for symbol in tracked_symbols:
            df_daily = daily_dfs.get(symbol)
            daily_records = df_to_records(df_daily, max_rows=mode_cfg.daily_max_rows, is_intraday=False)
            core_metrics = compute_core_metrics(daily_records)

            sym_doc: Dict[str, Any] = {
                "core_metrics": core_metrics,
                "history": {"daily": daily_records},
            }

            if mode_cfg.include_intraday:
                df_intra = yf_download_intraday_1h(symbol)
                intraday_records = df_to_records(df_intra, max_rows=INTRADAY_MAX_ROWS, is_intraday=True)
                sym_doc["history"]["intraday_1h_6m"] = intraday_records

            if mode_cfg.include_fundamentals:
                fundamentals: Dict[str, Any] = {}
                if FETCH_YFINANCE_FUNDAMENTALS_IN_FULL and not args.no_yf_fundamentals:
                    fundamentals["yfinance"] = fetch_yf_fundamentals(symbol)
                if FETCH_IBKR_FUNDAMENTALS_IN_FULL and not args.no_ibkr_fundamentals:
                    fundamentals["ibkr"] = fetch_ibkr_fundamentals(ib, symbol)
                sym_doc["fundamentals"] = fundamentals

            if open_orders:
                sym_doc["open_orders"] = [o for o in open_orders if (o.get("symbol") or "").upper() == symbol]

            symbols_doc[symbol] = sym_doc

        meta = build_meta(mode_cfg, tracked_symbols, ignored_symbols, portfolio)

        snapshot: Dict[str, Any] = {"meta": meta, "portfolio": portfolio, "symbols": symbols_doc}

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"portfolio_snapshot_{ts}.json"
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        print(f"SNAPSHOT_PATH={path}")
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
