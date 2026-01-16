#!/usr/bin/env python3
"""
IBKR → One JSON snapshot (portfolio + symbols), redesigned.

Debug-enhanced self-contained version:
- Does NOT read config.json
- Adds structured logging + per-symbol diagnostics to find failures quickly
- Always logs the *resolved yfinance ticker* used for each IBKR symbol
- Captures and stores exceptions (with short traceback) under snapshot['diagnostics']

FULL mode (default):
- Daily history (5y by default) via yfinance
- Intraday history: 1h bars over ~6 months (intraday_1h_6m) via yfinance
- Optional fundamentals (yfinance + IBKR)
- Optional open orders

LIGHT mode (--light):
- Minimal snapshot for the monthly decision engine (fast & small)
- Daily history shortened (last ~260 bars)
- NO intraday, NO fundamentals, NO open orders

Machine-friendly output:
- Prints: SNAPSHOT_PATH=<path>

Requirements:
- Python 3.10+
- ib_insync
- yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path

import yfinance as yf
from ib_insync import IB
import asyncio
import socket

# ===================== USER CONFIGURATION ===================== #

# Mapping: IBKR symbol -> Yahoo Finance ticker used by yfinance
# (This is the fix for your IB1T 404 problem.)
YF_TICKER_OVERRIDES: Dict[str, str] = {
    # iShares Bitcoin ETP on SIX (EBS) in CHF uses .SW on Yahoo
    "IB1T": "IB1T.SW",
}

# Mapping: IBKR symbol -> contract overrides for IBKR fundamental queries
# (Only needed for instruments where USD/SMART defaults are wrong.)
IBKR_CONTRACT_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Same instrument you showed:
    # exchange SMART, currency CHF, primaryExchange EBS
    "IB1T": {"exchange": "SMART", "currency": "CHF", "primaryExchange": "EBS"},
}

# Defaults for "normal" US stocks
IBKR_DEFAULT_EXCHANGE = "SMART"
IBKR_DEFAULT_CURRENCY = "USD"

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

# yfinance retries
YF_MAX_RETRIES = 3
YF_RETRY_BACKOFF_SEC = 1.5

# ============================================================= #

def load_symbols_from_config() -> list:
    config_path = Path("config.json").expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    # symbols
    symbols_raw = raw.get("symbols", {})
    return symbols_raw.get("tracked")

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
    """Convert scalars to float safely (handles pandas Series singletons)."""
    try:
        if x is None:
            return None

        # pandas Series or similar
        if hasattr(x, "iloc") and not isinstance(x, (str, bytes)):
            try:
                if hasattr(x, "empty") and x.empty:
                    return None
                return float(x.iloc[0])
            except Exception:
                pass

        return float(x)
    except Exception:
        return None


def resolve_yf_ticker(symbol: str) -> str:
    """Return the yfinance ticker for a tracked IBKR symbol."""
    s = symbol.upper()
    return YF_TICKER_OVERRIDES.get(s, s)


def resolve_ibkr_contract(symbol: str) -> Tuple[str, str, Optional[str]]:
    """Return (exchange, currency, primaryExchange) for IBKR contract resolution."""
    s = symbol.upper()
    o = IBKR_CONTRACT_OVERRIDES.get(s, {})
    exchange = o.get("exchange", IBKR_DEFAULT_EXCHANGE)
    currency = o.get("currency", IBKR_DEFAULT_CURRENCY)
    primary = o.get("primaryExchange")
    return exchange, currency, primary


def setup_logging(debug: bool) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger("ibkr_to_json")


def short_tb(exc: BaseException, max_lines: int = 10) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    lines = tb.strip().splitlines()
    if len(lines) <= max_lines:
        return tb
    return "\n".join(lines[-max_lines:])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a single IBKR snapshot JSON (portfolio + symbols). FULL by default."
    )
    p.add_argument("--light", action="store_true", help="Generate a LIGHT snapshot (faster, smaller).")
    p.add_argument("--debug", action="store_true", help="Verbose logging + extra diagnostics.")
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write snapshots (default: {DEFAULT_OUTPUT_DIR})",
    )
    # Symbols you want to track in the snapshot (IBKR symbols, not Yahoo symbols)
    tracked_symbols = load_symbols_from_config()
    p.add_argument(
        "--symbols",
        default=",".join(tracked_symbols),
        help="Comma-separated tracked symbols (IBKR symbols) (default from script).",
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


class IBKRConnectionError(RuntimeError):
    pass

def is_ibkr_exception(e: Exception) -> bool:
    """
    Version-proof IBKR exception detection.
    We treat as IBKR-related:
      - connection/transport errors
      - any exception whose class/module comes from ib_insync
      - any exception that looks like IBKR API error (has errorCode/reqId etc.)
    """
    if isinstance(e, (ConnectionRefusedError, TimeoutError, asyncio.TimeoutError, socket.error, OSError)):
        return True

    mod = type(e).__module__ or ""
    name = type(e).__name__ or ""
    if mod.startswith("ib_insync") or "ib_insync" in mod:
        return True

    # common ib_insync/IBKR error shapes
    if hasattr(e, "errorCode") or hasattr(e, "code") or hasattr(e, "reqId"):
        return True

    # last resort: class name used in some versions
    if name in {"RequestError", "IBError"}:
        return True

    return False

def connect_ib(host: str, port: int, client_id: int, log) -> IB:
    try:
        ib = IB()
        ib.connect(host, port, clientId=client_id)
        return ib

    except Exception as e:
        # We assume this function is ONLY used for IBKR connect,
        # so any exception here is an IBKR failure by definition.
        log.error(
            "IBKR connection failed (host=%s port=%s clientId=%s): %s",
            host, port, client_id, e
        )
        raise IBKRConnectionError("IBKR connection failed") from e


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
    tracked_set = set(s.upper() for s in tracked_symbols)
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

    return {
        "positions_tracked": tracked_positions,
        "derived": {
            "tracked_total_market_value": safe_float(total_tracked_mv),
            "tracked_weights_pct": tracked_weights_pct,
            "largest_holding_stock_symbol": largest_symbol,
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


def yf_download_with_retries(log: logging.Logger, *, tickers: str, **kwargs) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            return yf.download(tickers=tickers, **kwargs)
        except Exception as e:
            last_exc = e
            log.warning(f"yfinance.download failed (attempt {attempt}/{YF_MAX_RETRIES}) tickers={tickers!r}: {e}")
            if attempt < YF_MAX_RETRIES:
                time.sleep(YF_RETRY_BACKOFF_SEC * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("yfinance.download failed with unknown error")


def yf_download_daily(
    log: logging.Logger,
    tracked_symbols: List[str],
    years: int,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Download daily OHLCV for all tracked symbols using their yfinance tickers."""
    period = f"{years}y"

    # Map tracked -> yf ticker
    t2y = {sym: resolve_yf_ticker(sym) for sym in tracked_symbols}
    yf_tickers = list(dict.fromkeys(t2y.values()))  # stable unique

    log.info(f"Daily history: period={period}, symbols={tracked_symbols}")
    log.info(f"Daily history: yfinance tickers={yf_tickers} (resolved)")

    # Guard: if a known override exists but the raw symbol is passed, log it.
    for sym in tracked_symbols:
        yf_sym = t2y[sym]
        if sym == "IB1T" and yf_sym == "IB1T":
            log.error("BUG: IB1T was NOT mapped to IB1T.SW. Override missing or bypassed.")
            diagnostics.setdefault("global_errors", []).append(
                {"where": "yf_download_daily", "symbol": sym, "error": "IB1T not mapped to IB1T.SW"}
            )

    out: Dict[str, Any] = {}
    try:
        data = yf_download_with_retries(
            log,
            tickers=" ".join(yf_tickers),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=YFINANCE_THREADS,
            progress=False,
        )
    except Exception as e:
        diagnostics.setdefault("global_errors", []).append(
            {"where": "yf_download_daily", "error": str(e), "trace": short_tb(e)}
        )
        # Return None for each symbol so the snapshot can still be produced.
        for sym in tracked_symbols:
            out[sym] = None
        return out

    if len(yf_tickers) == 1:
        only_yf = yf_tickers[0]
        for tracked, yf_sym in t2y.items():
            out[tracked] = data if yf_sym == only_yf else None
        return out

    # Multiple tickers: grouped by yf symbol (top-level columns)
    for tracked, yf_sym in t2y.items():
        try:
            out[tracked] = data[yf_sym].dropna(how="all")
        except Exception as e:
            diagnostics.setdefault("symbol_errors", {}).setdefault(tracked, []).append(
                {
                    "where": "yf_download_daily.select_group",
                    "tracked_symbol": tracked,
                    "yfinance_ticker": yf_sym,
                    "error": str(e),
                }
            )
            out[tracked] = None
    return out


def yf_download_intraday_1h(
    log: logging.Logger,
    tracked_symbol: str,
    diagnostics: Dict[str, Any],
) -> Any:
    """yfinance intraday (1h). period=6mo usually supported."""
    yf_symbol = resolve_yf_ticker(tracked_symbol)
    log.debug(f"Intraday 1h: {tracked_symbol} -> {yf_symbol} period={INTRADAY_PERIOD}")
    try:
        df = yf_download_with_retries(
            log,
            tickers=yf_symbol,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return df.dropna(how="all") if df is not None else None
    except Exception as e:
        diagnostics.setdefault("symbol_errors", {}).setdefault(tracked_symbol, []).append(
            {
                "where": "yf_download_intraday_1h",
                "tracked_symbol": tracked_symbol,
                "yfinance_ticker": yf_symbol,
                "error": str(e),
                "trace": short_tb(e),
            }
        )
        return None


def get_row_value(row: Any, field: str) -> Any:
    """Robustly fetch a value from a pandas Series row."""
    try:
        return row.get(field)
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


def fetch_yf_fundamentals(
    log: logging.Logger,
    tracked_symbol: str,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    yf_symbol = resolve_yf_ticker(tracked_symbol)
    log.debug(f"yfinance fundamentals: {tracked_symbol} -> {yf_symbol}")
    try:
        t = yf.Ticker(yf_symbol)
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
        curated["yf_ticker"] = yf_symbol
        return {"curated_info": curated}
    except Exception as e:
        diagnostics.setdefault("symbol_errors", {}).setdefault(tracked_symbol, []).append(
            {
                "where": "fetch_yf_fundamentals",
                "tracked_symbol": tracked_symbol,
                "yfinance_ticker": yf_symbol,
                "error": str(e),
                "trace": short_tb(e),
            }
        )
        return {"error": str(e), "yf_ticker": yf_symbol}


def fetch_ibkr_fundamentals(
    log: logging.Logger,
    ib: IB,
    symbol: str,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Fetch IBKR fundamentals (ReportSnapshot) using in-script contract overrides."""
    out: Dict[str, Any] = {"symbol": symbol}
    try:
        exchange, currency, primary_exchange = resolve_ibkr_contract(symbol)

        contract = Stock(symbol, exchange, currency)
        if primary_exchange:
            contract.primaryExchange = primary_exchange

        ib.qualifyContracts(contract)

        out["contract"] = {
            "conId": contract.conId,
            "secType": contract.secType,
            "exchange": contract.exchange,
            "primaryExchange": getattr(contract, "primaryExchange", None),
            "currency": contract.currency,
        }

        log.debug(f"IBKR fundamentals: requesting ReportSnapshot for {symbol} ({out['contract']})")
        xml = ib.reqFundamentalData(contract, reportType="ReportSnapshot")
        out["report_type"] = "ReportSnapshot"
        out["raw_snapshot_xml"] = xml

    except Exception as e:
        diagnostics.setdefault("symbol_errors", {}).setdefault(symbol, []).append(
            {
                "where": "fetch_ibkr_fundamentals",
                "tracked_symbol": symbol,
                "error": str(e),
                "trace": short_tb(e),
            }
        )
        out["error"] = str(e)

    return out


def build_meta(mode_cfg: ModeConfig, tracked: List[str]) -> Dict[str, Any]:
    return {
        "generated_at_utc": iso_now_utc(),
        "mode": mode_cfg.mode,
        "tracked_symbols": tracked,
        "mappings": {
            "yfinance_overrides": YF_TICKER_OVERRIDES,
            "ibkr_contract_overrides": IBKR_CONTRACT_OVERRIDES,
        },
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
    log = setup_logging(args.debug)

    try:
        ib = connect_ib(args.host, args.port, args.client_id, log)
    except IBKRConnectionError:
        # clean early exit, no traceback
        return 2

    mode_cfg = build_mode_config(args)

    tracked_symbols = parse_csv_symbols(args.symbols)

    ensure_dir(args.output_dir)

    diagnostics: Dict[str, Any] = {
        "run": {"debug": bool(args.debug)},
        "global_errors": [],
        "symbol_errors": {},
    }

    # Print resolved mapping up-front (so we can spot if IB1T isn't mapped)
    resolved_map = {s: resolve_yf_ticker(s) for s in tracked_symbols}
    log.info(f"Resolved yfinance tickers: {resolved_map}")
    diagnostics["run"]["resolved_yfinance_map"] = resolved_map

    try:
        portfolio_base = fetch_portfolio(ib)
        portfolio_derived = derive_tracked(portfolio_base, tracked_symbols)
        portfolio = {**portfolio_base, **portfolio_derived}

        open_orders: List[Dict[str, Any]] = []
        if mode_cfg.include_open_orders:
            try:
                open_orders = fetch_open_orders(ib)
            except Exception as e:
                diagnostics["global_errors"].append({"where": "fetch_open_orders", "error": str(e), "trace": short_tb(e)})
                open_orders = []

        daily_dfs = yf_download_daily(log, tracked_symbols, years=mode_cfg.daily_years, diagnostics=diagnostics)

        symbols_doc: Dict[str, Any] = {}
        for symbol in tracked_symbols:
            log.info(f"Processing symbol {symbol} (yfinance={resolve_yf_ticker(symbol)})")
            df_daily = daily_dfs.get(symbol)
            daily_records = df_to_records(df_daily, max_rows=mode_cfg.daily_max_rows, is_intraday=False)
            core_metrics = compute_core_metrics(daily_records)

            sym_doc: Dict[str, Any] = {
                "core_metrics": core_metrics,
                "history": {"daily": daily_records},
                "yfinance_ticker": resolve_yf_ticker(symbol),
            }

            if mode_cfg.include_intraday:
                df_intra = yf_download_intraday_1h(log, symbol, diagnostics)
                intraday_records = df_to_records(df_intra, max_rows=INTRADAY_MAX_ROWS, is_intraday=True)
                sym_doc["history"]["intraday_1h_6m"] = intraday_records

            if mode_cfg.include_fundamentals:
                fundamentals: Dict[str, Any] = {}
                if FETCH_YFINANCE_FUNDAMENTALS_IN_FULL and not args.no_yf_fundamentals:
                    fundamentals["yfinance"] = fetch_yf_fundamentals(log, symbol, diagnostics)
                if FETCH_IBKR_FUNDAMENTALS_IN_FULL and not args.no_ibkr_fundamentals:
                    fundamentals["ibkr"] = fetch_ibkr_fundamentals(log, ib, symbol, diagnostics)
                sym_doc["fundamentals"] = fundamentals

            if open_orders:
                sym_doc["open_orders"] = [o for o in open_orders if (o.get("symbol") or "").upper() == symbol]

            symbols_doc[symbol] = sym_doc

        meta = build_meta(mode_cfg, tracked_symbols)
        snapshot: Dict[str, Any] = {
            "meta": meta,
            "diagnostics": diagnostics,
            "portfolio": portfolio,
            "symbols": symbols_doc,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"portfolio_snapshot_{ts}.json"
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        log.info(f"Wrote snapshot: {path}")
        print(f"SNAPSHOT_PATH={path}")
        return 0
    finally:
        ib.disconnect()
        log.info("Disconnected from IBKR.")


if __name__ == "__main__":
    raise SystemExit(main())
