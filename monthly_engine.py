#!/usr/bin/env python3
"""
MONTHLY ENGINE – DESIGN CONTRACT (READ THIS FIRST)

This file implements a DISCIPLINE ENGINE, not an optimizer and not a trading bot.

CORE PHILOSOPHY
---------------
- Risk-first, not return-maximizing
- Deterministic, explainable, auditable
- No reaction to short-term noise
- No dependency on precise market timing

The engine answers ONE question:
"Given today’s snapshot, how much am I allowed to buy NOW,
 and under which conditions am I allowed to buy MORE later?"

UNIVERSES
---------
1) ACTIONS (core portfolio)
   - Baseline DCA by target weights
   - Opportunistic overlay based on 12m drawdown + MA200 pacing
   - Hard caps on position weights
   - Cash accumulation is intentional (blocked DCA is not redistributed)

2) BTC (ETF, satellite)
   - Opportunistic ONLY (no DCA)
   - Same methodology as actions but different thresholds (BTC is more volatile)
   - BTC is NOT constrained by the actions position-weight caps in this file
   - BTC budget is NOT cash-constrained by the snapshot (cash can be deposited later)

IMPORTANT NON-OBVIOUS RULES
---------------------------
- Drawdowns are treated as CONTINUOUS variables, not binary triggers (linear intensity).
- MA200 is used for PACING (throttling), not timing. It is NOT an entry signal.
- TRANCHE-2 uses LIMIT orders to encode future conditions ("statefulness without state").
- Snapshot cash is NOT a hard constraint (the engine decides allocation, execution can happen later).

USAGE
-----
# Uses ./config.json and latest snapshot in ./snapshots
python monthly_engine.py

# Explicit snapshot path
python monthly_engine.py --snapshot snapshots/portfolio_snapshot_YYYYMMDD_HHMM.json

# Explicit config path + write output allocations JSON
python monthly_engine.py --config config.json --output allocations.json

CONFIG SHAPE (new opportunistic format, actions)
-----------------------------------------------
{
  "symbols": { ... },
  "dca": { ... },
  "opportunistic": {
    "enabled": true,
    "tranche1": {
      "monthly_budget_usd": 3000,
      "entry_drawdown_pct": -15,
      "full_drawdown_pct": -25,
      "buy_pct_at_entry": 0.30,
      "execution": "BUY_NOW",
      "overrides": [ ... ]
    },
    "tranche2": {
      "monthly_budget_usd": 3000,
      "entry_drawdown_pct": -25,
      "full_drawdown_pct": -40,
      "buy_pct_at_entry": 0.60,
      "execution": "LIMIT_ONLY",
      "ma200_threshold_multiplier": 1.00,
      "overrides": [ ... ]
    },
    "tranche3": { ... }
  },
  "btc": { ... }   # unchanged
}

SNAPSHOT ASSUMPTIONS
--------------------
This engine expects an IBKR+market snapshot JSON with at least:
- portfolio.derived.tracked_weights_pct (for tracked action symbols)
- symbols.<SYMBOL>.core_metrics.drawdown_12m_close_pct
- symbols.<SYMBOL>.core_metrics.price_vs_ma200_pct
- symbols.<SYMBOL>.core_metrics.high_12m_close
- symbols.<SYMBOL>.core_metrics.ma200_close

If BTC is enabled, the BTC ETF symbol must exist under snapshot["symbols"] too.
"""

from __future__ import annotations
import yfinance as yf

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ----------------------------- helpers ----------------------------- #

def safe_float(x: Any, default: float = float("nan")) -> float:
    """Convert to float safely (handles None, '', '123.45')."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def find_latest_snapshot(snapshot_dir: Path) -> Optional[Path]:
    """
    Pick the most recent file matching 'portfolio_snapshot_*.json' in snapshot_dir.
    Works cross-platform (mtime-based).
    """
    if not snapshot_dir.exists():
        return None
    candidates = sorted(snapshot_dir.glob("portfolio_snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def round_money(x: float) -> float:
    """Round money to cents (bankers rounding not important here)."""
    if math.isnan(x):
        return x
    return float(f"{x:.2f}")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_key_utc() -> str:
    d = datetime.now(timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


def compute_gtd_end_of_month_utc(hour: int = 20, minute: int = 0) -> str:
    """
    GTD end of month timestamp (UTC) as ISO string. This is only for reporting.
    (Your order placement layer can convert this to broker-specific format.)
    """
    d = datetime.now(timezone.utc)
    year, month = d.year, d.month
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    end_month = next_month - timedelta(seconds=1)  # 23:59:59 last day
    # Replace time to desired hour:minute, clamp to same day
    return end_month.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------- configuration ----------------------------- #

@dataclass(frozen=True)
class DcaConfig:
    monthly_budget_usd: float
    target_weights_pct: Dict[str, float]
    block_if_current_weight_pct_ge_target: bool
    block_tolerance_pct: float  # allow small drift


@dataclass(frozen=True)
class BtcTranche2Config:
    enabled: bool
    drawdown_unlock_threshold_pct: float
    ma200_multiplier: float  # single multiplier for BTC ETF
    tif: str
    gtd_end_of_month_utc_hour: int
    gtd_end_of_month_utc_minute: int


@dataclass(frozen=True)
class BtcOpportunisticConfig:
    """
    Used for BTC opportunistic (satellite).
    BTC is more volatile => thresholds are configured separately in config.json.
    """
    monthly_budget_usd: float
    drawdown_entry_threshold_pct: float
    buy_pct_at_entry: float
    drawdown_full_allocation_pct: float
    per_monthly_cap_usd: float  # cap for the single BTC instrument
    ma200_penalty_start_pct: float
    ma200_penalty_full_pct: float
    ma200_max_penalty: float
    tranche2: BtcTranche2Config


@dataclass(frozen=True)
class SymbolsConfig:
    tracked_actions: List[str]
    btc_symbol: Optional[str]  # e.g. "IBIT" (None disables BTC)


@dataclass(frozen=True)
class EngineConfig:
    symbols: SymbolsConfig
    dca: DcaConfig
    opportunistic_actions: Dict[str, Any]   # NEW: raw tranches dict (stateless, generic)
    opportunistic_actions_enabled: bool
    btc: Optional[BtcOpportunisticConfig]  # None disables BTC
    raw: Dict[str, Any]                    # keep full raw config for debugging/audit


def _require_non_empty_list(raw: Any, key_path: str) -> List[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"config.json must include {key_path} as a non-empty list")
    out = []
    for x in raw:
        s = str(x).strip().upper()
        if s:
            out.append(s)
    if not out:
        raise ValueError(f"config.json {key_path} resolved to empty symbols")
    return out


def _require_dict(raw: Any, key_path: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"config.json must include {key_path} as an object")
    return raw


def load_config(config_path: Path) -> EngineConfig:
    """
    New opportunistic ACTIONS format ONLY (old format abandoned).
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    # -------- symbols
    symbols_raw = raw.get("symbols", {})
    tracked_symbols = _require_non_empty_list(symbols_raw.get("tracked"), "symbols.tracked")
    btc_symbol = None
    btc_raw_sym = (symbols_raw.get("btc") or {}).get("symbol")
    if btc_raw_sym:
        btc_symbol = str(btc_raw_sym).strip().upper() or None

    symbols_cfg = SymbolsConfig(tracked_actions=tracked_symbols, btc_symbol=btc_symbol)

    # -------- DCA (actions)
    dca_raw = raw["dca"]
    dca = DcaConfig(
        monthly_budget_usd=float(dca_raw["monthly_budget_usd"]),
        target_weights_pct={k.upper(): float(v) for k, v in dca_raw["target_weights_pct"].items()},
        block_if_current_weight_pct_ge_target=bool(dca_raw.get("block_if_current_weight_pct_ge_target", True)),
        block_tolerance_pct=float(dca_raw.get("block_tolerance_pct", 0.0)),
    )

    # -------- Opportunistic (ACTIONS) NEW FORMAT (tranches only)
    opp_raw = _require_dict(raw.get("opportunistic", {}), "opportunistic")
    opp_enabled = bool(opp_raw.get("enabled", True))

    tranches: Dict[str, Any] = {}
    for k, v in opp_raw.items():
        if not k.startswith("tranche"):
            continue
        if not isinstance(v, dict):
            raise ValueError(f"opportunistic.{k} must be an object")
        if not v.get("enabled", True):
            continue

        # minimal required keys (per tranche)
        for req in ("monthly_budget_usd", "entry_drawdown_pct", "full_drawdown_pct", "buy_pct_at_entry", "execution"):
            if req not in v:
                raise ValueError(f"opportunistic.{k} missing required key: {req}")

        exec_mode = str(v["execution"]).strip().upper()
        if exec_mode not in ("BUY_NOW", "LIMIT_ONLY"):
            raise ValueError(f"opportunistic.{k}.execution must be BUY_NOW or LIMIT_ONLY")

        # Normalize
        vv = dict(v)
        vv["execution"] = exec_mode

        # Optional TIF settings (default GTD end-of-month 20:00 UTC, to preserve output structure)
        vv.setdefault("tif", "GTD")
        vv.setdefault("gtd_end_of_month_utc_hour", 20)
        vv.setdefault("gtd_end_of_month_utc_minute", 0)

        tranches[k] = vv

    if opp_enabled and not tranches:
        raise ValueError("opportunistic.enabled is true but no tranche* blocks found")

    # -------- BTC (optional, opportunistic only) UNCHANGED
    btc_cfg: Optional[BtcOpportunisticConfig] = None
    btc_raw = raw.get("btc")
    if btc_symbol and btc_raw:
        btc_opp_raw = (btc_raw.get("opportunistic") or {})
        btc_tr2_raw = btc_opp_raw.get("tranche2") or {}
        btc_tr2 = BtcTranche2Config(
            enabled=bool(btc_tr2_raw.get("enabled", True)),
            drawdown_unlock_threshold_pct=float(btc_tr2_raw.get("drawdown_unlock_threshold_pct", -35.0)),
            ma200_multiplier=float(btc_tr2_raw.get("ma200_multiplier", 1.05)),
            tif=str(btc_tr2_raw.get("tif", "GTD")).upper(),
            gtd_end_of_month_utc_hour=int(btc_tr2_raw.get("gtd_end_of_month_utc_hour", 20)),
            gtd_end_of_month_utc_minute=int(btc_tr2_raw.get("gtd_end_of_month_utc_minute", 0)),
        )
        btc_cfg = BtcOpportunisticConfig(
            monthly_budget_usd=float(btc_raw.get("monthly_budget_usd", 0.0)),
            drawdown_entry_threshold_pct=float(btc_opp_raw.get("drawdown_entry_threshold_pct", -25.0)),
            buy_pct_at_entry=float(btc_opp_raw.get("buy_pct_at_entry", 0.25)),
            drawdown_full_allocation_pct=float(btc_opp_raw.get("drawdown_full_allocation_pct", -45.0)),
            per_monthly_cap_usd=float(btc_opp_raw.get("per_monthly_cap_usd", btc_raw.get("monthly_budget_usd", 0.0))),
            ma200_penalty_start_pct=float(btc_opp_raw.get("ma200_penalty_start_pct", 0.0)),
            ma200_penalty_full_pct=float(btc_opp_raw.get("ma200_penalty_full_pct", 15.0)),
            ma200_max_penalty=float(btc_opp_raw.get("ma200_max_penalty", 0.50)),
            tranche2=btc_tr2,
        )

    return EngineConfig(
        symbols=symbols_cfg,
        dca=dca,
        opportunistic_actions=tranches,
        opportunistic_actions_enabled=opp_enabled,
        btc=btc_cfg,
        raw=raw,
    )


# ----------------------------- core math ----------------------------- #

def compute_drawdown_intensity(
    drawdown_12m_close_pct: float,
    entry_threshold_pct: float,
    buy_pct_at_entry: float,
    full_allocation_pct: float,
) -> float:
    """
    Map drawdown to a [0..1] intensity (CONTINUOUS, NOT BINARY).

    RATIONALE:
      - Drawdowns are noisy and can revert quickly.
      - A linear intensity reduces regret and avoids all-in timing decisions.
      - We prefer robust behavior over "smart" switches.

    - If drawdown is above entry threshold (e.g. -10%), intensity = 0 (not eligible).
    - At entry threshold (e.g. -15%), intensity = buy_pct_at_entry (e.g. 0.30).
    - At full allocation drawdown (e.g. -25%), intensity = 1.0
    - Linearly interpolated between entry and full.
    """
    if math.isnan(drawdown_12m_close_pct):
        return 0.0

    # Not eligible -> 0
    if drawdown_12m_close_pct > entry_threshold_pct:
        return 0.0

    # Guard against bad config
    if full_allocation_pct >= entry_threshold_pct:
        # e.g. -20 >= -15 is wrong
        return clamp(buy_pct_at_entry, 0.0, 1.0)

    # Progress in [0..1] where 0 at entry_threshold and 1 at full_allocation
    progress = (entry_threshold_pct - drawdown_12m_close_pct) / (entry_threshold_pct - full_allocation_pct)
    progress = clamp(progress, 0.0, 1.0)

    intensity = buy_pct_at_entry + (1.0 - buy_pct_at_entry) * progress
    return clamp(intensity, 0.0, 1.0)


def compute_ma200_pacing_multiplier(
    price_vs_ma200_pct: float,
    penalty_start_pct: float,
    penalty_full_pct: float,
    max_penalty: float,
) -> float:
    """
    Convert distance to MA200 into a pacing multiplier in (0..1].

    IMPORTANT:
      - MA200 is NOT an entry signal.
      - It is ONLY a pacing / throttling mechanism.
      - Reason: strong assets can stay above MA200 for months; we want slower exposure, not zero.

    - If price is at/below penalty_start_pct (often 0%), multiplier = 1.0 (no slowdown).
    - If price is above, linearly reduce to (1 - max_penalty) at penalty_full_pct (e.g. +10%).
    - Never reduces below (1 - max_penalty).
    """
    if math.isnan(price_vs_ma200_pct):
        return 1.0

    max_pen = clamp(max_penalty, 0.0, 0.95)

    if price_vs_ma200_pct <= penalty_start_pct:
        return 1.0

    if penalty_full_pct <= penalty_start_pct:
        return 1.0 - max_pen

    if price_vs_ma200_pct >= penalty_full_pct:
        return 1.0 - max_pen

    frac = (price_vs_ma200_pct - penalty_start_pct) / (penalty_full_pct - penalty_start_pct)
    penalty = max_pen * clamp(frac, 0.0, 1.0)
    return clamp(1.0 - penalty, 0.05, 1.0)


# ----------------------------- snapshot accessors ----------------------------- #

def load_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def get_portfolio_weight_pct(snapshot: Dict[str, Any], symbol: str) -> float:
    """
    Portfolio weight % for reporting.
    If the symbol is not present in tracked_weights_pct (not held), we return 0.0 (not NaN).
    """
    weights = snapshot.get("portfolio", {}).get("derived", {}).get("tracked_weights_pct", {}) or {}
    v = safe_float(weights.get(symbol), default=float("nan"))
    return 0.0 if math.isnan(v) else v


def get_available_cash_usd(snapshot: Dict[str, Any]) -> float:
    acct = snapshot.get("portfolio", {}).get("account_summary", {})
    return safe_float(acct.get("TotalCashValue_USD"), default=float("nan"))


def get_symbol_core(snapshot: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    return snapshot.get("symbols", {}).get(symbol, {}).get("core_metrics", {}) or {}


def get_symbol_last_price(snapshot: Dict[str, Any], symbol: str) -> float:
    """Best-effort extraction of a 'current' price from the snapshot.

    This is for DISPLAY only (reporting). The engine decisions are based on
    drawdown/MA200 metrics already present in the snapshot.

    We try a few common paths to keep backward compatibility with different
    snapshot generators.
    """
    sym = (symbol or "").upper()
    sym_obj = snapshot.get("symbols", {}).get(sym, {}) or {}

    # Prefer core_metrics.last_close if present
    core = sym_obj.get("core_metrics", {}) or {}
    for k in ("last_close", "last_price", "price", "close"):
        v = safe_float(core.get(k), default=float("nan"))
        if not math.isnan(v):
            return v

    # Some generators store quotes under a different subtree
    for path in (
        ("market", "last"),
        ("market", "last_price"),
        ("quote", "last"),
        ("quote", "last_price"),
        ("quote", "mid"),
        ("quote", "close"),
    ):
        cur: Any = sym_obj
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok:
            v = safe_float(cur, default=float("nan"))
            if not math.isnan(v):
                return v

    return float("nan")


def resolve_symbol_last_price_usd(symbol: str, snapshot: Dict[str, Any], yf_ticker: Optional[str] = None) -> Tuple[float, str]:
    """Resolve last price for reporting.

    Priority:
      1) snapshot-derived price (fast, offline)
      2) yfinance fast_info/1d close (online; best-effort)

    Returns: (price, source)
    """
    snap_px = get_symbol_last_price(snapshot, symbol)
    if not math.isnan(snap_px):
        return snap_px, f"snapshot:{symbol}"

    # Best-effort yfinance fallback (do NOT fail the engine if it breaks)
    t = (yf_ticker or symbol).strip()
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None) or {}
        px = safe_float(fi.get("last_price"), default=float("nan"))
        if not math.isnan(px):
            return px, f"yfinance:{t}(fast_info)"
    except Exception:
        pass

    try:
        df = yf.download(t, period="5d", interval="1d", auto_adjust=False, progress=False, threads=False)
        closes = _extract_close_series_from_yf_df(df)
        if closes:
            return float(closes[-1]), f"yfinance:{t}(close)"
    except Exception:
        pass

    return float("nan"), f"missing:{symbol}"


# ---------------- BTC metrics fallback (self-contained, with rollback) ----------------

def _extract_close_series_from_yf_df(df) -> List[float]:
    """Extract close prices from yfinance.download() output.

    Supports both simple columns and MultiIndex columns like:
      MultiIndex([('Close','BTC-USD'), ...], names=['Price','Ticker'])
    """
    if df is None or getattr(df, "empty", False):
        return []
    cols = df.columns
    # simple case
    try:
        if "Close" in cols:
            s = df["Close"].dropna()
            return [float(x) for x in s.tolist()]
    except Exception:
        pass

    # MultiIndex case
    try:
        import pandas as pd
        if isinstance(cols, pd.MultiIndex):
            # Prefer any column where any level equals 'Close'
            for lvl in range(cols.nlevels):
                if "Close" in cols.get_level_values(lvl):
                    close_cols = [c for c in cols if c[lvl] == "Close"]
                    if close_cols:
                        s = df[close_cols[0]].dropna()
                        return [float(x) for x in s.tolist()]
    except Exception:
        pass
    return []


def _compute_core_metrics_from_closes(closes: List[float]) -> Dict[str, Any]:
    """Compute core metrics with the SAME semantics as your snapshot generator.
    - 12M metrics require >=252 trading days
    - MA200 requires >=200 trading days
    """
    if not closes:
        return {
            "last_close": None,
            "high_12m_close": None,
            "drawdown_12m_close_pct": None,
            "ma200_close": None,
            "price_vs_ma200_pct": None,
            "below_ma200": None,
        }

    last_close = float(closes[-1])

    if len(closes) < 252:
        high_12m = None
        dd12m = None
    else:
        high_12m = float(max(closes[-252:]))
        dd12m = (last_close / high_12m - 1.0) * 100.0

    if len(closes) < 200:
        ma200 = None
        pvma = None
        below = None
    else:
        ma200 = float(sum(closes[-200:]) / 200.0)
        pvma = (last_close / ma200 - 1.0) * 100.0
        below = last_close < ma200

    return {
        "last_close": last_close,
        "high_12m_close": high_12m,
        "drawdown_12m_close_pct": dd12m,
        "ma200_close": ma200,
        "price_vs_ma200_pct": pvma,
        "below_ma200": below,
    }

def _missing_key_core_metrics(core: Dict[str, Any]) -> List[str]:
    missing = []

    dd = safe_float(core.get("drawdown_12m_close_pct"), float("nan"))
    hi = safe_float(core.get("high_12m_close"), float("nan"))
    ma = safe_float(core.get("ma200_close"), float("nan"))

    if math.isnan(dd):
        missing.append("drawdown_12m_close_pct")
    if math.isnan(hi):
        missing.append("high_12m_close")
    if math.isnan(ma):
        missing.append("ma200_close")

    return missing

def _has_required_core_metrics(core: Dict[str, Any]) -> bool:
    """True if DD12M and MA200 metrics are present (non-None)."""
    if not core:
        return False
    return (
        core.get("drawdown_12m_close_pct") is not None
        and core.get("price_vs_ma200_pct") is not None
        and core.get("high_12m_close") is not None
        and core.get("ma200_close") is not None
    )


def _get_yf_core_metrics(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch 5y daily history via yfinance and compute core metrics."""
    try:
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        closes = _extract_close_series_from_yf_df(df)
        core = _compute_core_metrics_from_closes(closes)
        return core if _has_required_core_metrics(core) else None
    except Exception:
        return None

def _core_metrics_complete(core: Dict[str, Any]) -> bool:
    if not core:
        return False
    dd = safe_float(core.get("drawdown_12m_close_pct"), float("nan"))
    pv = safe_float(core.get("price_vs_ma200_pct"), float("nan"))
    h = safe_float(core.get("high_12m_close"), float("nan"))
    m = safe_float(core.get("ma200_close"), float("nan"))
    return (not math.isnan(dd)) and (not math.isnan(pv)) and (not math.isnan(h)) and (not math.isnan(m))


def resolve_actions_core_metrics_non_strict(snapshot: Dict[str, Any], symbol: str) -> Tuple[Dict[str, Any], str]:
    """
    Return (core_metrics, source).
    Priority:
      1) snapshot symbols[symbol].core_metrics if complete
      2) yfinance download() computed metrics (5y daily)
      3) snapshot core_metrics (incomplete) as last resort
    """
    sym = symbol.upper()
    snap_core = get_symbol_core(snapshot, sym) or {}
    if _core_metrics_complete(snap_core):
        return snap_core, f"snapshot:{sym}"

    # yfinance fallback (best-effort)
    core = _get_yf_core_metrics(sym)  # already defined in your file (BTC section)
    if core and _core_metrics_complete(core):
        return core, f"yfinance:{sym}"

    return snap_core, f"snapshot:{sym}(incomplete)"

def resolve_actions_core_metrics(snapshot: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    STRICT MODE:
    - Snapshot is the single source of truth
    - No fallback (no yfinance)
    - Missing data MUST stay missing (NaN)
    """
    return get_symbol_core(snapshot, symbol) or {}


def resolve_btc_core_metrics(cfg: "EngineConfig", snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Resolve BTC core metrics with automatic rollback.

    Priority:
      1) snapshot core_metrics for trade symbol, if complete (rollback when IB1T has enough history)
      2) yfinance IB1T.SW
      3) yfinance BTC-USD
      4) snapshot incomplete metrics (last resort, will lead to NaN/skip)
    """
    trade_sym = cfg.symbols.btc_symbol
    snap_core = get_symbol_core(snapshot, trade_sym) if trade_sym else {}
    if trade_sym and _has_required_core_metrics(snap_core):
        return snap_core, f"snapshot:{trade_sym}"

    # Fallback sources (metrics only)
    core = _get_yf_core_metrics("IB1T.SW")
    if core:
        return core, "yfinance:IB1T.SW"
    core = _get_yf_core_metrics("BTC-USD")
    if core:
        return core, "yfinance:BTC-USD"

    # nothing usable
    return snap_core, f"snapshot:{trade_sym}(insufficient_history)"


# ----------------------------- ACTIONS: DCA ----------------------------- #

def build_dca_plan(cfg: EngineConfig, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Baseline DCA: distribute monthly budget by target weights.

    DESIGN CHOICE (IMPORTANT):
      If a symbol is blocked (already >= target weight),
      its DCA budget is NOT redistributed to others.

    RATIONALE:
      - Prevents over-optimization
      - Accumulates dry powder
      - Preserves discipline over "always invested" bias

    NEW RULE (2026-01):
      Symbols present in symbols.tracked but missing in dca.target_weights_pct
      are treated as "opportunistic-only" => DCA amount = 0, and MUST NOT
      contaminate totals with NaN.
    """
    budget = cfg.dca.monthly_budget_usd
    planned: Dict[str, Any] = {}
    spent = 0.0

    for symbol in cfg.symbols.tracked_actions:
        # Read target weight
        target_pct = safe_float(cfg.dca.target_weights_pct.get(symbol), default=float("nan"))

        current_weight = get_portfolio_weight_pct(snapshot, symbol)

        blocked = False
        reasons: List[str] = []

        # If symbol has no DCA target, treat as opportunistic-only (DCA=0)
        if math.isnan(target_pct):
            planned_amount = 0.0
            blocked = True
            reasons.append("no_dca_target_weight (opportunistic-only)")
        else:
            planned_amount = budget * (target_pct / 100.0)

            # Normal DCA block rule
            if cfg.dca.block_if_current_weight_pct_ge_target and not math.isnan(current_weight):
                if current_weight >= (target_pct - cfg.dca.block_tolerance_pct):
                    blocked = True
                    reasons.append(
                        f"current_weight_pct {current_weight:.2f} >= target_weight_pct {target_pct:.2f}"
                    )

        final_amount = 0.0 if blocked else planned_amount
        final_amount = round_money(final_amount)

        planned[symbol] = {
            "target_weight_pct": target_pct,
            "current_weight_pct": current_weight,
            "planned_amount_usd": round_money(planned_amount),
            "blocked": blocked,
            "block_reasons": reasons,
            "final_amount_usd": final_amount,
        }

        # IMPORTANT: use final_amount to avoid NaN contaminating totals
        spent += final_amount

    spent = round_money(spent)
    return planned, spent


# ----------------------------- ACTIONS: Opportunistic (generic tranches) ----------------------------- #

def _resolve_tranche_params(tranche_cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Apply overrides for a tranche if symbol is in a group.
    Override structure is IDENTICAL to tranche structure.
    """
    out = dict(tranche_cfg)
    for ov in tranche_cfg.get("overrides", []):
        groups = [str(x).strip().upper() for x in (ov.get("groups") or [])]
        if symbol.upper() in groups:
            for k, v in ov.items():
                if k != "groups":
                    out[k] = v
    return out


def build_opportunistic_plan(cfg: EngineConfig, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float, List[Dict[str, Any]]]:
    """
    Opportunistic (ACTIONS) – NEW FORMAT:
      - N tranches (tranche1, tranche2, tranche3, ...)
      - Same formula everywhere (continuous linear intensity by drawdown)
      - Tranches independent (no TR1 -> TR2 dependency)
      - BUY_NOW tranches contribute to OPP_TR1_USD output column (sum)
      - LIMIT_ONLY tranches generate limit orders list (returned separately)

    IMPORTANT ASSUMPTION:
      Snapshot cash is NOT a hard constraint.
      Reason: cash can be deposited / converted after this script runs.
      -> budgets are used as-is (desired == effective).
    """
    if not cfg.opportunistic_actions_enabled:
        # preserve structure
        plan = {}
        for symbol in cfg.symbols.tracked_actions:
            core = get_symbol_core(snapshot, symbol)
            plan[symbol] = {
                "eligible": False,
                "current_weight_pct": get_portfolio_weight_pct(snapshot, symbol),
                "drawdown_12m_close_pct": safe_float(core.get("drawdown_12m_close_pct"), float("nan")),
                "price_vs_ma200_pct": safe_float(core.get("price_vs_ma200_pct"), float("nan")),
                "base_budget_usd": 0.0,
                "drawdown_intensity": 0.0,
                "ma200_pacing_multiplier": 1.0,
                "raw_amount_usd": 0.0,
                "capped_amount_usd": 0.0,
                "blocked_reasons": ["opportunistic_disabled"],
                "buy_now_by_tranche": {},
            }
        return plan, 0.0, 0.0, []

    desired_effective_budget = 0.0
    limit_orders: List[Dict[str, Any]] = []

    # We keep the legacy per-symbol plan shape, but "capped_amount_usd"
    # becomes "total BUY_NOW across all BUY_NOW tranches" (typically tranche1).
    plan: Dict[str, Any] = {}
    for symbol in cfg.symbols.tracked_actions:
        core = get_symbol_core(snapshot, symbol) or {}

        missing_keys = _missing_key_core_metrics(core)
        if missing_keys:
            # Block ALL opportunistic for this symbol (strict safety)
            plan[symbol] = {
                "eligible": False,
                "current_weight_pct": get_portfolio_weight_pct(snapshot, symbol),
                "drawdown_12m_close_pct": safe_float(core.get("drawdown_12m_close_pct"), float("nan")),
                "price_vs_ma200_pct": safe_float(core.get("price_vs_ma200_pct"), float("nan")),
                "base_budget_usd": 0.0,
                "drawdown_intensity": 0.0,
                "ma200_pacing_multiplier": 1.0,
                "raw_amount_usd": 0.0,
                "capped_amount_usd": 0.0,
                "blocked_reasons": [f"missing_core_metrics: {', '.join(missing_keys)}"],
                "buy_now_by_tranche": {},
            }
            continue

        plan[symbol] = {
            "eligible": False,  # becomes True if any BUY_NOW tranche allocates >0
            "current_weight_pct": get_portfolio_weight_pct(snapshot, symbol),
            "drawdown_12m_close_pct": safe_float(core.get("drawdown_12m_close_pct"), float("nan")),
            "price_vs_ma200_pct": safe_float(core.get("price_vs_ma200_pct"), float("nan")),
            "base_budget_usd": 0.0,  # informational (for tranche1 only, kept for report)
            "drawdown_intensity": 0.0,  # informational (for tranche1 only, kept for report)
            "ma200_pacing_multiplier": 1.0,  # unused for actions in new approach
            "raw_amount_usd": 0.0,  # sum BUY_NOW raw
            "capped_amount_usd": 0.0,  # sum BUY_NOW
            "blocked_reasons": [],
            "buy_now_by_tranche": {},  # { trancheX: amount }
        }

    total_buy_now_spent = 0.0

    # Process each tranche independently
    tranche_items = sorted(cfg.opportunistic_actions.items(), key=lambda kv: kv[0])
    for tranche_name, tranche_cfg in tranche_items:
        monthly_budget = float(tranche_cfg["monthly_budget_usd"])
        desired_effective_budget += monthly_budget
        if monthly_budget <= 0:
            continue

        # determine eligible symbols for this tranche
        tranche_eligible: List[str] = []
        tranche_blocked: Dict[str, List[str]] = {}

        for symbol in cfg.symbols.tracked_actions:
            core = get_symbol_core(snapshot, symbol)
            dd = safe_float(core.get("drawdown_12m_close_pct"), float("nan"))
            params = _resolve_tranche_params(tranche_cfg, symbol)
            entry_dd = float(params["entry_drawdown_pct"])

            reasons: List[str] = []
            if math.isnan(dd) or dd > entry_dd:
                reasons.append(f"not_eligible: drawdown_12m_close_pct {dd if not math.isnan(dd) else 'NaN'} > entry_threshold {entry_dd}")

            if reasons:
                tranche_blocked[symbol] = reasons
            else:
                tranche_eligible.append(symbol)

        if not tranche_eligible:
            continue

        base_budget = monthly_budget / len(tranche_eligible)

        for symbol in tranche_eligible:
            core = get_symbol_core(snapshot, symbol)
            dd = safe_float(core.get("drawdown_12m_close_pct"), float("nan"))
            high_12m_close = safe_float(core.get("high_12m_close"), float("nan"))
            ma200_close = safe_float(core.get("ma200_close"), float("nan"))

            params = _resolve_tranche_params(tranche_cfg, symbol)
            entry_dd = float(params["entry_drawdown_pct"])
            full_dd = float(params["full_drawdown_pct"])
            buy_at_entry = float(params["buy_pct_at_entry"])
            execution = str(params["execution"]).upper()

            intensity = compute_drawdown_intensity(
                drawdown_12m_close_pct=dd,
                entry_threshold_pct=entry_dd,
                buy_pct_at_entry=buy_at_entry,
                full_allocation_pct=full_dd,
            )

            amount = round_money(base_budget * intensity)
            if amount <= 0:
                continue

            if execution == "BUY_NOW":
                plan[symbol]["eligible"] = True
                plan[symbol]["raw_amount_usd"] = round_money(plan[symbol]["raw_amount_usd"] + amount)
                plan[symbol]["capped_amount_usd"] = round_money(plan[symbol]["capped_amount_usd"] + amount)
                plan[symbol]["buy_now_by_tranche"][tranche_name] = amount
                total_buy_now_spent += amount

                # Keep tranche1-compatible fields populated for reporting (best-effort).
                # If tranche1 is BUY_NOW, this keeps old report semantics close to previous output.
                if tranche_name == "tranche1":
                    plan[symbol]["base_budget_usd"] = round_money(base_budget)
                    plan[symbol]["drawdown_intensity"] = round(intensity, 4)

            elif execution == "LIMIT_ONLY":
                # limit_price = min(high_12m_close*(1+entry_dd/100), ma200_close*multiplier)
                if math.isnan(high_12m_close) or math.isnan(ma200_close):
                    continue
                mult = float(params.get("ma200_threshold_multiplier", 1.0))
                price_dd_unlock = high_12m_close * (1.0 + entry_dd / 100.0)
                price_ma2 = ma200_close * mult
                limit_price = round_money(min(price_dd_unlock, price_ma2))

                tif = str(params.get("tif", "GTD")).upper()
                gtd_until_utc = compute_gtd_end_of_month_utc(
                    int(params.get("gtd_end_of_month_utc_hour", 20)),
                    int(params.get("gtd_end_of_month_utc_minute", 0)),
                ) if tif == "GTD" else None

                limit_orders.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "order_type": "LMT",
                    "amount_usd": amount,
                    "limit_price": limit_price,
                    "tif": tif,
                    "gtd_until_utc": gtd_until_utc,
                    "tag": f"OPP_{tranche_name.upper()}_{month_key_utc()}",
                    "rationale": {
                        "unlock_rule": f"DD12M <= {entry_dd:.2f}% AND price <= MA200*multiplier",
                        "computed": {
                            "high_12m_close": round_money(high_12m_close),
                            "ma200_close": round_money(ma200_close),
                            "ma200_multiplier": mult,
                            "price_dd_unlock": round_money(price_dd_unlock),
                            "price_ma200_threshold": round_money(price_ma2),
                            "limit_price": limit_price,
                            "tranche": tranche_name,
                            "entry_drawdown_pct": entry_dd,
                            "full_drawdown_pct": full_dd,
                            "buy_pct_at_entry": buy_at_entry,
                        }
                    },
                })

    # finalize blocked reasons (for symbols with no BUY_NOW allocation)
    for symbol in cfg.symbols.tracked_actions:
        if not plan[symbol]["eligible"]:
            plan[symbol]["blocked_reasons"] = plan[symbol]["blocked_reasons"] or ["no_buy_now_tranche_allocations"]

    return plan, round_money(total_buy_now_spent), round_money(desired_effective_budget), limit_orders


# ----------------------------- BTC: Opportunistic only ----------------------------- #

def build_btc_opportunistic_plan(
    cfg: EngineConfig,
    snapshot: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    """
    BTC is a satellite. We run a single-instrument opportunistic plan (TR1 now + TR2 limit).

    IMPORTANT:
      - No DCA for BTC in this engine (explicitly disabled by design).
      - BTC is NOT constrained by the actions position weight caps here.
      - Snapshot cash is NOT a hard constraint.
    """
    if not cfg.btc or not cfg.symbols.btc_symbol:
        return None, 0.0, 0.0

    btc = cfg.btc
    sym = cfg.symbols.btc_symbol

    desired_budget = float(btc.monthly_budget_usd)
    effective_budget = max(0.0, desired_budget)
    core, metrics_source = resolve_btc_core_metrics(cfg, snapshot)
    if not core:
        # Missing from snapshot -> disable but report
        return {
            "symbol": sym, "metrics_source": metrics_source,
            "enabled": True,
            "error": f"BTC symbol '{sym}' not found in snapshot['symbols']",
        }, 0.0, effective_budget

    drawdown = safe_float(core.get("drawdown_12m_close_pct"), default=float("nan"))
    price_vs_ma200 = safe_float(core.get("price_vs_ma200_pct"), default=float("nan"))
    last_px = safe_float(core.get("last_close"), default=float("nan"))
    if math.isnan(last_px):
        last_px = get_symbol_last_price(snapshot, sym)

    # Eligibility: drawdown threshold
    if math.isnan(drawdown) or drawdown > btc.drawdown_entry_threshold_pct:
        plan = {
            "symbol": sym,
            "eligible": False,
            "last_price": (None if math.isnan(last_px) else float(last_px)),
            "last_price_source": metrics_source if not math.isnan(last_px) else "unavailable",
            "drawdown_12m_close_pct": drawdown,
            "price_vs_ma200_pct": price_vs_ma200,
            "base_budget_usd": round_money(effective_budget),
            "drawdown_intensity": 0.0,
            "ma200_pacing_multiplier": 1.0,
            "raw_amount_usd": 0.0,
            "tr1_amount_usd": 0.0,
            "blocked_reasons": [
                f"not_eligible: drawdown_12m_close_pct {drawdown if not math.isnan(drawdown) else 'NaN'} > entry_threshold {btc.drawdown_entry_threshold_pct}"
            ],
        }
        return plan, 0.0, effective_budget

    drawdown_intensity = compute_drawdown_intensity(
        drawdown_12m_close_pct=drawdown,
        entry_threshold_pct=btc.drawdown_entry_threshold_pct,
        buy_pct_at_entry=btc.buy_pct_at_entry,
        full_allocation_pct=btc.drawdown_full_allocation_pct,
    )

    ma200_multiplier = compute_ma200_pacing_multiplier(
        price_vs_ma200_pct=price_vs_ma200,
        penalty_start_pct=btc.ma200_penalty_start_pct,
        penalty_full_pct=btc.ma200_penalty_full_pct,
        max_penalty=btc.ma200_max_penalty,
    )

    raw_amount = effective_budget * drawdown_intensity * ma200_multiplier
    tr1_amount = min(raw_amount, btc.per_monthly_cap_usd)

    plan = {
        "symbol": sym,
        "eligible": True,
        "last_price": (None if math.isnan(last_px) else float(last_px)),
        "last_price_source": metrics_source if not math.isnan(last_px) else "unavailable",
        "drawdown_12m_close_pct": drawdown,
        "price_vs_ma200_pct": price_vs_ma200,
        "base_budget_usd": round_money(effective_budget),
        "drawdown_intensity": round(drawdown_intensity, 4),
        "ma200_pacing_multiplier": round(ma200_multiplier, 4),
        "raw_amount_usd": round_money(raw_amount),
        "tr1_amount_usd": round_money(tr1_amount),
        "blocked_reasons": [],
    }
    return plan, round_money(tr1_amount), round_money(effective_budget)


def build_tranche2_limit_order_btc(
    cfg: EngineConfig,
    snapshot: Dict[str, Any],
    btc_plan: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    BTC TR2: LIMIT order that encodes:
      (DD12M <= unlock_threshold) AND (price <= MA200 * multiplier)

    limit_price = min(high_12m_close*(1+unlock/100), ma200_close*multiplier)

    Amount:
      remaining = base_budget - tr1_amount
      capped by per_monthly_cap_usd
    """
    if not cfg.btc or not cfg.symbols.btc_symbol or not btc_plan:
        return None, None

    btc = cfg.btc
    tr2 = btc.tranche2
    sym = cfg.symbols.btc_symbol

    core, metrics_source = resolve_btc_core_metrics(cfg, snapshot)

    if not tr2.enabled:
        return None, None

    tr1_amount = safe_float(btc_plan.get("tr1_amount_usd"), default=0.0)
    base_budget = safe_float(btc_plan.get("base_budget_usd"), default=0.0)
    if tr1_amount <= 0.0 or base_budget <= 0.0:
        return None, None

    remaining = max(0.0, base_budget - tr1_amount)
    tr2_amount = min(remaining, btc.per_monthly_cap_usd)
    tr2_amount = round_money(tr2_amount)
    if tr2_amount <= 0.0:
        return None, None
    high_12m_close = safe_float(core.get("high_12m_close"), default=float("nan"))
    ma200_close = safe_float(core.get("ma200_close"), default=float("nan"))

    if math.isnan(high_12m_close) or math.isnan(ma200_close):
        return None, f"{sym}: cannot compute BTC TR2 limit (need high_12m_close and ma200_close)"

    unlock_dd = tr2.drawdown_unlock_threshold_pct
    price_dd_unlock = high_12m_close * (1.0 + unlock_dd / 100.0)
    price_ma2 = ma200_close * float(tr2.ma200_multiplier)

    limit_price = round_money(min(price_dd_unlock, price_ma2))
    gtd_until_utc = compute_gtd_end_of_month_utc(tr2.gtd_end_of_month_utc_hour, tr2.gtd_end_of_month_utc_minute) if tr2.tif == "GTD" else None

    order = {
        "symbol": sym,
        "metrics_source": metrics_source,
        "side": "BUY",
        "order_type": "LMT",
        "amount_usd": tr2_amount,
        "limit_price": limit_price,
        "tif": tr2.tif,
        "gtd_until_utc": gtd_until_utc,
        "tag": f"BTC_TR2_{month_key_utc()}",
        "rationale": {
            "unlock_rule": f"DD12M <= {unlock_dd:.2f}% AND price <= MA200*{float(tr2.ma200_multiplier):.4f}",
            "computed": {
                "high_12m_close": round_money(high_12m_close),
                "ma200_close": round_money(ma200_close),
                "ma200_multiplier": float(tr2.ma200_multiplier),
                "price_dd_unlock": round_money(price_dd_unlock),
                "price_ma200_threshold": round_money(price_ma2),
                "limit_price": limit_price,
            }
        },
    }
    return order, None


# ----------------------------- reporting ----------------------------- #

def format_table_row(cols: List[str], widths: List[int]) -> str:
    padded = []
    for c, w in zip(cols, widths):
        c = "" if c is None else str(c)
        padded.append(c[:w].ljust(w))
    return " | ".join(padded)


def print_opportunistic_walkthrough_actions(cfg: EngineConfig, snapshot: Dict[str, Any], opp_plan: Dict[str, Any], totals: Dict[str, float]) -> None:
    print("Opportunistic Decision Walkthrough (ACTIONS):")
    if not cfg.opportunistic_actions_enabled:
        print("  Opportunistic disabled.")
        print("")
        return

    # For legacy report continuity, we describe tranche1 if present.
    tr1 = cfg.opportunistic_actions.get("tranche1")
    if tr1:
        print(f"  Tranche1 budget (desired): {float(tr1['monthly_budget_usd']):.2f} USD")
    else:
        print("  Tranche1 not found. (Report still shows BUY_NOW sums in OPP_TR1_USD.)")

    if not math.isnan(totals.get('available_cash_usd', float('nan'))):
        print(f"  Available cash (snapshot, informational): {totals['available_cash_usd']:.2f} USD")
    print(f"  Opportunistic effective budget (sum of tranche budgets): {totals['opp_effective_budget_usd']:.2f} USD")
    print("")
    print("  Eligibility rule (per tranche):")
    print("    - drawdown_12m_close_pct <= entry_drawdown_pct")
    print("    - intensity is continuous linear (entry -> full)")
    print("  Execution:")
    print("    - BUY_NOW tranches: contribute to OPP_TR1_USD (summed)")
    print("    - LIMIT_ONLY tranches: emitted as limit orders (TR2/3/...)")
    print("")

    eligible = [s for s in cfg.symbols.tracked_actions if opp_plan.get(s, {}).get("eligible")]
    blocked = [s for s in cfg.symbols.tracked_actions if s not in eligible]

    if not eligible:
        print("  Result: no BUY_NOW allocations this run → opportunistic buy-now allocation is 0 for all symbols.")
        print("")
        return

    print(f"  Symbols with BUY_NOW allocations ({len(eligible)}): {', '.join(eligible)}")
    print("")

    for symbol in eligible:
        rec = opp_plan[symbol]
        dd = safe_float(rec.get("drawdown_12m_close_pct"))
        pv = safe_float(rec.get("price_vs_ma200_pct"))
        total_buy_now = safe_float(rec.get("capped_amount_usd"))
        by_tranche = rec.get("buy_now_by_tranche", {})

        print(f"  {symbol}:")
        print(f"    drawdown_12m_close_pct = {dd:.2f}%")
        if not math.isnan(pv):
            print(f"    price_vs_ma200_pct = {pv:.2f}% (informational)")
        if by_tranche:
            parts = ", ".join([f"{k}={v:.2f}" for k, v in sorted(by_tranche.items())])
            print(f"    BUY_NOW breakdown: {parts}")
        print(f"    Total BUY_NOW (summed) = {total_buy_now:.2f} USD")
        print("")

    if blocked:
        print("  Why the other symbols got 0 BUY_NOW allocation:")
        for symbol in blocked:
            reasons = opp_plan.get(symbol, {}).get("blocked_reasons", [])
            if reasons:
                print(f"    - {symbol}: " + "; ".join(reasons))
            else:
                print(f"    - {symbol}: no BUY_NOW allocation")
        print("")

    spent = totals["opp_spent_usd"]
    eff = totals["opp_effective_budget_usd"]
    leftover = max(0.0, eff - spent)
    print(f"  Opportunistic BUY_NOW spent: {spent:.2f} USD")
    print(f"  Opportunistic leftover (not allocated BUY_NOW; may exist as LIMIT_ONLY orders): {leftover:.2f} USD")
    print("")


def print_btc_section(btc_plan: Optional[Dict[str, Any]], btc_tr2_order: Optional[Dict[str, Any]], btc_warn: Optional[str]) -> None:
    print("BTC Opportunistic (SATELLITE):")
    if not btc_plan:
        print("  BTC disabled (no symbols.btc.symbol or no btc section in config).")
        print("")
        return

    if "error" in btc_plan:
        print(f"  ERROR: {btc_plan['error']}")
        print("")
        return

    sym = btc_plan.get("symbol", "BTC")
    eligible = btc_plan.get("eligible", False)

    if not eligible:
        dd = btc_plan.get("drawdown_12m_close_pct")
        pv = btc_plan.get("price_vs_ma200_pct")
        lp = btc_plan.get("last_price")
        lps = btc_plan.get("last_price_source")
        reasons = btc_plan.get("blocked_reasons", [])
        print(f"  {sym}: NOT eligible this run.")
        if lp is not None:
            print(f"    last_price = {lp} ({lps})")
        print(f"    drawdown_12m_close_pct = {dd}")
        print(f"    price_vs_ma200_pct = {pv}")
        if reasons:
            print("    reasons: " + "; ".join(reasons))
        print("")
        return

    dd = btc_plan.get("drawdown_12m_close_pct")
    pv = btc_plan.get("price_vs_ma200_pct")
    lp = btc_plan.get("last_price")
    lps = btc_plan.get("last_price_source")
    base = btc_plan.get("base_budget_usd")
    tr1 = btc_plan.get("tr1_amount_usd")
    intensity = btc_plan.get("drawdown_intensity")
    mult = btc_plan.get("ma200_pacing_multiplier")

    print(f"  {sym}: eligible")
    print(f"    base_budget_usd = {base}")
    if lp is not None:
        print(f"    last_price = {lp} ({lps})")
    print(f"    drawdown_12m_close_pct = {dd}")
    print(f"    price_vs_ma200_pct = {pv}")
    print(f"    TR1 amount (buy now) = {tr1} USD  (intensity={intensity}, ma200_multiplier={mult})")

    if btc_tr2_order:
        lp = btc_tr2_order["limit_price"]
        amt = btc_tr2_order["amount_usd"]
        tif = btc_tr2_order["tif"]
        gtd = btc_tr2_order.get("gtd_until_utc")
        comp = btc_tr2_order["rationale"]["computed"]
        print(f"    TR2 LIMIT: BUY {amt:.2f} USD @ {lp:.2f}  TIF={tif}" + (f"  GTD_until_utc={gtd}" if gtd else ""))
        print(f"      limit_price = min(price_dd_unlock={comp['price_dd_unlock']:.2f}, price_ma200_threshold={comp['price_ma200_threshold']:.2f})")
    else:
        print("    TR2 LIMIT: none generated this run.")

    if btc_warn:
        print(f"    WARNING: {btc_warn}")
    print("")


def print_tranche2_orders_actions(orders: List[Dict[str, Any]], warnings: List[str]) -> None:
    print("Opportunistic Tranche-2 LIMIT Orders (ACTIONS):")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
        print("")
    if not orders:
        print("  No TR2 limit orders generated this run.")
        print("")
        return

    for o in orders:
        sym = o["symbol"]
        amt = o["amount_usd"]
        lp = o["limit_price"]
        tif = o["tif"]
        gtd = o.get("gtd_until_utc")
        comp = o["rationale"].get("computed", {})
        print(f"  {sym}: BUY LMT amount={amt:.2f} USD @ limit_price={lp:.2f}  TIF={tif}" + (f"  GTD_until_utc={gtd}" if gtd else ""))
        if comp:
            print(f"    limit_price = min(price_dd_unlock={comp.get('price_dd_unlock', float('nan')):.2f}, price_ma200_threshold={comp.get('price_ma200_threshold', float('nan')):.2f})")
            print(f"    tranche: {comp.get('tranche')}")
        print(f"    unlock rule: {o['rationale'].get('unlock_rule')}")
        print("")


def print_report(
    cfg: EngineConfig,
    symbols_all_sorted: list,
    snapshot_path: Path,
    snapshot: Dict[str, Any],
    dca_plan: Dict[str, Any],
    opp_plan: Dict[str, Any],
    totals: Dict[str, float],
    tranche2_orders_actions: List[Dict[str, Any]],
    tranche2_warnings_actions: List[str],
    btc_plan: Optional[Dict[str, Any]],
    btc_tr2_order: Optional[Dict[str, Any]],
    btc_warn: Optional[str],
) -> None:
    print("")
    print("=== Monthly Engine Report ===")
    print(f"snapshot_path: {snapshot_path}")
    print(f"generated_at_utc (snapshot.meta): {snapshot.get('meta', {}).get('generated_at_utc')}")
    print(f"mode (snapshot.meta): {snapshot.get('meta', {}).get('mode')}")
    print(f"run_id_utc: {iso_utc_now()}")
    print("")

    headers = ["SYMBOL", "PRICE", "WGT%", "DD12M%", "Px-vs-MA200%", "DCA_USD", "OPP_TR1_USD", "TOTAL_USD", "NOTES"]
    widths = [6, 9, 7, 7, 12, 8, 11, 9, 48]
    print(format_table_row(headers, widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for symbol in symbols_all_sorted:
        px, _px_src = resolve_symbol_last_price_usd(symbol, snapshot)
        wgt = safe_float(dca_plan[symbol]["current_weight_pct"])

        # STRICT: read core metrics only from snapshot (no fallback)
        core = get_symbol_core(snapshot, symbol) or {}
        dd = safe_float(core.get("drawdown_12m_close_pct"), float("nan"))
        high12 = safe_float(core.get("high_12m_close"), float("nan"))
        ma200 = safe_float(core.get("ma200_close"), float("nan"))
        pv = safe_float(core.get("price_vs_ma200_pct"), float("nan"))

        dca_amt = safe_float(dca_plan[symbol]["final_amount_usd"], 0.0)
        opp_tr1_amt = safe_float(opp_plan[symbol].get("capped_amount_usd", 0.0), 0.0)
        total_amt = dca_amt + opp_tr1_amt

        notes: List[str] = []

        # Metrics-missing note (explicit + consistent with strict blocking)
        missing = []
        if math.isnan(dd):
            missing.append("DD12M")
        if math.isnan(high12):
            missing.append("High12M")
        if math.isnan(ma200):
            missing.append("MA200")
        if math.isnan(pv):
            missing.append("Px-vs-MA200")
        if missing:
            notes.append("Metrics missing: " + ", ".join(missing))

        if dca_plan[symbol]["blocked"]:
            notes.append("DCA blocked: " + "; ".join(dca_plan[symbol]["block_reasons"]))
        if not opp_plan[symbol].get("eligible", False):
            notes.append("Opp blocked: " + "; ".join(opp_plan[symbol].get("blocked_reasons", [])))

        row = [
            symbol,
            f"{px:8.2f}" if not math.isnan(px) else "     NaN",
            f"{wgt:5.2f}" if not math.isnan(wgt) else "  NaN",
            f"{dd:5.2f}" if not math.isnan(dd) else "  NaN",
            f"{pv:8.2f}" if not math.isnan(pv) else "     NaN",
            f"{dca_amt:7.2f}",
            f"{opp_tr1_amt:10.2f}",
            f"{total_amt:8.2f}",
            " | ".join(notes)[:widths[-1]],
        ]
        print(format_table_row(row, widths))

    print("")
    print("Totals (ACTIONS):")
    print(f"  DCA spent:              {totals['dca_spent_usd']:.2f} USD")
    print(f"  Opportunistic TR1 spent:{totals['opp_spent_usd']:.2f} USD")
    print(f"  Combined spent:         {totals['total_spent_usd']:.2f} USD")
    if not math.isnan(totals.get("available_cash_usd", float("nan"))):
        print(f"  Available cash (snapshot, informational): {totals['available_cash_usd']:.2f} USD")
    print(f"  Opportunistic effective budget used for split: {totals['opp_effective_budget_usd']:.2f} USD")
    print("")

    print_opportunistic_walkthrough_actions(cfg, snapshot, opp_plan, totals)
    print_tranche2_orders_actions(tranche2_orders_actions, tranche2_warnings_actions)
    print_btc_section(btc_plan, btc_tr2_order, btc_warn)

    print("Methodology (summary):")
    print("  1) Baseline DCA (ACTIONS): allocate monthly DCA budget across symbols by target weights.")
    print("     If a symbol is already at/above its target weight, its DCA is blocked and NOT redistributed.")
    print("  2) Opportunistic (ACTIONS): N-tranches, each uses the same linear intensity mapping by drawdown.")
    print("     BUY_NOW tranches are summed into OPP_TR1_USD (report column preserved).")
    print("     LIMIT_ONLY tranches generate limit orders (TR2/3/...) using limit_price = min(dd_price, ma200_price).")
    print("  3) BTC logic unchanged.")
    print("  4) Snapshot cash is informational only; budgets are not cash-capped here.")
    print("")


def write_output_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


# ----------------------------- main ----------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, default=None, help="Path to a snapshot JSON. If omitted, uses latest in ./snapshots")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json (default: ./config.json)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path (allocations + orders).")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    cfg = load_config(config_path)

    if args.snapshot:
        snapshot_path = Path(args.snapshot).expanduser().resolve()
    else:
        snapshot_path = find_latest_snapshot(Path("snapshots").resolve())

    if snapshot_path is None or (not snapshot_path.exists()):
        raise FileNotFoundError("No snapshot found. Provide --snapshot or put snapshots in ./snapshots")

    snapshot = load_snapshot(snapshot_path)

    # ----- actions
    dca_plan, dca_spent = build_dca_plan(cfg, snapshot)

    # NEW opportunistic actions
    opp_plan, opp_spent, opp_effective_budget, tr2_orders_actions = build_opportunistic_plan(cfg, snapshot)
    tr2_warnings_actions: List[str] = []  # no longer produced in new tranche model

    # ----- BTC (optional)
    btc_plan, btc_tr1_spent, btc_effective_budget = build_btc_opportunistic_plan(cfg, snapshot)
    btc_tr2_order, btc_warn = build_tranche2_limit_order_btc(cfg, snapshot, btc_plan)

    totals = {
        "dca_spent_usd": round_money(dca_spent),
        "opp_spent_usd": round_money(opp_spent),
        "total_spent_usd": round_money(dca_spent + opp_spent),
        "available_cash_usd": get_available_cash_usd(snapshot),
        "opp_effective_budget_usd": round_money(opp_effective_budget),
    }

    def _dd_sort_key(dd: float) -> float:
        # On met les NaN à la fin
        if dd is None or math.isnan(dd):
            return 1e9
        return dd

    def sort_key(symbol: str) -> tuple:
        target = safe_float(cfg.dca.target_weights_pct.get(symbol), float("nan"))
        is_core = (not math.isnan(target)) and (target > 0.0)
        dd = safe_float(opp_plan.get(symbol, {}).get("drawdown_12m_close_pct"), float("nan"))
        # core d'abord => 0, non-core => 1
        return (0 if is_core else 1, _dd_sort_key(dd), symbol)
        
    symbols_all = list(snapshot.get("symbols", {}).keys())  # ou la liste que tu utilises déjà
    symbols_all_sorted = sorted(symbols_all, key=sort_key)


    print_report(
        cfg=cfg,
        symbols_all_sorted=symbols_all_sorted,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        dca_plan=dca_plan,
        opp_plan=opp_plan,
        totals=totals,
        tranche2_orders_actions=tr2_orders_actions,
        tranche2_warnings_actions=tr2_warnings_actions,
        btc_plan=btc_plan,
        btc_tr2_order=btc_tr2_order,
        btc_warn=btc_warn,
    )

    payload = {
        "meta": {
            "generated_at_local": datetime.now().isoformat(timespec="seconds"),
            "generated_at_utc": iso_utc_now(),
            "run_month_utc": month_key_utc(),
            "snapshot_path": str(snapshot_path),
            "config_path": str(config_path),
        },
        "actions": {
            "tracked_symbols": cfg.symbols.tracked_actions,
            "dca": dca_plan,
            "opportunistic_tr1": opp_plan,
            "orders": {
                "opportunistic_tr1_buy_now": [
                    {
                        "symbol": s,
                        "amount_usd": round_money(safe_float(opp_plan.get(s, {}).get("capped_amount_usd"), 0.0))
                    }
                    for s in cfg.symbols.tracked_actions
                    if safe_float(opp_plan.get(s, {}).get("capped_amount_usd"), 0.0) > 0.0
                ],
                "opportunistic_tr2_limit": tr2_orders_actions,
                "warnings": tr2_warnings_actions,
            },
            "totals": totals,
        },
        "btc": {
            "enabled": bool(cfg.btc and cfg.symbols.btc_symbol),
            "symbol": cfg.symbols.btc_symbol,
            "plan": btc_plan,
            "tr1_spent_usd": btc_tr1_spent,
            "effective_budget_usd": btc_effective_budget,
            "tr2_limit_order": btc_tr2_order,
            "warning": btc_warn,
        },
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        write_output_json(out_path, payload)
        print(f"OUTPUT_PATH={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
