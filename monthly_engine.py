#!/usr/bin/env python3
"""
monthly_engine.py

Reads an IBKR+market snapshot JSON produced by ibkr_to_json.py and prints:
- DCA (baseline) allocations
- Opportunistic allocations (drawdown + MA200 pacing, no binary triggers)
- NEW: Opportunistic TRANCHE-2 LIMIT orders (statefulness without saving state)

Design goals:
- Minimal deps (stdlib only)
- Explicit / readable names (no "dd", no cryptic abbreviations)
- Configuration-driven (config.json)
- Deterministic + explainable output

USAGE
-----
# Use latest snapshot in ./snapshots and ./config.json
python monthly_engine.py

# Explicit snapshot path
python monthly_engine.py --snapshot snapshots/portfolio_snapshot_YYYYMMDD_HHMM.json

# Explicit config path + write output allocations JSON
python monthly_engine.py --config config.json --output allocations.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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
class Tranche2LimitConfig:
    enabled: bool
    drawdown_unlock_threshold_pct: float  # e.g. -20
    # MA200 threshold multiplier per symbol: price <= MA200 * multiplier
    ma200_threshold_multipliers: Dict[str, float]  # e.g. {"GOOGL": 1.02, "AMZN": 1.03}
    tif: str  # "GTD" or "GTC" (recommend GTD)
    gtd_end_of_month_utc_hour: int
    gtd_end_of_month_utc_minute: int


@dataclass(frozen=True)
class OpportunisticConfig:
    monthly_budget_usd: float
    allow_exceed_available_cash: bool  # if False, cap by TotalCashValue_USD
    drawdown_entry_threshold_pct: float  # e.g. -15
    buy_pct_at_entry: float  # e.g. 0.30 -> 30% at entry threshold
    drawdown_full_allocation_pct: float  # e.g. -25 -> 100%

    # Max position weight cap (default + optional per-symbol overrides)
    max_position_weight_pct_default: float  # e.g. 20
    max_position_weight_pct_overrides: Dict[str, float]  # e.g. {"MSFT": 35}

    per_symbol_monthly_cap_usd: float  # e.g. 1500
    forbid_nvda_when_largest_holding: bool

    # MA200 pacing: ONLY used to slow down when price is far above MA200.
    ma200_penalty_start_pct: float  # e.g. 0 -> starts penalizing above MA200
    ma200_penalty_full_pct: float   # e.g. 10 -> full penalty by +10% above MA200
    ma200_max_penalty: float        # e.g. 0.50 -> at most cut buy by 50%

    tranche2_limit: Tranche2LimitConfig


@dataclass(frozen=True)
class EngineConfig:
    tracked_symbols: List[str]
    ignored_symbols: List[str]
    dca: DcaConfig
    opportunistic: OpportunisticConfig


def load_config(config_path: Path) -> EngineConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    tracked_symbols = list(raw["symbols"]["tracked"])
    ignored_symbols = list(raw.get("symbols", {}).get("ignored", []))

    dca_raw = raw["dca"]
    dca = DcaConfig(
        monthly_budget_usd=float(dca_raw["monthly_budget_usd"]),
        target_weights_pct={k: float(v) for k, v in dca_raw["target_weights_pct"].items()},
        block_if_current_weight_pct_ge_target=bool(dca_raw.get("block_if_current_weight_pct_ge_target", True)),
        block_tolerance_pct=float(dca_raw.get("block_tolerance_pct", 0.0)),
    )

    opp_raw = raw["opportunistic"]

    tr2_raw = opp_raw.get("tranche2_limit", {})
    tranche2_limit = Tranche2LimitConfig(
        enabled=bool(tr2_raw.get("enabled", True)),
        drawdown_unlock_threshold_pct=float(tr2_raw.get("drawdown_unlock_threshold_pct", -20.0)),
        ma200_threshold_multipliers={k.upper(): float(v) for k, v in tr2_raw.get("ma200_threshold_multipliers", {}).items()},
        tif=str(tr2_raw.get("tif", "GTD")).upper(),
        gtd_end_of_month_utc_hour=int(tr2_raw.get("gtd_end_of_month_utc_hour", 20)),
        gtd_end_of_month_utc_minute=int(tr2_raw.get("gtd_end_of_month_utc_minute", 0)),
    )

    opportunistic = OpportunisticConfig(
        monthly_budget_usd=float(opp_raw["monthly_budget_usd"]),
        allow_exceed_available_cash=bool(opp_raw.get("allow_exceed_available_cash", False)),
        drawdown_entry_threshold_pct=float(opp_raw["drawdown_entry_threshold_pct"]),
        buy_pct_at_entry=float(opp_raw["buy_pct_at_entry"]),
        drawdown_full_allocation_pct=float(opp_raw["drawdown_full_allocation_pct"]),

        max_position_weight_pct_default=float(opp_raw.get("max_position_weight_pct_default", opp_raw.get("max_position_weight_pct", 20.0))),
        max_position_weight_pct_overrides={k.upper(): float(v) for k, v in opp_raw.get("max_position_weight_pct_overrides", {}).items()},

        per_symbol_monthly_cap_usd=float(opp_raw["per_symbol_monthly_cap_usd"]),
        forbid_nvda_when_largest_holding=bool(opp_raw.get("forbid_nvda_when_largest_holding", True)),
        ma200_penalty_start_pct=float(opp_raw.get("ma200_penalty_start_pct", 0.0)),
        ma200_penalty_full_pct=float(opp_raw.get("ma200_penalty_full_pct", 10.0)),
        ma200_max_penalty=float(opp_raw.get("ma200_max_penalty", 0.50)),

        tranche2_limit=tranche2_limit,
    )

    return EngineConfig(
        tracked_symbols=tracked_symbols,
        ignored_symbols=ignored_symbols,
        dca=dca,
        opportunistic=opportunistic,
    )


# ----------------------------- core math ----------------------------- #

def compute_drawdown_intensity(
    drawdown_12m_close_pct: float,
    entry_threshold_pct: float,
    buy_pct_at_entry: float,
    full_allocation_pct: float,
) -> float:
    """
    Map drawdown to a [0..1] intensity.

    - If drawdown is above entry threshold (e.g. -10%), intensity = 0 (not eligible).
    - At entry threshold (e.g. -15%), intensity = buy_pct_at_entry (e.g. 0.30).
    - At full allocation drawdown (e.g. -25%), intensity = 1.0
    - Linearly interpolated between entry and full.

    Note:
      drawdown_12m_close_pct is negative (e.g. -18.5).
      entry_threshold_pct and full_allocation_pct are negative too.
    """
    if math.isnan(drawdown_12m_close_pct):
        return 0.0

    # Not eligible -> 0
    if drawdown_12m_close_pct > entry_threshold_pct:
        return 0.0

    # Guard against bad config
    if full_allocation_pct >= entry_threshold_pct:
        # e.g. -20 >= -15 is wrong
        return buy_pct_at_entry

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

    - If price is at/below penalty_start_pct (often 0%), multiplier = 1.0 (no slowdown).
    - If price is above, linearly reduce to (1 - max_penalty) at penalty_full_pct (e.g. +10%).
    - Never reduces below (1 - max_penalty).

    Important: This is NOT a trigger; it only throttles buys when price is "too far" above MA200.
    """
    if math.isnan(price_vs_ma200_pct):
        return 1.0

    if price_vs_ma200_pct <= penalty_start_pct:
        return 1.0

    if penalty_full_pct <= penalty_start_pct:
        return 1.0 - clamp(max_penalty, 0.0, 0.95)

    if price_vs_ma200_pct >= penalty_full_pct:
        return 1.0 - clamp(max_penalty, 0.0, 0.95)

    frac = (price_vs_ma200_pct - penalty_start_pct) / (penalty_full_pct - penalty_start_pct)
    penalty = clamp(max_penalty, 0.0, 0.95) * frac
    return clamp(1.0 - penalty, 0.05, 1.0)


# ----------------------------- decision engine ----------------------------- #

def load_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def get_portfolio_weight_pct(snapshot: Dict[str, Any], symbol: str) -> float:
    weights = snapshot.get("portfolio", {}).get("derived", {}).get("tracked_weights_pct", {})
    return safe_float(weights.get(symbol), default=float("nan"))


def get_available_cash_usd(snapshot: Dict[str, Any]) -> float:
    acct = snapshot.get("portfolio", {}).get("account_summary", {})
    # TotalCashValue_USD exists in your snapshots and is the most direct value.
    return safe_float(acct.get("TotalCashValue_USD"), default=float("nan"))


def get_largest_holding_symbol(snapshot: Dict[str, Any]) -> Optional[str]:
    return snapshot.get("portfolio", {}).get("derived", {}).get("largest_holding_stock_symbol")


def get_symbol_core(snapshot: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    return snapshot.get("symbols", {}).get(symbol, {}).get("core_metrics", {}) or {}


def build_dca_plan(cfg: EngineConfig, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Baseline DCA: distribute monthly budget by target weights.
    Rule: if current portfolio weight >= target weight (with tolerance), block that symbol's DCA.
    IMPORTANT: blocked amount is NOT redistributed.
    """
    budget = cfg.dca.monthly_budget_usd
    planned: Dict[str, Any] = {}
    spent = 0.0

    for symbol in cfg.tracked_symbols:
        target_pct = safe_float(cfg.dca.target_weights_pct.get(symbol), default=float("nan"))
        planned_amount = budget * (target_pct / 100.0) if not math.isnan(target_pct) else float("nan")

        current_weight = get_portfolio_weight_pct(snapshot, symbol)
        blocked = False
        reasons: List[str] = []

        if cfg.dca.block_if_current_weight_pct_ge_target and not math.isnan(current_weight) and not math.isnan(target_pct):
            if current_weight >= (target_pct - cfg.dca.block_tolerance_pct):
                blocked = True
                reasons.append(f"current_weight_pct {current_weight:.2f} >= target_weight_pct {target_pct:.2f}")

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
        spent += 0.0 if blocked else safe_float(planned_amount, 0.0)

    spent = round_money(spent)
    return planned, spent


def build_opportunistic_plan(cfg: EngineConfig, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
    """
    Opportunistic:
      1) Eligible symbols: drawdown <= entry_threshold AND not blocked by risk rules.
      2) Monthly budget is split equally across eligible symbols (N_actives).
      3) Per symbol, buy intensity is:
           intensity(drawdown) * pacing_multiplier(price_vs_ma200)
      4) Per-symbol monthly cap applies.
      5) Leftover remains as cash (not reallocated).
    """
    opp = cfg.opportunistic

    desired_budget = opp.monthly_budget_usd
    available_cash = get_available_cash_usd(snapshot)
    if (not opp.allow_exceed_available_cash) and (not math.isnan(available_cash)):
        effective_budget = min(desired_budget, available_cash)
    else:
        effective_budget = desired_budget
    effective_budget = max(0.0, float(effective_budget))

    largest = get_largest_holding_symbol(snapshot)
    weights = snapshot.get("portfolio", {}).get("derived", {}).get("tracked_weights_pct", {})

    eligible: List[str] = []
    blocked_reasons: Dict[str, List[str]] = {}

    for symbol in cfg.tracked_symbols:
        reasons: List[str] = []

        current_weight = safe_float(weights.get(symbol), default=float("nan"))
        max_allowed_weight = opp.max_position_weight_pct_overrides.get(symbol.upper(), opp.max_position_weight_pct_default)
        if not math.isnan(current_weight) and current_weight >= max_allowed_weight:
            reasons.append(f"position_cap: current_weight_pct {current_weight:.2f} >= max_allowed_weight_pct {max_allowed_weight:.2f}")

        if opp.forbid_nvda_when_largest_holding and symbol.upper() == "NVDA" and (largest or "").upper() == "NVDA":
            reasons.append("NVDA_blocked: NVDA is currently the largest holding")

        core = get_symbol_core(snapshot, symbol)
        drawdown = safe_float(core.get("drawdown_12m_close_pct"), default=float("nan"))
        if math.isnan(drawdown) or drawdown > opp.drawdown_entry_threshold_pct:
            reasons.append(f"not_eligible: drawdown_12m_close_pct {drawdown if not math.isnan(drawdown) else 'NaN'} > entry_threshold {opp.drawdown_entry_threshold_pct}")

        if reasons:
            blocked_reasons[symbol] = reasons
        else:
            eligible.append(symbol)

    plan: Dict[str, Any] = {}
    spent = 0.0

    if not eligible or effective_budget <= 0:
        for symbol in cfg.tracked_symbols:
            plan[symbol] = {
                "eligible": False,
                "base_budget_usd": 0.0,
                "drawdown_intensity": 0.0,
                "ma200_pacing_multiplier": 1.0,
                "raw_amount_usd": 0.0,
                "capped_amount_usd": 0.0,
                "blocked_reasons": blocked_reasons.get(symbol, ["no_eligible_symbols_or_zero_budget"]),
            }
        return plan, 0.0, effective_budget

    base_budget = effective_budget / len(eligible)

    for symbol in cfg.tracked_symbols:
        core = get_symbol_core(snapshot, symbol)
        drawdown = safe_float(core.get("drawdown_12m_close_pct"), default=float("nan"))
        price_vs_ma200 = safe_float(core.get("price_vs_ma200_pct"), default=float("nan"))
        current_weight = get_portfolio_weight_pct(snapshot, symbol)

        if symbol not in eligible:
            plan[symbol] = {
                "eligible": False,
                "current_weight_pct": current_weight,
                "drawdown_12m_close_pct": drawdown,
                "price_vs_ma200_pct": price_vs_ma200,
                "base_budget_usd": round_money(base_budget),
                "drawdown_intensity": 0.0,
                "ma200_pacing_multiplier": 1.0,
                "raw_amount_usd": 0.0,
                "capped_amount_usd": 0.0,
                "blocked_reasons": blocked_reasons.get(symbol, ["not_eligible"]),
            }
            continue

        drawdown_intensity = compute_drawdown_intensity(
            drawdown_12m_close_pct=drawdown,
            entry_threshold_pct=opp.drawdown_entry_threshold_pct,
            buy_pct_at_entry=opp.buy_pct_at_entry,
            full_allocation_pct=opp.drawdown_full_allocation_pct,
        )

        ma200_multiplier = compute_ma200_pacing_multiplier(
            price_vs_ma200_pct=price_vs_ma200,
            penalty_start_pct=opp.ma200_penalty_start_pct,
            penalty_full_pct=opp.ma200_penalty_full_pct,
            max_penalty=opp.ma200_max_penalty,
        )

        raw_amount = base_budget * drawdown_intensity * ma200_multiplier
        capped_amount = min(raw_amount, opp.per_symbol_monthly_cap_usd)

        raw_amount = round_money(raw_amount)
        capped_amount = round_money(capped_amount)

        plan[symbol] = {
            "eligible": True,
            "current_weight_pct": current_weight,
            "drawdown_12m_close_pct": drawdown,
            "price_vs_ma200_pct": price_vs_ma200,
            "base_budget_usd": round_money(base_budget),
            "drawdown_intensity": round(drawdown_intensity, 4),
            "ma200_pacing_multiplier": round(ma200_multiplier, 4),
            "raw_amount_usd": raw_amount,
            "capped_amount_usd": capped_amount,
            "blocked_reasons": [],
        }

        spent += safe_float(capped_amount, 0.0)

    spent = round_money(spent)
    return plan, spent, effective_budget


def build_tranche2_limit_orders(
    cfg: EngineConfig,
    snapshot: Dict[str, Any],
    opp_plan: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    TRANCHE 2 design (your chosen variant):
      - TR1 is bought "now" (from opp_plan capped_amount_usd).
      - TR2 is a LIMIT order only.
      - TR2 should only trigger if BOTH are true:
          (a) drawdown_12m_close_pct <= unlock_threshold (e.g. -20%)
          (b) price <= ma200_close * multiplier(symbol)
        We encode (a)&(b) as a single limit price:
          limit_price = min(high_12m_close*(1+unlock/100), ma200_close*multiplier)
    Amount:
      - remaining per-symbol budget = base_budget - tranche1_amount
      - capped by per_symbol_monthly_cap_usd
      - if remaining <= 0 => no TR2 order
    """
    opp = cfg.opportunistic
    tr2 = opp.tranche2_limit

    if not tr2.enabled:
        return [], []

    warnings: List[str] = []
    orders: List[Dict[str, Any]] = []

    # Determine base_budget from any eligible symbol (present in plan), else 0
    eligible_symbols = [s for s in cfg.tracked_symbols if opp_plan.get(s, {}).get("eligible")]
    base_budget = safe_float(opp_plan[eligible_symbols[0]].get("base_budget_usd"), default=0.0) if eligible_symbols else 0.0

    # TIF formatting for reporting
    if tr2.tif == "GTD":
        # end of month UTC time
        # NOTE: we intentionally keep this simple and only for output.
        # Your order sender can convert it to IBKR's GTD format.
        from datetime import timedelta  # local import to keep header clean
        gtd_until_utc = compute_gtd_end_of_month_utc(tr2.gtd_end_of_month_utc_hour, tr2.gtd_end_of_month_utc_minute)
    else:
        gtd_until_utc = None

    for symbol in cfg.tracked_symbols:
        rec = opp_plan.get(symbol, {})
        tranche1_amount = safe_float(rec.get("capped_amount_usd"), default=0.0)

        # Only generate TR2 if TR1 is actually planned to buy now (>0)
        if tranche1_amount <= 0.0:
            continue

        # Remaining budget for this symbol (by construction base_budget is per eligible symbol)
        remaining = max(0.0, base_budget - tranche1_amount)
        tranche2_amount = min(remaining, opp.per_symbol_monthly_cap_usd)

        tranche2_amount = round_money(tranche2_amount)
        if tranche2_amount <= 0.0:
            continue

        core = get_symbol_core(snapshot, symbol)
        high_12m_close = safe_float(core.get("high_12m_close"), default=float("nan"))
        ma200_close = safe_float(core.get("ma200_close"), default=float("nan"))

        if math.isnan(high_12m_close) or math.isnan(ma200_close):
            warnings.append(
                f"{symbol}: cannot compute TR2 limit (need high_12m_close and ma200_close). "
                f"Got high_12m_close={core.get('high_12m_close')}, ma200_close={core.get('ma200_close')}."
            )
            continue

        # (a) drawdown unlock threshold price
        unlock_dd = tr2.drawdown_unlock_threshold_pct  # e.g. -20
        price_dd_unlock = high_12m_close * (1.0 + unlock_dd / 100.0)  # -20 => *0.80

        # (b) ma200 threshold price
        mult = tr2.ma200_threshold_multipliers.get(symbol.upper(), 1.0)
        price_ma2 = ma200_close * mult

        # Encode (a)&(b) by taking the stricter (lower) price
        limit_price = min(price_dd_unlock, price_ma2)
        limit_price = round_money(limit_price)

        rationale = {
            "unlock_rule": f"DD12M <= {unlock_dd:.2f}% AND price <= MA200*multiplier",
            "computed": {
                "high_12m_close": round_money(high_12m_close),
                "ma200_close": round_money(ma200_close),
                "ma200_multiplier": mult,
                "price_dd_unlock": round_money(price_dd_unlock),
                "price_ma200_threshold": round_money(price_ma2),
                "limit_price": limit_price,
            }
        }

        orders.append({
            "symbol": symbol,
            "side": "BUY",
            "order_type": "LMT",
            "amount_usd": tranche2_amount,
            "limit_price": limit_price,
            "tif": tr2.tif,
            "gtd_until_utc": gtd_until_utc,
            "tag": f"OPP_TR2_{month_key_utc()}",
            "rationale": rationale,
        })

    return orders, warnings


# ----------------------------- reporting ----------------------------- #

def format_table_row(cols: List[str], widths: List[int]) -> str:
    padded = []
    for c, w in zip(cols, widths):
        c = "" if c is None else str(c)
        padded.append(c[:w].ljust(w))
    return " | ".join(padded)


def print_opportunistic_walkthrough(cfg: EngineConfig, snapshot: Dict[str, Any], opp_plan: Dict[str, Any], totals: Dict[str, float]) -> None:
    opp = cfg.opportunistic
    print("Opportunistic Decision Walkthrough:")
    print(f"  Opportunistic budget (desired): {opp.monthly_budget_usd:.2f} USD")
    if not math.isnan(totals.get("available_cash_usd", float("nan"))):
        print(f"  Available cash (USD): {totals['available_cash_usd']:.2f} USD")
    print(f"  Effective budget used for split: {totals['opp_effective_budget_usd']:.2f} USD")
    print("")
    print("  Eligibility rule:")
    print(f"    - drawdown_12m_close_pct <= {opp.drawdown_entry_threshold_pct:.2f}%")
    print("    - and not blocked by risk locks (max weight cap, NVDA lock if enabled)")
    print("")

    eligible = [s for s in cfg.tracked_symbols if opp_plan.get(s, {}).get("eligible")]
    blocked = [s for s in cfg.tracked_symbols if s not in eligible]

    if not eligible:
        print("  Result: no eligible symbols this month → opportunistic allocation is 0 for all symbols.")
        print("")
        return

    base_budget = safe_float(opp_plan[eligible[0]].get("base_budget_usd"), default=0.0)
    print(f"  Eligible symbols ({len(eligible)}): {', '.join(eligible)}")
    print(f"  Base budget per eligible symbol (effective_budget / N_actives): {base_budget:.2f} USD")
    print("")

    for symbol in eligible:
        rec = opp_plan[symbol]
        dd = safe_float(rec.get("drawdown_12m_close_pct"))
        pv = safe_float(rec.get("price_vs_ma200_pct"))
        intensity = safe_float(rec.get("drawdown_intensity"))
        mult = safe_float(rec.get("ma200_pacing_multiplier"))
        raw_amt = safe_float(rec.get("raw_amount_usd"))
        capped_amt = safe_float(rec.get("capped_amount_usd"))
        cap = opp.per_symbol_monthly_cap_usd

        print(f"  {symbol}:")
        print(f"    drawdown_12m_close_pct = {dd:.2f}% → eligible (<= {opp.drawdown_entry_threshold_pct:.2f}%)")
        if not math.isnan(pv):
            if pv <= opp.ma200_penalty_start_pct:
                print(f"    price_vs_ma200_pct = {pv:.2f}% (<= {opp.ma200_penalty_start_pct:.2f}%) → no MA200 slowdown (multiplier = 1.0)")
            else:
                print(f"    price_vs_ma200_pct = {pv:.2f}% (> {opp.ma200_penalty_start_pct:.2f}%) → MA200 pacing multiplier = {mult:.4f}")
        else:
            print(f"    price_vs_ma200_pct = NaN → MA200 pacing multiplier = {mult:.4f}")

        entry = opp.drawdown_entry_threshold_pct
        full = opp.drawdown_full_allocation_pct
        buy_at_entry = opp.buy_pct_at_entry

        if (not math.isnan(dd)) and (full < entry):
            numer = entry - dd
            denom = entry - full
            progress = clamp(numer / denom, 0.0, 1.0)
            print("    Linear drawdown intensity mapping:")
            print(f"      progress = ({entry:.2f} - ({dd:.2f})) / ({entry:.2f} - ({full:.2f})) = {numer:.2f} / {denom:.2f} = {progress:.4f}")
            print(f"      intensity = {buy_at_entry:.2f} + (1 - {buy_at_entry:.2f}) * progress = {intensity:.4f}")
        else:
            print(f"    Linear drawdown intensity mapping: intensity = {intensity:.4f} (config guard / NaN)")

        print(f"    raw_amount = base_budget * intensity * ma200_multiplier")
        print(f"              ≈ {base_budget:.2f} * {intensity:.4f} * {mult:.4f} = {raw_amt:.2f} USD")

        if capped_amt < raw_amt:
            print(f"    per_symbol_monthly_cap_usd = {cap:.2f} → capped to {capped_amt:.2f} USD")
        else:
            print(f"    per_symbol_monthly_cap_usd = {cap:.2f} → no cap applied ({capped_amt:.2f} USD)")

        print("")

    if blocked:
        print("  Why the other symbols got 0 opportunistic allocation:")
        for symbol in blocked:
            reasons = opp_plan.get(symbol, {}).get("blocked_reasons", [])
            if reasons:
                print(f"    - {symbol}: " + "; ".join(reasons))
            else:
                print(f"    - {symbol}: not eligible")
        print("")

    spent = totals["opp_spent_usd"]
    eff = totals["opp_effective_budget_usd"]
    leftover = max(0.0, eff - spent)
    print(f"  Opportunistic spent: {spent:.2f} USD")
    print(f"  Opportunistic leftover (stays as cash by design): {leftover:.2f} USD")
    print("")


def print_tranche2_orders(orders: List[Dict[str, Any]], warnings: List[str]) -> None:
    print("Opportunistic Tranche-2 LIMIT Orders (stateful without saving state):")
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
        comp = o["rationale"]["computed"]
        print(f"  {sym}: BUY LMT amount={amt:.2f} USD @ limit_price={lp:.2f}  TIF={tif}" + (f"  GTD_until_utc={gtd}" if gtd else ""))
        print(f"    limit_price = min(price_dd_unlock={comp['price_dd_unlock']:.2f}, price_ma200_threshold={comp['price_ma200_threshold']:.2f})")
        print(f"    unlock rule: {o['rationale']['unlock_rule']}")
        print("")


def print_report(
    cfg: EngineConfig,
    snapshot_path: Path,
    snapshot: Dict[str, Any],
    dca_plan: Dict[str, Any],
    opp_plan: Dict[str, Any],
    totals: Dict[str, float],
    tranche2_orders: List[Dict[str, Any]],
    tranche2_warnings: List[str],
) -> None:
    print("")
    print("=== Monthly Engine Report ===")
    print(f"snapshot_path: {snapshot_path}")
    print(f"generated_at_utc (snapshot.meta): {snapshot.get('meta', {}).get('generated_at_utc')}")
    print(f"mode (snapshot.meta): {snapshot.get('meta', {}).get('mode')}")
    print(f"run_id_utc: {iso_utc_now()}")
    print("")

    headers = ["SYMBOL", "WGT%", "DD12M%", "Px-vs-MA200%", "DCA_USD", "OPP_TR1_USD", "TOTAL_USD", "NOTES"]
    widths = [6, 7, 7, 12, 8, 11, 9, 40]
    print(format_table_row(headers, widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for symbol in cfg.tracked_symbols:
        wgt = safe_float(dca_plan[symbol]["current_weight_pct"])
        dd = safe_float(opp_plan[symbol].get("drawdown_12m_close_pct", float("nan")))
        pv = safe_float(opp_plan[symbol].get("price_vs_ma200_pct", float("nan")))
        dca_amt = safe_float(dca_plan[symbol]["final_amount_usd"], 0.0)
        opp_tr1_amt = safe_float(opp_plan[symbol].get("capped_amount_usd", 0.0), 0.0)
        total_amt = dca_amt + opp_tr1_amt

        notes = []
        if dca_plan[symbol]["blocked"]:
            notes.append("DCA blocked: " + "; ".join(dca_plan[symbol]["block_reasons"]))
        if not opp_plan[symbol].get("eligible", False):
            notes.append("Opp blocked: " + "; ".join(opp_plan[symbol].get("blocked_reasons", [])))

        row = [
            symbol,
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
    print("Totals:")
    print(f"  DCA spent:           {totals['dca_spent_usd']:.2f} USD")
    print(f"  Opportunistic TR1 spent: {totals['opp_spent_usd']:.2f} USD")
    print(f"  Combined spent:      {totals['total_spent_usd']:.2f} USD")
    if not math.isnan(totals.get("available_cash_usd", float("nan"))):
        print(f"  Available cash (USD): {totals['available_cash_usd']:.2f} USD")
    print(f"  Opportunistic effective budget used for split: {totals['opp_effective_budget_usd']:.2f} USD")
    print("")

    print_opportunistic_walkthrough(cfg, snapshot, opp_plan, totals)
    print_tranche2_orders(tranche2_orders, tranche2_warnings)

    print("Methodology (English):")
    print("  1) Baseline DCA: allocate the monthly DCA budget across symbols by target weights.")
    print("     If a symbol is already at/above its target weight, its DCA is blocked and NOT redistributed.")
    print("  2) Opportunistic TR1: only symbols with drawdown_12m_close_pct <= entry_threshold are considered.")
    print("     Eligible symbols share the opportunistic budget equally (N_actives).")
    print("  3) TR1 intensity is linear with drawdown, paced by MA200 distance (not a binary trigger).")
    print("  4) Opportunistic TR2 (LIMIT): generated only when TR1 amount > 0.")
    print("     It encodes: DD12M <= unlock_threshold AND price <= MA200*multiplier as a single limit price:")
    print("       limit_price = min(high_12m_close*(1+unlock/100), ma200_close*multiplier)")
    print("     This adds statefulness without saving state files.")
    print("  5) Risk locks: max weight cap (default + per-symbol overrides) and NVDA blocked when largest holding (optional).")
    print("  6) Any leftover budget remains cash (no forced redistribution).")
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

    dca_plan, dca_spent = build_dca_plan(cfg, snapshot)
    opp_plan, opp_spent, opp_effective_budget = build_opportunistic_plan(cfg, snapshot)

    tranche2_orders, tranche2_warnings = build_tranche2_limit_orders(cfg, snapshot, opp_plan)

    totals = {
        "dca_spent_usd": round_money(dca_spent),
        "opp_spent_usd": round_money(opp_spent),
        "total_spent_usd": round_money(dca_spent + opp_spent),
        "available_cash_usd": get_available_cash_usd(snapshot),
        "opp_effective_budget_usd": round_money(opp_effective_budget),
    }

    print_report(cfg, snapshot_path, snapshot, dca_plan, opp_plan, totals, tranche2_orders, tranche2_warnings)

    payload = {
        "meta": {
            "generated_at_local": datetime.now().isoformat(timespec="seconds"),
            "generated_at_utc": iso_utc_now(),
            "run_month_utc": month_key_utc(),
            "snapshot_path": str(snapshot_path),
            "config_path": str(config_path),
        },
        "dca": dca_plan,
        "opportunistic": opp_plan,
        "orders": {
            "opportunistic_tr1_buy_now": [
                {"symbol": s, "amount_usd": round_money(safe_float(opp_plan.get(s, {}).get("capped_amount_usd"), 0.0))}
                for s in cfg.tracked_symbols
                if safe_float(opp_plan.get(s, {}).get("capped_amount_usd"), 0.0) > 0.0
            ],
            "opportunistic_tr2_limit": tranche2_orders,
            "warnings": tranche2_warnings,
        },
        "totals": totals,
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        write_output_json(out_path, payload)
        print(f"OUTPUT_PATH={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
