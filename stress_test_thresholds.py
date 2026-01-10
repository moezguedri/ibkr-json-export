# stress_test_thresholds.py
# Python 3.10+ (ok on 3.12)
# pip install yfinance pandas numpy

import math
import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------
# Config
# -----------------------------
TICKERS = ["MSFT", "AMZN", "GOOGL", "META"]

START = "2006-01-01"  # will auto-trim to common valid months across all tickers
END = None            # None = today

MONTHLY_DCA_TOTAL = 1000.0
MONTHLY_OPP_BUDGET_DESIRED = 3000.0
ENTRY_THRESHOLDS = [-0.12, -0.15, -0.18]  # compare these

TARGET_WEIGHTS = {
    "MSFT": 0.35,
    "AMZN": 0.25,
    "GOOGL": 0.25,
    "META": 0.15,
}

# Risk locks
MAX_WEIGHT_CAP = 0.20
BLOCK_NVDA_IF_LARGEST = False

# MA200 permissiveness multipliers
MA200_MULT = {
    "MSFT": 1.00,
    "GOOGL": 1.02,
    "AMZN": 1.03,
    "META": 1.00,
    "NVDA": 1.00,
}

# Opportunistic intensity mapping
INTENSITY_FLOOR = 0.30
FULL_DD = -0.25
PER_SYMBOL_MONTHLY_CAP = 1500.0

# -----------------------------
# Metrics helpers
# -----------------------------
def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())

def annualized_vol(returns: pd.Series, periods_per_year=12) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))

def sharpe_ratio(returns: pd.Series, rf_annual=0.0, periods_per_year=12) -> float:
    excess = returns - (rf_annual / periods_per_year)
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((excess.mean() / vol) * math.sqrt(periods_per_year))

def intensity_from_dd(dd: float, entry_dd: float) -> float:
    if dd > entry_dd:
        return 0.0
    if dd <= FULL_DD:
        return 1.0
    progress = (entry_dd - dd) / (entry_dd - FULL_DD)
    progress = max(0.0, min(1.0, progress))
    return INTENSITY_FLOOR + (1.0 - INTENSITY_FLOOR) * progress

# -----------------------------
# Data + indicators
# -----------------------------
def load_data(tickers, start, end=None) -> pd.DataFrame:
    px = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame()

    cols = [c for c in tickers if c in px.columns]
    missing = [c for c in tickers if c not in px.columns]
    if missing:
        raise RuntimeError(f"Missing tickers in data: {missing}. Returned columns: {list(px.columns)}")

    px = px[cols].dropna(how="all")
    return px

def compute_indicators(daily_close: pd.DataFrame) -> dict:
    ma200_daily = daily_close.rolling(200).mean()

    # Month-end: use "ME" (MonthEnd) to avoid FutureWarning about "M"
    m_close = daily_close.resample("ME").last()
    m_ma200 = ma200_daily.resample("ME").last()

    high_12m = m_close.rolling(12).max()
    dd12m = (m_close / high_12m) - 1.0

    return {"m_close": m_close, "m_ma200": m_ma200, "dd12m": dd12m}

def build_monthly_panel(ind: dict) -> pd.DataFrame:
    m_close = ind["m_close"]
    dd12m = ind["dd12m"]
    m_ma200 = ind["m_ma200"]

    # Pandas new stack: when future_stack=True, do NOT pass dropna=
    close_long = m_close.stack(future_stack=True).rename("close").to_frame()
    dd_long    = dd12m.stack(future_stack=True).rename("dd12m").to_frame()
    ma_long    = m_ma200.stack(future_stack=True).rename("ma200").to_frame()

    panel = close_long.join(dd_long, how="inner").join(ma_long, how="inner").dropna()
    panel.index = panel.index.set_names(["date", "ticker"])
    panel = panel.reset_index()

    # Keep only dates where ALL tickers are present (prevents KeyError later)
    counts = panel.groupby("date")["ticker"].nunique()
    full_dates = counts[counts == len(TICKERS)].index
    panel = panel[panel["date"].isin(full_dates)].copy()

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Safety checks
    need = {"date", "ticker", "close", "dd12m", "ma200"}
    assert need.issubset(panel.columns), panel.columns
    assert panel.groupby("date")["ticker"].nunique().min() == len(TICKERS), "Not all tickers present each month"

    return panel

# -----------------------------
# Backtest engine (monthly)
# -----------------------------
def run_backtest(panel: pd.DataFrame, entry_threshold: float):
    dates = panel["date"].drop_duplicates().to_list()

    shares = {t: 0.0 for t in TICKERS}
    cash = 0.0

    equity_curve = []
    cash_curve = []
    opp_active_months = 0

    for d in dates:
        slice_d = panel[panel["date"] == d].set_index("ticker").reindex(TICKERS)
        if slice_d[["close", "dd12m", "ma200"]].isna().any().any():
            continue

        prices = slice_d["close"].to_dict()

        # Monthly inflow
        cash += MONTHLY_DCA_TOTAL + MONTHLY_OPP_BUDGET_DESIRED

        def portfolio_value(curr_cash: float) -> float:
            return curr_cash + sum(shares[t] * prices[t] for t in TICKERS)

        pv = portfolio_value(cash)

        def current_weight(t: str) -> float:
            v = shares[t] * prices[t]
            return (v / pv) if pv > 0 else 0.0

        largest = max(TICKERS, key=lambda t: shares[t] * prices[t])
        nvda_blocked = bool(BLOCK_NVDA_IF_LARGEST and largest == "NVDA")

        # 1) Baseline DCA (no redistribution)
        for t in TICKERS:
            target_w = TARGET_WEIGHTS[t]
            amt = MONTHLY_DCA_TOTAL * target_w

            if current_weight(t) >= target_w:
                continue

            if cash <= 0:
                break

            buy_amt = min(amt, cash)
            shares[t] += buy_amt / prices[t]
            cash -= buy_amt

        # Recompute pv
        pv = portfolio_value(cash)

        # 2) Opportunistic TR1
        opp_budget = min(MONTHLY_OPP_BUDGET_DESIRED, cash)

        eligible = []
        for t in TICKERS:
            dd = float(slice_d.loc[t, "dd12m"])
            if dd <= entry_threshold:
                w = (shares[t] * prices[t]) / pv if pv > 0 else 0.0
                if w >= MAX_WEIGHT_CAP:
                    continue
                if t == "NVDA" and nvda_blocked:
                    continue
                eligible.append(t)

        if eligible and opp_budget > 0:
            opp_active_months += 1
            base_per = opp_budget / len(eligible)

            for t in eligible:
                dd = float(slice_d.loc[t, "dd12m"])
                ma200 = float(slice_d.loc[t, "ma200"])
                price = float(prices[t])

                ma_mult = float(MA200_MULT.get(t, 1.0))
                ma_ok = (price <= ma200 * ma_mult)
                ma_multiplier = 1.0 if ma_ok else 0.5  # slowdown

                inten = intensity_from_dd(dd, entry_threshold)
                raw = base_per * inten * ma_multiplier
                raw = min(raw, PER_SYMBOL_MONTHLY_CAP)

                if raw > 0 and cash > 0:
                    buy_amt = min(raw, cash)
                    shares[t] += buy_amt / price
                    cash -= buy_amt

        pv_end = portfolio_value(cash)
        equity_curve.append((d, pv_end))
        cash_curve.append((d, cash))

    eq = pd.Series({d: v for d, v in equity_curve}).sort_index()
    cash_s = pd.Series({d: v for d, v in cash_curve}).sort_index()

    if len(eq) < 24:
        raise RuntimeError(f"Not enough monthly points: {len(eq)} months")

    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")

    res = {
        "entry_threshold": entry_threshold,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "months": int(len(eq)),
        "final_value": float(eq.iloc[-1]),
        "CAGR": float(cagr),
        "vol_annual": annualized_vol(rets),
        "max_dd": max_drawdown(eq),
        "sharpe": sharpe_ratio(rets),
        "opp_active_months_pct": float((eq.index.to_series().groupby(eq.index).size().shape[0] and 0) or 0),  # placeholder
        "avg_cash": float(cash_s.mean()),
        "min_cash": float(cash_s.min()),
    }

    # Correct opp_active_months_pct (avoid weirdness above; keep simple)
    res["opp_active_months_pct"] = float(opp_active_months / len(eq) * 100.0)

    return res, eq, cash_s

# -----------------------------
# Main
# -----------------------------
def main():
    daily = load_data(TICKERS, START, END)
    ind = compute_indicators(daily)
    panel = build_monthly_panel(ind)

    print(f"Common monthly range (all tickers valid): {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Months: {panel['date'].nunique()}")

    results = []
    for thr in ENTRY_THRESHOLDS:
        res, _, _ = run_backtest(panel, entry_threshold=thr)
        results.append(res)

    summary = pd.DataFrame(results).sort_values("entry_threshold")

    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print("\n=== Summary (monthly backtest) ===")
        print(summary[[
            "entry_threshold", "start", "end", "months",
            "final_value", "CAGR", "vol_annual", "max_dd", "sharpe",
            "opp_active_months_pct", "avg_cash", "min_cash"
        ]].to_string(index=False))

if __name__ == "__main__":
    main()
