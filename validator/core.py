"""validator — a small, dependency-light backtest validation engine.

Purpose
-------
Demonstrate, with reproducible synthetic data, the *mechanistic* checks that separate a
trustworthy backtest from an overstated one:

  - lag sensitivity         (shift the signal 1 bar -> does the edge vanish?)
  - period expansion        (is a low-frequency column reused at bar level?)
  - fill-timing sensitivity (shift fills 1 bar -> does profit collapse?)
  - randomized control      (permutation test: real edge vs bull-market beta?)
  - return independence     (Ljung-Box + ACF -> how many independent trades really?)

This is a REFERENCE implementation used to showcase methodology. It is deliberately generic:
a strategy is any pure function ``bt(df) -> {"pnl": float, "trades": int}`` that reads the
columns it needs from the frame and never mutates it.

Python >= 3.9. Dependencies: numpy, pandas only.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

# --- tuning constants (documented, deterministic) -------------------------------
EXPANSION_HOURS = 2.0      # longest constant run >= this => SUSPECT
SHRINK_RATIO = 0.50        # lagged pnl < 50% of baseline => edge came from timing
FLIP_MIN_RATIO = 0.30      # sign flip with |lagged| >= 30% of |baseline| => FAIL
RC_N_SHUFFLES = 200        # randomized-control shuffles
RC_ALPHA = 0.05
AC_MAX_LAG = 10
AC_MIN_N = 20

_BacktestFn = Callable[[pd.DataFrame], Dict]


def _chi2_sf(x: float, df: int) -> float:
    """chi2 survival P(X > x) — Wilson-Hilferty normal approximation (no scipy needed)."""
    if x <= 0.0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _pnl(res: Dict) -> float:
    return float(res.get("pnl", res.get("total_pnl", 0.0)))


def _trades(res: Dict) -> int:
    return int(res.get("trades", 0))


# ---------------------------------------------------------------------------
# 1. Lag sensitivity — future-function / look-ahead detector
# ---------------------------------------------------------------------------

def lag_sensitivity(df: pd.DataFrame, col: str, bt: _BacktestFn,
                    lag: int = 1, verbose: bool = False) -> Dict:
    """Re-run the strategy with the state column shifted `lag` bars later.

    If the edge depended on information that only becomes available one bar later,
    the lagged run loses it (collapse or flip).
    """
    base = bt(df.copy())
    base_pnl = _pnl(base)
    lagged_df = df.copy()
    lagged_df[col] = df[col].shift(lag).fillna(df[col].iloc[0])  # carry first value
    lag_res = bt(lagged_df)
    lag_pnl = _pnl(lag_res)
    base_tr, lag_tr = _trades(base), _trades(lag_res)

    if base_tr == 0 or abs(base_pnl) < 1e-12:
        verdict = "INSUFFICIENT"
    else:
        ratio = abs(lag_pnl) / abs(base_pnl)
        flipped = (lag_pnl < 0) != (base_pnl < 0)
        if flipped and ratio >= FLIP_MIN_RATIO:
            verdict = "FAIL"          # direction reversed with material magnitude
        elif ratio < SHRINK_RATIO:
            verdict = "FAIL"          # >50% of profit came from one-bar timing
        elif abs(lag_pnl - base_pnl) / abs(base_pnl) > 0.30:
            verdict = "WARN"
        else:
            verdict = "PASS"
    rep = {"verdict": verdict, "base_pnl": base_pnl, "lagged_pnl": lag_pnl,
           "base_trades": base_tr, "lagged_trades": lag_tr, "lag_bars": lag}
    if verbose:
        print(f"lag_sensitivity({col}, +{lag}bar): {verdict}  "
              f"pnl {base_pnl:,.1f} -> {lag_pnl:,.1f}")
    return rep


# ---------------------------------------------------------------------------
# 2. Period expansion — low-frequency column reused at bar level
# ---------------------------------------------------------------------------

def period_expansion(df: pd.DataFrame, col: str, bar_seconds: int = 300,
                     verbose: bool = False) -> Dict:
    """Longest run of identical consecutive values, expressed in hours.

    A daily signal repeated over every intraday bar forms a multi-hour constant run =>
    SUSPECT (implicit look-ahead) unless the caller confirms how the column was shifted.
    """
    vals = df[col].to_numpy()
    longest = change_points = 0
    prev = None
    run = 0
    for v in vals:
        same = (prev is not None) and (pd.isna(v) and pd.isna(prev) or v == prev)
        if same:
            run += 1
        else:
            run = 1
            change_points += 1
        longest = max(longest, run)
        prev = v
    hours = longest * bar_seconds / 3600.0
    suspect = hours >= EXPANSION_HOURS
    rep = {"longest_run_bars": longest, "longest_run_hours": round(hours, 2),
           "change_points": max(change_points, 1), "verdict": "SUSPECT" if suspect else "OK"}
    if verbose:
        print(f"period_expansion({col}): {rep['verdict']}  longest run {longest} bars "
              f"({rep['longest_run_hours']}h @ {bar_seconds}s)")
    return rep


# ---------------------------------------------------------------------------
# 3. Fill-timing sensitivity — execution-level look-ahead
# ---------------------------------------------------------------------------

def fill_timing_sensitivity(df: pd.DataFrame, bt: _BacktestFn,
                            price_cols: Sequence[str] = ("open",),
                            lag: int = 1, verbose: bool = False) -> Dict:
    """Shift every fill price one bar later and re-run.

    Catches strategies whose profit only exists because they "know the close and fill at
    the open" — even when the signal column itself is clean.
    """
    base_pnl = _pnl(bt(df.copy()))
    shifted = df.copy()
    for c in price_cols:
        shifted[c] = df[c].shift(-lag).ffill()   # fills now happen at the NEXT bar's price
    lag_pnl = _pnl(bt(shifted))
    if abs(base_pnl) < 1e-12:
        verdict = "INSUFFICIENT"
    else:
        ratio = abs(lag_pnl) / abs(base_pnl)
        flipped = (lag_pnl < 0) != (base_pnl < 0)
        if flipped and ratio >= FLIP_MIN_RATIO:
            verdict = "FAIL"
        elif ratio < SHRINK_RATIO:
            verdict = "FAIL"
        elif abs(lag_pnl - base_pnl) / abs(base_pnl) > 0.30:
            verdict = "WARN"
        else:
            verdict = "PASS"
    rep = {"verdict": verdict, "base_pnl": base_pnl, "lagged_pnl": lag_pnl, "lag_bars": lag}
    if verbose:
        print(f"fill_timing(+{lag}bar): {verdict}  pnl {base_pnl:,.1f} -> {lag_pnl:,.1f}")
    return rep


# ---------------------------------------------------------------------------
# 4. Randomized control — permutation test against random signals
# ---------------------------------------------------------------------------

def randomized_control(df: pd.DataFrame, col: str, bt: _BacktestFn,
                       n_shuffles: int = RC_N_SHUFFLES,
                       seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """Shuffle the signal's time order, re-run `n_shuffles` times, compare.

    Separates genuine edge from market beta: if random signals earn as much as the real
    one, the "edge" was just being long in a bull market.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    used_seed = int(seed) if seed is not None else int(
        np.random.default_rng().integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(used_seed)
    real_pnl = _pnl(bt(df.copy()))
    vals = df[col].to_numpy()
    n = len(df)
    shuf_pnls, shuf_trades, failed = [], [], 0
    for _ in range(n_shuffles):
        s = df.copy()
        s[col] = vals[rng.permutation(n)]
        try:
            r = bt(s)
            shuf_pnls.append(_pnl(r))
            shuf_trades.append(_trades(r))
        except Exception:
            failed += 1
    arr = np.asarray(shuf_pnls, dtype=float)
    if len(arr) == 0:
        return {"verdict": "NO_EDGE", "note": "all shuffles failed", "seed": used_seed}
    pct = {p: float(np.percentile(arr, p)) for p in (50, 95)}
    percentile = float(np.mean(arr < real_pnl) * 100.0)
    p_value = (float(np.sum(arr >= real_pnl)) + 1.0) / (len(arr) + 1.0)
    if real_pnl > pct[95] and p_value < RC_ALPHA:
        verdict = "EDGE_CONFIRMED"
    elif real_pnl > pct[50]:
        verdict = "EDGE_WEAK"
    else:
        verdict = "NO_EDGE"
    rep = {"verdict": verdict, "real_pnl": real_pnl,
           "shuffled_mean": float(np.mean(arr)), "shuffled_std": float(np.std(arr)),
           "p50": pct[50], "p95": pct[95], "percentile": round(percentile, 1),
           "p_value": round(p_value, 4), "n_shuffles": len(arr),
           "failed_shuffles": failed, "seed": used_seed}
    if verbose:
        print(f"randomized_control({col}): {verdict}  real {real_pnl:,.0f} vs "
              f"p50 {pct[50]:,.0f} / p95 {pct[95]:,.0f}  (p={p_value:.3f})")
    return rep


# ---------------------------------------------------------------------------
# 5. Return independence — effective sample size
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0:
        return np.zeros(max_lag + 1)
    return np.array([np.dot(x[: n - k], x[k:]) / var for k in range(max_lag + 1)])


def return_independence(rets: Sequence[float], max_lag: int = AC_MAX_LAG,
                        verbose: bool = False) -> Dict:
    """Ljung-Box + ACF. Autocorrelated (overlapping) trades => N_eff << n."""
    x = np.asarray(rets, dtype=float)
    n = len(x)
    if n == 0 or n < AC_MIN_N:
        return {"verdict": "INSUFFICIENT", "n": n, "n_eff": None}
    rho = _acf(x, max_lag)
    q = n * (n + 2) * float(np.sum((rho[1:] ** 2) / (n - np.arange(1, max_lag + 1))))
    lb_p = float(_chi2_sf(q, max_lag))
    sig = [int(k) for k in range(1, max_lag + 1)
           if abs(rho[k]) > 1.96 / math.sqrt(n)]
    denom = 1.0 + 2.0 * float(np.sum((1.0 - np.arange(1, max_lag + 1) / n) * rho[1:] ** 2))
    n_eff = n / denom if denom > 0 else n
    verdict = "AUTOCORRELATED" if lb_p < 0.05 else "INDEPENDENT"
    rep = {"verdict": verdict, "n": n, "n_eff": round(n_eff, 1), "lb_q": q, "lb_p": lb_p,
           "significant_lags": sig, "acf_lag1": round(float(rho[1]), 3)}
    if verbose:
        print(f"return_independence: {verdict}  n={n} -> N_eff={n_eff:.0f}  "
              f"lb_p={lb_p:.4f} lag1={rho[1]:+.3f}")
    return rep


# ---------------------------------------------------------------------------
# 6. Full audit — verdict synthesis
# ---------------------------------------------------------------------------

def full_audit(df: pd.DataFrame, col: str, bt: _BacktestFn,
               bar_seconds: int = 300,
               expansion_confirmation: Optional[str] = None,
               n_shuffles: int = RC_N_SHUFFLES,
               seed: Optional[int] = None,
               verbose: bool = True) -> Dict:
    """Run the gate checks and produce a PASS / CONDITIONAL PASS / FAIL report."""
    lag = lag_sensitivity(df, col, bt, verbose=verbose)
    exp = period_expansion(df, col, bar_seconds=bar_seconds, verbose=verbose)
    fill = fill_timing_sensitivity(df, bt, verbose=verbose)
    rc = randomized_control(df, col, bt, n_shuffles=n_shuffles, seed=seed, verbose=verbose)
    indep = return_independence(bt(df.copy()).get("rets", []), verbose=verbose)

    problems = []
    if lag["verdict"] == "FAIL":
        problems.append(f"lag sensitivity FAIL: pnl {lag['base_pnl']:,.0f} -> "
                        f"{lag['lagged_pnl']:,.0f} after +{lag['lag_bars']} bar shift")
    if fill["verdict"] == "FAIL":
        problems.append(f"fill-timing FAIL: pnl {fill['base_pnl']:,.0f} -> "
                        f"{fill['lagged_pnl']:,.0f} after fills shifted +{fill['lag_bars']} bar")
    if exp["verdict"] == "SUSPECT":
        ok = expansion_confirmation in ("shifted", "completed")
        if not ok:
            problems.append(
                f"period expansion SUSPECT: longest constant run "
                f"{exp['longest_run_bars']} bars ({exp['longest_run_hours']}h) — "
                f"must be explicitly confirmed as shifted/completed")
    if lag["base_trades"] == 0:
        problems.append("no trades in baseline — nothing to validate")

    passed = not problems
    warnings = []
    if rc["verdict"] != "EDGE_CONFIRMED":
        warnings.append(f"randomized control {rc['verdict']} (informational)")
    if indep["verdict"] == "AUTOCORRELATED":
        warnings.append(f"returns autocorrelated: N_eff={indep['n_eff']} < n={indep['n']} "
                        f"(informational)")
    report = {"passed": passed, "lag": lag, "expansion": exp, "fill_timing": fill,
              "randomized_control": rc, "independence": indep,
              "problems": problems, "warnings": warnings,
              "verdict": "PASS" if passed else "FAIL"}
    if verbose:
        print("=" * 62)
        print(f"Full audit [{col}] -> {report['verdict']}")
        for p in problems:
            print(f"  [FAIL] {p}")
        for w in warnings:
            print(f"  [info] {w}")
        print("=" * 62)
    return report
