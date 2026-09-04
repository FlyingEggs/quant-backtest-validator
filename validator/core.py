"""validator — a small, dependency-light backtest validation engine.

Purpose
-------
Demonstrate, with reproducible synthetic data, the *mechanistic* checks that separate a
trustworthy backtest from an overstated one:

  - lag sensitivity         (shift the signal 1 bar -> does the edge depend on timing?)
  - period expansion        (is a low-frequency column reused at bar level?)
  - fill-timing sensitivity (shift fills 1 bar -> does profit collapse?)
  - randomized control      (permutation test: better than a shuffled-null?)
  - return independence     (Ljung-Box + ACF -> how many independent samples?)

Epistemology (v1.1)
-------------------
This is a REFERENCE implementation used to showcase methodology. The engines produce
*evidence*, and the report synthesises it into PASS / CONDITIONAL PASS / FAIL:

  - PASS               no P0/P1 findings
  - CONDITIONAL PASS   no P0, but P1 evidence that needs manual confirmation
                       (e.g. lag-dependence — the signal may be legitimately
                        short-horizon; code review of construction is required)
  - FAIL               at least one P0 finding (execution look-ahead, unconfirmed
                       low-frequency expansion, non-next-open entry semantics)

A lag collapse is *evidence* of look-ahead, not proof: a genuinely short-horizon
momentum strategy also loses money when its signal is shifted one bar. That is why
lag-dependence is P1 (review) while fill-timing / expansion violations are P0 (hard).

Randomized control compares the real signal against its own time-shuffled null
(same value set, time order destroyed; the null keeps the average long exposure).
It controls for static-exposure beta; it does NOT by itself prove "alpha".

Python >= 3.9. Dependencies: numpy, pandas only.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

# --- tuning constants (documented, deterministic) -------------------------------
EXPANSION_HOURS = 2.0      # longest constant run >= this => SUSPECT
SHRINK_RATIO = 0.50        # shifted pnl < 50% of baseline => timing-dependent
SENSITIVE_RATIO = 0.70     # shifted pnl < 70% of baseline => sensitive
FLIP_MIN_RATIO = 0.30      # sign flip with |shifted| >= 30% of |baseline|
RC_N_SHUFFLES = 200        # randomized-control shuffles
RC_ALPHA = 0.05
AC_MAX_LAG = 10
AC_MIN_N = 20

_BacktestFn = Callable[[pd.DataFrame], Dict]

# Verdict vocabularies
LAG_V = ("STABLE", "SENSITIVE", "LAG_DEPENDENT", "INSUFFICIENT")
FILL_V = ("PASS", "SENSITIVE", "FAIL", "INSUFFICIENT")
RC_V = ("BEATS_SHUFFLED_NULL", "WEAK_VS_SHUFFLED_NULL", "NO_EDGE_VS_SHUFFLED_NULL")
INDEP_V = ("INDEPENDENT", "AUTOCORRELATED", "INSUFFICIENT")
AUDIT_V = ("PASS", "CONDITIONAL PASS", "FAIL", "INSUFFICIENT")


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
# 1. Lag sensitivity — timing-dependence / look-ahead *evidence*
# ---------------------------------------------------------------------------

def lag_sensitivity(df: pd.DataFrame, col: str, bt: _BacktestFn,
                    lag: int = 1, verbose: bool = False) -> Dict:
    """Re-run the strategy with the state column shifted `lag` bars later.

    The shifted column keeps NaN at the head (no artificial warm-up fill): strategies
    must be NaN-robust, exactly as they must be in production at the start of history.

    Interpretation is evidence, not verdict: a collapse shows the PnL depends on the
    signal being one bar fresher — which is consistent with look-ahead OR with a
    legitimately short-horizon strategy. Manual construction review is required
    (reported as a P1 issue by full_audit).
    """
    base = bt(df.copy())
    base_pnl = _pnl(base)
    base_tr = _trades(base)
    lagged_df = df.copy()
    lagged_df[col] = df[col].shift(lag)            # NaN head — no artificial initial value
    try:
        lag_res = bt(lagged_df)
        lag_pnl = _pnl(lag_res)
        lag_tr = _trades(lag_res)
        run_error = None
    except Exception as e:                          # NaN-intolerant strategy
        lag_pnl, lag_tr, run_error = float("nan"), 0, f"{type(e).__name__}: {e}"

    if base_tr == 0 or not np.isfinite(base_pnl) or abs(base_pnl) < 1e-12:
        verdict = "INSUFFICIENT"
    elif run_error is not None or not np.isfinite(lag_pnl):
        verdict = "LAG_DEPENDENT"                   # cannot even run shifted => fragile
    else:
        ratio = abs(lag_pnl) / abs(base_pnl)
        flipped = (lag_pnl < 0) != (base_pnl < 0)
        if (flipped and ratio >= FLIP_MIN_RATIO) or ratio < SHRINK_RATIO:
            verdict = "LAG_DEPENDENT"
        elif ratio < SENSITIVE_RATIO or abs(lag_pnl - base_pnl) / abs(base_pnl) > 0.30:
            verdict = "SENSITIVE"
        else:
            verdict = "STABLE"
    rep = {"verdict": verdict, "base_pnl": round(base_pnl, 4),
           "shifted_pnl": None if lag_pnl is None or not np.isfinite(lag_pnl)
           else round(float(lag_pnl), 4),
           "base_trades": base_tr, "lagged_trades": lag_tr, "lag_bars": lag,
           "run_error": run_error,
           "interpretation": (
               "profit depends on the signal being one bar fresher; consistent with "
               "look-ahead OR a legitimately short-horizon strategy - manual review "
               "required (P1)" if verdict == "LAG_DEPENDENT" else
               "edge survives a one-bar signal lag" if verdict == "STABLE" else
               "moderately timing-sensitive; check construction" if verdict == "SENSITIVE"
               else "not enough trades/edge to assess")}
    if verbose:
        print(f"lag_sensitivity({col}, +{lag}bar): {verdict}  "
              f"pnl {base_pnl:,.1f} -> {rep['shifted_pnl'] if rep['shifted_pnl'] is not None else 'n/a'}")
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
    """Shift every fill-relevant price column one bar later and re-run.

    Contract:
      * Pass the columns your fills actually touch (open for market entries,
        high/low for stop/limit exits, close for market-on-close, ...).
      * A column `__fill_shifted__` is set to `lag` on the returned frame so that
        path-dependent strategies can delay their exit scans by one bar.

    Interpretation: a strategy that *claims* next-open entry already fills at the next
    bar's open; shifting one more bar should barely matter. If profit collapses, the
    baseline was implicitly filling one bar early (same-bar knowledge) => P0.

    Scope note: a full execution model (intrabar stop/limit simulation, slippage,
    partial fills) is a separate module (roadmap), not this check.
    """
    base_pnl = _pnl(bt(df.copy()))
    shifted = df.copy()
    for c in price_cols:
        shifted[c] = df[c].shift(-lag).ffill()      # fills now happen at the NEXT bar
    shifted["__fill_shifted__"] = float(lag)
    try:
        lag_pnl = _pnl(bt(shifted))
        run_error = None
    except Exception as e:
        lag_pnl, run_error = float("nan"), f"{type(e).__name__}: {e}"
    if abs(base_pnl) < 1e-12:
        verdict = "INSUFFICIENT"
    elif run_error is not None or not np.isfinite(lag_pnl):
        verdict = "FAIL"
    else:
        ratio = abs(lag_pnl) / abs(base_pnl)
        flipped = (lag_pnl < 0) != (base_pnl < 0)
        if (flipped and ratio >= FLIP_MIN_RATIO) or ratio < SHRINK_RATIO:
            verdict = "FAIL"
        elif ratio < SENSITIVE_RATIO or abs(lag_pnl - base_pnl) / abs(base_pnl) > 0.30:
            verdict = "SENSITIVE"
        else:
            verdict = "PASS"
    rep = {"verdict": verdict, "base_pnl": round(base_pnl, 4),
           "shifted_pnl": None if not np.isfinite(lag_pnl) else round(float(lag_pnl), 4),
           "lag_bars": lag, "price_cols": list(price_cols), "run_error": run_error,
           "interpretation": (
               "profit collapses when fills move one bar later - baseline implicitly "
               "filled with same-bar knowledge (P0)" if verdict == "FAIL" else
               "fills are not timing-sensitive (consistent with next-open)" if verdict == "PASS"
               else "fills moderately timing-sensitive" if verdict == "SENSITIVE"
               else "not assessable")}
    if verbose:
        print(f"fill_timing(+{lag}bar): {verdict}  pnl {base_pnl:,.1f} -> "
              f"{rep['shifted_pnl'] if rep['shifted_pnl'] is not None else 'n/a'}")
    return rep


# ---------------------------------------------------------------------------
# 4. Randomized control — permutation test vs the shuffled null
# ---------------------------------------------------------------------------

def randomized_control(df: pd.DataFrame, col: str, bt: _BacktestFn,
                       n_shuffles: int = RC_N_SHUFFLES,
                       seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """Shuffle the signal's time order (same value set, timing destroyed) and re-run.

    The null keeps the average long exposure of the real signal, so this controls for
    *static-exposure* beta ("random longs also made money"). It does NOT prove alpha:
    regime-structured, state-dependent timing is destroyed by shuffling, so beating the
    shuffled null is evidence of timing information, not a standalone alpha certificate.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    used_seed = int(seed) if seed is not None else int(
        np.random.default_rng().integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(used_seed)
    real_pnl = _pnl(bt(df.copy()))
    real_tr = _trades(bt(df.copy()))
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
        return {"verdict": "NO_EDGE_VS_SHUFFLED_NULL", "note": "all shuffles failed",
                "seed": used_seed}
    pct = {p: float(np.percentile(arr, p)) for p in (50, 95)}
    percentile = float(np.mean(arr < real_pnl) * 100.0)
    p_value = (float(np.sum(arr >= real_pnl)) + 1.0) / (len(arr) + 1.0)
    if real_pnl > pct[95] and p_value < RC_ALPHA:
        verdict = "BEATS_SHUFFLED_NULL"
    elif real_pnl > pct[50]:
        verdict = "WEAK_VS_SHUFFLED_NULL"
    else:
        verdict = "NO_EDGE_VS_SHUFFLED_NULL"
    rep = {"verdict": verdict, "real_pnl": round(real_pnl, 4),
           "real_trades": real_tr,
           "shuffled_mean": round(float(np.mean(arr)), 4),
           "shuffled_std": round(float(np.std(arr)), 4),
           "p50": round(pct[50], 4), "p95": round(pct[95], 4),
           "percentile": round(percentile, 1), "p_value": round(p_value, 4),
           "n_shuffles": len(arr), "failed_shuffles": failed, "seed": used_seed,
           "interpretation": (
               "real signal beats its own time-shuffled null (controls static-exposure "
               "beta; evidence of timing information, not a standalone alpha proof)"
               if verdict == "BEATS_SHUFFLED_NULL" else
               "real signal not clearly better than the shuffled null (informational)")}
    if verbose:
        print(f"randomized_control({col}): {verdict}  real {real_pnl:,.0f} vs "
              f"p50 {pct[50]:,.0f} / p95 {pct[95]:,.0f}  (p={p_value:.3f})")
    return rep


# ---------------------------------------------------------------------------
# 5. Return independence — effective sample size of the mean
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
    """Ljung-Box (autocorrelation *detection*, uses squared rho) + effective sample size.

    N_eff is the effective sample size for estimating the MEAN return of a stationary
    autocorrelated series (Kass et al. 1998; the linear-rho inflation factor
    1 + 2*sum((1 - k/n)*rho_k), not the squared form). Overlapping positions inflate
    variance, so 400 correlated trades behave like far fewer independent samples.
    """
    x = np.asarray(rets, dtype=float)
    n = len(x)
    if n == 0 or n < AC_MIN_N:
        return {"verdict": "INSUFFICIENT", "n": n, "n_eff": None}
    rho = _acf(x, max_lag)
    # Ljung-Box: squared rho (detection)
    q = n * (n + 2) * float(np.sum((rho[1:] ** 2) / (n - np.arange(1, max_lag + 1))))
    lb_p = float(_chi2_sf(q, max_lag))
    sig = [int(k) for k in range(1, max_lag + 1) if abs(rho[k]) > 1.96 / math.sqrt(n)]
    # ESS of the mean: LINEAR rho inflation factor
    infl = 1.0 + 2.0 * float(np.sum((1.0 - np.arange(1, max_lag + 1) / n) * rho[1:]))
    n_eff = n / infl if infl > 0 else float(n)
    n_eff = min(n_eff, float(n))
    verdict = "AUTOCORRELATED" if lb_p < 0.05 else "INDEPENDENT"
    rep = {"verdict": verdict, "n": n, "n_eff": round(n_eff, 1),
           "inflation_factor": round(infl, 3), "lb_q": q, "lb_p": lb_p,
           "significant_lags": sig, "acf_lag1": round(float(rho[1]), 3),
           "interpretation": (
               f"{n} overlapping trades behave like ~{n_eff:.0f} independent samples "
               "for the mean return" if verdict == "AUTOCORRELATED" else
               "no significant return autocorrelation")}
    if verbose:
        print(f"return_independence: {verdict}  n={n} -> N_eff={n_eff:.0f}  "
              f"(inflation x{infl:.2f}, lb_p={lb_p:.4f}, lag1={rho[1]:+.3f})")
    return rep


# ---------------------------------------------------------------------------
# 6. Full audit — three-tier verdict + P0-P4 issue engine + JSON report
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def _issue(code: str, severity: str, finding: str) -> Dict:
    return {"code": code, "severity": severity, "finding": finding}


def full_audit(df: pd.DataFrame, col: str, bt: _BacktestFn,
               bar_seconds: int = 300,
               expansion_confirmation: Optional[str] = None,
               entry_semantics: str = "next_open",
               n_shuffles: int = RC_N_SHUFFLES,
               seed: Optional[int] = None,
               verbose: bool = True) -> Dict:
    """Run the gate checks and produce a three-tier report with a P0-P4 issue log."""
    lag = lag_sensitivity(df, col, bt, verbose=verbose)
    exp = period_expansion(df, col, bar_seconds=bar_seconds, verbose=verbose)
    fill = fill_timing_sensitivity(df, bt, verbose=verbose)
    base_res = bt(df.copy())
    rc = randomized_control(df, col, bt, n_shuffles=n_shuffles, seed=seed, verbose=verbose)
    indep = return_independence(base_res.get("rets", []), verbose=verbose)

    issues = []
    # Execution look-ahead: collapse magnitude decides severity.
    #  <10% retained  -> P0: only same-bar / impossible fills erase ~everything
    #  10-50%         -> P1: also consistent with a legitimately short holding
    #                        horizon; manual execution-semantics review required
    # (lag & fill shifts are the SAME perturbation for 1-bar-horizon signals, so
    #  a moderate collapse can never by itself prove look-ahead.)
    if fill["verdict"] == "FAIL":
        ret = float("nan")
        if fill["shifted_pnl"] is not None and abs(fill["base_pnl"]) > 1e-12:
            ret = abs(fill["shifted_pnl"]) / abs(fill["base_pnl"])
        if not np.isfinite(ret) or ret < 0.10:
            issues.append(_issue("EXECUTION_FILL", "P0",
                                 f"fill-timing FAIL: pnl {fill['base_pnl']:,.0f} -> "
                                 f"{fill['shifted_pnl']:,.0f} after fills shifted "
                                 f"+{fill['lag_bars']} bar - retains {ret*100:.2f}% "
                                 f"(baseline effectively filled with same-bar knowledge)"))
        else:
            issues.append(_issue("EXECUTION_FILL_REVIEW", "P1",
                                 f"fill-timing sensitive: pnl {fill['base_pnl']:,.0f} -> "
                                 f"{fill['shifted_pnl']:,.0f} after fills shifted "
                                 f"+{fill['lag_bars']} bar - retains {ret*100:.0f}%; "
                                 f"consistent with same-bar fills OR a very short holding "
                                 f"horizon - review execution semantics"))
    # P0: entry semantics other than next-open must be justified
    if entry_semantics != "next_open":
        issues.append(_issue("ENTRY_SEMANTICS", "P0",
                             f"entry semantics '{entry_semantics}' - only next_open is "
                             f"self-certifying; anything else requires an execution model"))
    # P0: low-frequency column reused at bar level without confirmation
    if exp["verdict"] == "SUSPECT" and expansion_confirmation not in ("shifted", "completed"):
        issues.append(_issue("PERIOD_EXPANSION", "P0",
                             f"period expansion SUSPECT: longest constant run "
                             f"{exp['longest_run_bars']} bars ({exp['longest_run_hours']}h) - "
                             f"state must be explicitly confirmed as shifted/completed"))
    # P1: lag dependence = evidence requiring manual construction review
    if lag["verdict"] == "LAG_DEPENDENT":
        issues.append(_issue("LAG_DEPENDENCE", "P1",
                             f"lag sensitivity {lag['verdict']}: pnl "
                             f"{lag['base_pnl']:,.0f} -> "
                             f"{lag['shifted_pnl'] if lag['shifted_pnl'] is not None else 'n/a'} "
                             f"after +{lag['lag_bars']} bar shift - review signal "
                             f"construction/timestamps before relying on the edge"))
    # P2: informational hygiene
    if lag["base_trades"] == 0:
        issues.append(_issue("NO_TRADES", "P2", "no trades in baseline - not assessable"))
    if fill["verdict"] == "SENSITIVE":
        issues.append(_issue("FILL_SENSITIVE", "P2", "fills moderately timing-sensitive"))
    if exp["verdict"] == "SUSPECT" and expansion_confirmation in ("shifted", "completed"):
        issues.append(_issue("PERIOD_EXPANSION_CONFIRMED", "P3",
                             "expansion SUSPECT resolved by explicit confirmation - "
                             "analyst must be able to justify the semantics"))

    warnings = []
    if rc["verdict"] != "BEATS_SHUFFLED_NULL":
        warnings.append(f"randomized control: {rc['verdict']} - signal not clearly better "
                        f"than its time-shuffled null (informational)")
    if indep["verdict"] == "AUTOCORRELATED":
        warnings.append(f"returns autocorrelated: N_eff={indep['n_eff']} < n={indep['n']} "
                        f"(informational)")

    has_p0 = any(i["severity"] == "P0" for i in issues)
    has_p1 = any(i["severity"] == "P1" for i in issues)
    assessable = lag["base_trades"] > 0
    if not assessable:
        verdict = "INSUFFICIENT"
    elif has_p0:
        verdict = "FAIL"
    elif has_p1:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    report = {"passed": verdict == "PASS", "verdict": verdict,
              "lag": lag, "expansion": exp, "fill_timing": fill,
              "randomized_control": rc, "independence": indep,
              "issues": issues, "warnings": warnings,
              "severity": max((SEVERITY_ORDER[i["severity"]] for i in issues), default=9)}
    if verbose:
        print("=" * 62)
        print(f"Full audit [{col}] -> {verdict}")
        for i in sorted(issues, key=lambda x: SEVERITY_ORDER[x["severity"]]):
            print(f"  [{i['severity']}] {i['code']}: {i['finding']}")
        for w in warnings:
            print(f"  [info] {w}")
        print("=" * 62)
    return report


# ---------------------------------------------------------------------------
# 7. Report serialization (JSON-safe)
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def save_report(report: Dict, path: str) -> str:
    """Write a full-audit report as indented JSON. Returns the path."""
    with open(path, "w") as fh:
        json.dump(to_jsonable(report), fh, indent=2, ensure_ascii=False)
    return path
