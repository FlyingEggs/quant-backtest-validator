"""V3.4 — Parameter Surface audit (2D plateau / island / ridge) + trade clustering.

A 1D adjacent-cliff rule cannot tell a ROBUST region from a parameter-mining ISLAND:
    plateau : many cells near the best   -> healthy
    island  : best cell isolated, tiny plateau -> overfit risk (P1)
    ridge   : best along one axis only    -> one parameter matters, other is noise
    cluster : trades concentrated in a few days -> block dependence (not iid)

config['surface'] = {"x": p1, "y": p2,
                     "x_values": [...], "y_values": [...]}   # pnl = run(df, {.., p1, p2})
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from validator.types import Strategy, run_metrics

PLATEAU_OK_FRAC = 0.70     # cell >= 70% of best counts toward the plateau
ISLAND_MAX_PLATEAU = 0.25  # plateau below this + isolated best => ISLAND
RIDGE_AXIS_FRAC = 0.60


def _pnl(res: Dict) -> float:
    return float(res.get("pnl", 0.0))


def surface_audit(strategy: Strategy, df, config: Dict) -> Dict:
    s = config.get("surface")
    if not s:
        return {"verdict": "NOT VERIFIED", "reason": "no config['surface'] supplied",
                "issues": []}
    x, y = s["x"], s["y"]
    xs, ys = list(s["x_values"]), list(s["y_values"])
    if not xs or not ys:
        return {"verdict": "NOT VERIFIED", "reason": "empty surface axes", "issues": []}
    if len(xs) < 2 or len(ys) < 2:
        return {"verdict": "DEGENERATE_GRID",
                "reason": "a 2D surface needs >=2 values per axis "
                          f"(got {len(xs)}x{len(ys)}) - a single point/line cannot "
                          "support plateau/island/ridge claims",
                "issues": [{"code": "PARAM_DEGENERATE_GRID", "severity": "P1",
                            "finding": f"surface grid is {len(xs)}x{len(ys)} - no 2D "
                            f"structure to classify; widen x_values/y_values or treat "
                            f"parameter robustness as unverified"}],
                "shape": [len(xs), len(ys)]}

    M = np.empty((len(xs), len(ys)))
    bad = []
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            params = dict(strategy.default_params or {})
            params[x], params[y] = xi, yj
            try:
                val = _pnl(run_metrics(strategy, df, params))
            except Exception as exc:   # one exploding cell must not sink the audit
                bad.append(f"({xi}, {yj}) raised {type(exc).__name__}: "
                           f"{str(exc)[:80]}")
                M[i, j] = float("nan")
                continue
            M[i, j] = val
            if not np.isfinite(val):
                bad.append(f"({xi}, {yj}) pnl={val!r}")

    if bad:
        detail = "; ".join(bad[:6])
        if len(bad) > 6:
            detail += f" (+{len(bad) - 6} more)"
        return {"verdict": "NON_FINITE_PNL", "n_bad": len(bad), "reason": detail,
                "issues": [{"code": "PARAM_NONFINITE_PNL", "severity": "P1",
                            "finding": f"{len(bad)} surface cell(s) failed to evaluate "
                            f"({bad[0]}) - surface not classifiable; fix the strategy "
                            f"before trusting parameter robustness"}],
                "shape": [len(xs), len(ys)]}

    best = float(np.max(M))
    bi, bj = (int(v) for v in np.unravel_index(np.argmax(M), M.shape))
    if best <= 0:
        return {"verdict": "NO_POSITIVE_REGION", "issues": [],
                "best": best, "best_params": {x: xs[bi], y: ys[bj]},
                "shape": [len(xs), len(ys)]}

    ok = M >= best * PLATEAU_OK_FRAC
    plateau_frac = float(np.mean(ok))

    # island: every existing orthogonal neighbour is far below the best
    nb = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        i2, j2 = bi + di, bj + dj
        if 0 <= i2 < M.shape[0] and 0 <= j2 < M.shape[1]:
            nb.append(float(M[i2, j2]))
    isolated = bool(nb) and all(v < best * PLATEAU_OK_FRAC for v in nb)

    # ridge: best sits on a high row (fixed x) or high column (fixed y)
    row_ok = float(np.mean(ok[bi, :]))          # vary y, fixed best x
    col_ok = float(np.mean(ok[:, bj]))          # vary x, fixed best y
    ridge_x = row_ok >= RIDGE_AXIS_FRAC and col_ok < RIDGE_AXIS_FRAC
    ridge_y = col_ok >= RIDGE_AXIS_FRAC and row_ok < RIDGE_AXIS_FRAC

    issues = []
    if isolated and plateau_frac < ISLAND_MAX_PLATEAU:
        verdict = "ISLAND"
        issues.append({"code": "PARAM_ISLAND", "severity": "P1",
                       "finding": f"best point ({x}={xs[bi]}, {y}={ys[bj]}, pnl "
                       f"{best:,.0f}) is an isolated optimum: all orthogonal neighbours "
                       f"< {PLATEAU_OK_FRAC:.0%} of best and plateau only "
                       f"{plateau_frac:.0%} of the surface - parameter-mining island / "
                       f"overfit signature; not a robust region"})
    elif ridge_x or ridge_y:
        verdict = "RIDGE"
        axis = x if ridge_x else y
        issues.append({"code": "PARAM_RIDGE", "severity": "P2",
                       "finding": f"performance depends mainly on '{axis}': best "
                       f"row/column covers {max(row_ok, col_ok):.0%} while the other "
                       f"axis carries {min(row_ok, col_ok):.0%} - one parameter is "
                       f"informative, the other is flat noise"})
    elif plateau_frac >= 0.6:
        verdict = "PLATEAU"
        issues.append({"code": "PARAM_PLATEAU", "severity": "P4",
                       "finding": f"robust plateau: {plateau_frac:.0%} of the 2D surface "
                       f"within {PLATEAU_OK_FRAC:.0%} of best - parameters sit in a "
                       f"stable region"})
    else:
        verdict = "NOISY"
        issues.append({"code": "PARAM_NOISY", "severity": "P2",
                       "finding": "surface is neither plateau, ridge nor island - "
                                  "fragmented/noisy parameter response"})

    return {"verdict": verdict, "issues": issues,
            "best_pnl": round(best, 4), "best_params": {x: xs[bi], y: ys[bj]},
            "plateau_frac": round(plateau_frac, 3), "isolated_best": isolated,
            "ridge_along_x": ridge_x, "ridge_along_y": ridge_y,
            "row_ok_frac": round(row_ok, 3), "col_ok_frac": round(col_ok, 3),
            "surface_stats": {"min": round(float(np.min(M)), 4),
                              "median": round(float(np.median(M)), 4),
                              "max": round(best, 4),
                              "std": round(float(np.std(M)), 4)},
            "shape": [len(xs), len(ys)]}


def cluster_audit(trades_log: List[Dict]) -> Dict:
    """Block/cluster dependence: how many calendar days actually carry the trades?"""
    if not trades_log:
        return {"verdict": "NOT VERIFIED", "reason": "no trades_log", "issues": []}
    import pandas as pd
    from validator.execution import _to_seconds   # same ns/ms/s normalisation as timeline
    days: Dict[str, int] = {}
    for t in trades_log:
        ts = t.get("entry_ts") or t.get("signal_ts")
        if ts is None:
            return {"verdict": "NOT VERIFIED",
                    "reason": "trades lack entry timestamps", "issues": []}
        secs = _to_seconds(ts)
        if not np.isfinite(secs):
            return {"verdict": "NOT VERIFIED",
                    "reason": "trades carry non-finite timestamps", "issues": []}
        d = pd.Timestamp(secs, unit="s").strftime("%Y-%m-%d")
        days[d] = days.get(d, 0) + 1
    n = len(trades_log)
    counts = sorted(days.values(), reverse=True)
    top_day = counts[0] if counts else 0
    issues = []
    if len(days) < 3 and n >= 5:
        issues.append({"code": "TRADE_CLUSTERING", "severity": "P3",
                       "finding": f"{n} trades concentrated in only {len(days)} "
                       f"day(s) (top day {top_day}) - results behave like a handful "
                       f"of market episodes, not iid samples"})
    return {"verdict": "PASS" if not issues else "CLUSTERED",
            "raw_trades": n, "active_days": len(days),
            "top_day_trades": top_day, "issues": issues}
