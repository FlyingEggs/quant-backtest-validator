"""V3.3 — OOS / Walk-Forward as an enforceable research contract.

Three machine contracts:

1) TRADE BOUNDARY POLICY  - how a trade straddling an IS/OOS cutoff is counted:
     ENTRY_IN_WINDOW   : trade belongs where its ENTRY lies
     EXIT_IN_WINDOW    : trade belongs where its EXIT lies
     FULL_TRADE_IN_WINDOW : both entry and exit must lie in the window
   The chosen policy is reported; cross-boundary trades are counted, not hidden.

2) PARAMETER FREEZE      - OOS runs must use the frozen IS parameters. Two probes:
     determinism  : identical (df, params) must give identical output
     contamination: if the strategy DECLARES a tunable param_grid yet produces
                    IDENTICAL results across the grid extremes on OOS, its frozen
                    parameters are being ignored -> internal re-fit suspected (P0).

3) WALK FORWARD          - expanding-IS / rolling-OOS windows with per-window
   IS/OOS metrics and OOS-consistency aggregates (positive-window %, expectancy
   consistency, trade adequacy).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validator.types import DataSpec, Strategy, run_metrics

POLICIES = ("ENTRY_IN_WINDOW", "EXIT_IN_WINDOW", "FULL_TRADE_IN_WINDOW")


def _secs(ts) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, np.datetime64):
        return float(ts.astype("datetime64[ns]").astype(np.int64)) / 1e9
    try:
        v = float(ts.value if isinstance(ts, pd.Timestamp) else ts)
    except AttributeError:
        v = float(ts)
    if v > 1e17:
        return v / 1e9
    if v > 1e13:
        return v / 1e3
    return v


def _trade_pnl(tr: Dict) -> Optional[float]:
    if "entry_price" not in tr or "exit_price" not in tr:
        return None
    qty = float(tr.get("qty", tr.get("contracts", 1.0)))
    d = 1.0 if tr.get("side", "long") in ("long", "buy") else -1.0
    return (float(tr["exit_price"]) - float(tr["entry_price"])) * d * qty


def filter_trades(trades: List[Dict], lo_s: float, hi_s: float,
                  policy: str) -> Dict:
    """Apply the boundary policy; report included/excluded/crossing counts."""
    if policy not in POLICIES:
        raise ValueError(f"unknown boundary policy {policy!r}; choose {POLICIES}")
    kept, crossed = [], 0
    for tr in trades:
        e, x = _secs(tr.get("entry_ts")), _secs(tr.get("exit_ts"))
        if policy == "ENTRY_IN_WINDOW":
            if e is None:
                continue
            inside = lo_s <= e < hi_s
        elif policy == "EXIT_IN_WINDOW":
            if x is None:
                continue
            inside = lo_s <= x < hi_s
        else:  # FULL_TRADE_IN_WINDOW
            if e is None or x is None:
                continue
            inside = lo_s <= e and x < hi_s
        straddles = (e is not None and e < lo_s and x is not None and x >= hi_s) or \
                    (e is not None and lo_s <= e < hi_s and x is not None and x >= hi_s) or \
                    (e is not None and e < lo_s and x is not None and lo_s <= x < hi_s)
        if inside:
            kept.append(tr)
        if straddles:
            crossed += 1
    return {"kept": kept, "crossed": crossed,
            "policy": policy, "total": len(trades)}


def _sum_pnl(trades: List[Dict]) -> Dict:
    pnls = [p for p in (_trade_pnl(t) for t in trades) if p is not None]
    return {"pnl": float(np.sum(pnls)) if pnls else 0.0, "trades": len(pnls)}


def parameter_freeze_audit(strategy: Strategy, df: pd.DataFrame,
                           config: Dict) -> Dict:
    """Determinism + contamination probes over the OOS material."""
    out = {"determinism": "NOT VERIFIED", "refit_probe": "NOT VERIFIED",
           "issues": []}
    probe_params: List[Dict] = []
    default = dict(strategy.default_params or {})
    probe_params.append(default)
    grid = strategy.param_grid or {}
    for k, vals in grid.items():
        if len(vals) >= 2 and default.get(k) is not None:
            a, b = dict(default), dict(default)
            a[k], b[k] = vals[0], vals[-1]
            probe_params = [a, b]
            break

    # determinism: same (df, params) twice
    try:
        r1 = run_metrics(strategy, df, probe_params[0])
        r2 = run_metrics(strategy, df, probe_params[0])
        if _pnl(r1) == _pnl(r2) and r1.get("trades") == r2.get("trades"):
            out["determinism"] = "PASS"
        else:
            out["determinism"] = "FAIL"
            out["issues"].append({"code": "NON_DETERMINISTIC", "severity": "P0",
                                  "finding": "identical (df, params) produced "
                                             "different results - strategy is not a "
                                             "pure function of its inputs"})
    except Exception as e:
        out["determinism"] = "FAIL"
        out["issues"].append({"code": "PROBE_ERROR", "severity": "P2",
                              "finding": f"parameter-freeze probe failed: {e}"})

    # contamination: declared-tunable params must CHANGE OOS output
    if len(probe_params) >= 2:
        ra = run_metrics(strategy, df, probe_params[0])
        rb = run_metrics(strategy, df, probe_params[1])
        same = _pnl(ra) == _pnl(rb) and ra.get("trades") == rb.get("trades")
        if same and int(ra.get("trades", 0)) > 0:
            out["refit_probe"] = "FAIL"
            out["issues"].append({"code": "PARAM_FREEZE", "severity": "P0",
                                  "finding": "declared tunable param_grid yet OOS "
                                             "output is identical across grid extremes "
                                             "- frozen parameters are being ignored; "
                                             "internal re-fit / dead parameter "
                                             "suspected (manual review to confirm)"})
        else:
            out["refit_probe"] = "PASS"
    return out


def _pnl(res: Dict) -> float:
    return float(res.get("pnl", 0.0))


def walk_forward_audit(strategy: Strategy, df: pd.DataFrame,
                       config: Dict, spec: DataSpec) -> Dict:
    oos = config.get("oos", {})
    policy = oos.get("policy", "FULL_TRADE_IN_WINDOW")
    n_windows = int(oos.get("n_windows", 3))
    oos_bars = int(oos.get("oos_bars", 200))
    min_is_bars = int(oos.get("min_is_bars", 400))
    min_oos_trades = int(oos.get("min_oos_trades", 5))
    times = df.index.values.astype("datetime64[s]").astype(np.int64)
    n = len(df)
    supports = getattr(strategy, "supports_from_bar", False)

    rows = []
    for w in range(n_windows):
        o_start = min_is_bars + w * oos_bars
        o_end = min(o_start + oos_bars, n)
        if o_start >= n or o_end <= o_start:
            break
        lo_s, hi_s = float(times[o_start]), float(times[o_end - 1]) + 1.0

        # IS window: full history up to cutoff (warm from sample start)
        is_res = run_metrics(strategy, df.iloc[:o_start])
        is_trades = is_res.get("trades_log") or []
        is_f = filter_trades(is_trades, float(times[0]), lo_s, policy)
        is_m = _sum_pnl(is_f["kept"])

        # OOS window: warm-up context (full history) when supported, else cold slice.
        # _from_bar = o_start-1 so a signal on the LAST IS bar (whose fill lands at the
        # OOS open) is generated; the boundary policy then decides where it belongs.
        params = dict(strategy.default_params or {})
        if supports:
            params["_from_bar"] = max(0, o_start - 1)
            oos_res = run_metrics(strategy, df.iloc[:o_end], params)
        else:
            oos_res = run_metrics(strategy, df.iloc[o_start:o_end])
        oos_trades = oos_res.get("trades_log") or []
        oos_f = filter_trades(oos_trades, lo_s, hi_s, policy)
        oos_m = _sum_pnl(oos_f["kept"])
        crossed = oos_f["crossed"] + is_f["crossed"]

        oos_pt = oos_m["pnl"] / oos_m["trades"] if oos_m["trades"] else None
        is_pt = is_m["pnl"] / is_m["trades"] if is_m["trades"] else None
        if oos_m["trades"] < min_oos_trades:
            status = "INSUFFICIENT"
        elif oos_pt is not None and oos_pt > 0:
            status = "PASS"
        else:
            status = "FAIL"
        rows.append({"window": f"W{w + 1}",
                     "is_pnl": round(is_m["pnl"], 2), "is_trades": is_m["trades"],
                     "oos_pnl": round(oos_m["pnl"], 2), "oos_trades": oos_m["trades"],
                     "oos_pnl_per_trade": (round(oos_pt, 4) if oos_pt is not None else None),
                     "status": status, "cross_boundary": crossed,
                     "policy": policy})

    scored = [r for r in rows if r["status"] != "INSUFFICIENT"]
    positive = sum(1 for r in scored if (r["oos_pnl_per_trade"] or 0) > 0)
    pos_pct = positive / len(scored) if scored else 0.0
    consist = []
    for r in rows:
        if r["is_trades"] and r["oos_pnl_per_trade"] is not None:
            is_pt = r["is_pnl"] / r["is_trades"]
            if is_pt != 0:
                consist.append((r["oos_pnl_per_trade"] > 0) == (is_pt > 0))
    consist_pct = sum(consist) / len(consist) if consist else None
    adequate = sum(1 for r in rows if r["oos_trades"] >= min_oos_trades)
    adequacy_pct = adequate / len(rows) if rows else 0.0

    issues = []
    if scored and pos_pct < 0.6:
        issues.append({"code": "WF_LOW_CONSISTENCY", "severity": "P1",
                       "finding": f"OOS positive-window rate {pos_pct:.0%} < 60% "
                       f"(expectancy consistency "
                       f"{f'{consist_pct:.0%}' if consist_pct is not None else 'n/a'})"})
    for r in rows:
        if r["status"] == "FAIL":
            issues.append({"code": "WF_WINDOW_FAIL", "severity": "P2",
                           "finding": f"window {r['window']} OOS PnL/trade "
                           f"{r['oos_pnl_per_trade']} not positive"})
    return {"policy": policy, "windows": rows, "issues": issues,
            "positive_window_pct": round(pos_pct, 3),
            "expectancy_consistency_pct": (round(consist_pct, 3)
                                           if consist_pct is not None else None),
            "trade_adequacy_pct": round(adequacy_pct, 3),
            "cross_boundary_total": int(sum(r["cross_boundary"] for r in rows))}
