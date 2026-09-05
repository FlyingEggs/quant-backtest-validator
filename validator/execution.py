"""Execution section (V3.1) — entry semantics + fill-timing perturbation +
information-boundary timeline (t_information -> t_decision -> t_order -> t_fill).

Timeline audit (P0): the strategy may return per-trade records
`trades=[{"signal_ts": ..., "entry_ts": ..., ...}]`. A fill whose timestamp is NOT
strictly after its signal timestamp used information before it was actionable
(e.g. "decide at 09:35 close, fill at 09:35 close"). Without per-trade timestamps
this sub-check is honestly NOT VERIFIED (the perturbation test still runs).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from validator import core
from validator.types import DataSpec, Strategy, run_metrics


def _to_seconds(ts) -> float:
    """Normalise np.datetime64 / pandas Timestamp / epoch-ms|ns|s ints to seconds."""
    if ts is None:
        return float("nan")
    if isinstance(ts, np.datetime64):
        return float(ts.astype("datetime64[ns]").astype(np.int64)) / 1e9
    try:
        val = float(ts.value if isinstance(ts, pd.Timestamp) else ts)
    except AttributeError:
        val = float(ts)
    if val > 1e17:            # ns
        return val / 1e9
    if val > 1e13:            # ms
        return val / 1e3
    return val                # seconds (or plain numeric comparison axis)


def timeline_audit(trades: List[Dict], min_latency_s: float = 0.0) -> Dict:
    """Mechanical check of the information boundary on per-trade timestamps."""
    need = {"signal_ts", "entry_ts"}
    missing = [i for i, t in enumerate(trades) if not need.issubset(t)]
    if missing:
        return {"verdict": "NOT VERIFIED",
                "reason": f"{len(missing)}/{len(trades)} trades lack signal_ts/entry_ts",
                "violations": []}
    violations, unverifiable = [], []
    for i, t in enumerate(trades):
        d = _to_seconds(t["entry_ts"]) - _to_seconds(t["signal_ts"])
        if not np.isfinite(d):                # cannot judge -> never a silent pass
            unverifiable.append({"trade": i, "signal_ts": str(t["signal_ts"]),
                                 "entry_ts": str(t["entry_ts"])})
            continue
        # min_latency is a floor: entry at signal+latency is legal (d >= latency).
        # A fill at-or-before the signal instant is always illegal.
        if (d < min_latency_s) or (d <= 0.0):
            violations.append({"trade": i, "gap_s": round(d, 3),
                               "signal_ts": str(t["signal_ts"]),
                               "entry_ts": str(t["entry_ts"])})
    if violations:
        return {"verdict": "FAIL", "violations": violations,
                "reason": f"{len(violations)}/{len(trades)} fills at or before their "
                          f"signal time - information used before it was actionable"}
    if unverifiable:
        return {"verdict": "NOT VERIFIED", "violations": [],
                "reason": f"{len(unverifiable)}/{len(trades)} trades have non-finite "
                          f"timestamps - cannot verify the information boundary"}
    return {"verdict": "PASS", "violations": [],
            "reason": f"{len(trades)} trades: entry strictly after signal "
                      f"(min latency {min_latency_s}s)"}


def check(strategy: Strategy, df: pd.DataFrame, spec: DataSpec, config: Dict) -> Dict:
    issues, notes = [], []

    # 1) declared entry semantics: only next_open is self-certifying
    if strategy.entry_semantics != "next_open":
        issues.append({"code": "ENTRY_SEMANTICS", "severity": "P0",
                       "finding": f"entry semantics '{strategy.entry_semantics}' - only "
                       f"next_open is self-certifying; anything else needs an execution "
                       f"model before it can be validated"})
        notes.append("entry semantics not next_open")

    # 2) fill-timing perturbation: always computable via run() (generic, black-box ok)
    def bt(frame: pd.DataFrame) -> Dict:
        return run_metrics(strategy, frame)
    base_res = bt(df)
    fill = core.fill_timing_sensitivity(df, bt, verbose=False)
    if fill["verdict"] == "FAIL":
        ret = float("nan")
        if fill["shifted_pnl"] is not None and abs(fill["base_pnl"]) > 1e-12:
            ret = abs(fill["shifted_pnl"]) / abs(fill["base_pnl"])
        sev, code = ("P0", "EXECUTION_FILL") if (not _isfinite(ret) or ret < 0.10) else \
                    ("P1", "EXECUTION_FILL_REVIEW")
        issues.append({"code": code, "severity": sev,
                       "finding": f"fill-timing pnl {fill['base_pnl']:,.0f} -> "
                       f"{fill['shifted_pnl']:,.0f} after +{fill['lag_bars']} bar fill "
                       f"shift (retains {ret*100:.1f}%) - perturbation evidence; "
                       f"corroborate with an execution model before declaring look-ahead"})
    elif fill["verdict"] == "SENSITIVE":
        issues.append({"code": "FILL_SENSITIVE", "severity": "P2",
                       "finding": "fills moderately timing-sensitive (perturbation)"})
    notes.append(f"fill perturbation={fill['verdict']}, price_cols={list(fill['price_cols'])}")

    # 3) information-boundary timeline (needs per-trade timestamps via 'trades_log')
    trades = base_res.get("trades_log")
    if isinstance(trades, list) and trades:
        tl = timeline_audit(trades, min_latency_s=float(config.get("min_latency_s", 0.0)))
        if tl["verdict"] == "FAIL":
            issues.append({"code": "EXECUTION_TIMELINE", "severity": "P0",
                           "finding": tl["reason"] + f" (first violations: "
                           f"{[v['trade'] for v in tl['violations'][:3]]})"})
        notes.append(f"information-boundary timeline={tl['verdict']} "
                     f"({tl.get('reason', '')})")
    else:
        notes.append("information-boundary timeline NOT VERIFIED - strategy returned no "
                     "per-trade signal_ts/entry_ts records (perturbation test still ran)")

    status = "FAIL" if any(i["severity"] == "P0" for i in issues) else \
             ("CONDITIONAL PASS" if any(i["severity"] == "P1" for i in issues) else "PASS")
    return {"status": status, "issues": issues, "notes": notes}


def _isfinite(x) -> bool:
    import math
    return isinstance(x, float) and math.isfinite(x)
