"""Robustness section (V2) — randomized control + OOS split + parameter sensitivity."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from validator import core
from validator.types import DataSpec, Strategy, run_metrics

PARAM_CLIFF_RATIO = 2.0   # adjacent-parameter |pnl delta| > 2x median |pnl| => cliff


def check(strategy: Strategy, df: pd.DataFrame, spec: DataSpec, config: Dict) -> Dict:
    issues, notes = [], []
    frac = float(config.get("oos_frac", 0.30))
    n = len(df)

    # ---- randomized control (mechanism tier; informational) -------------------
    rc_note = None
    if strategy.signal_col is not None and strategy.bt_mechanism is not None:
        rc = core.randomized_control(df, strategy.signal_col, strategy.bt_mechanism,
                                     n_shuffles=int(config.get("n_shuffles", 200)),
                                     seed=config.get("seed", 42), verbose=False)
        if rc["verdict"] != "BEATS_SHUFFLED_NULL":
            rc_note = (f"randomized control {rc['verdict']}: real {rc['real_pnl']:,.0f} vs "
                       f"null p50 {rc['p50']:,.0f} - not clearly better than the "
                       f"time-shuffled null")
    else:
        rc_note = "randomized control NOT VERIFIED (signal column not exposed)"

    # ---- chronological OOS split ----------------------------------------------
    base = run_metrics(strategy, df)
    split = int(n * (1.0 - frac))
    if split < 50 or n - split < 50:
        notes.append(f"OOS skipped: sample too small (n={n}) for a {frac:.0%} split")
    else:
        is_res = run_metrics(strategy, df.iloc[:split])
        oos_res = run_metrics(strategy, df.iloc[split:])
        is_p, oos_p, oos_t = is_res["pnl"], oos_res["pnl"], oos_res["trades"]
        if oos_t == 0:
            notes.append("OOS produced no trades - inconclusive")
        elif (is_p > 0) != (oos_p > 0) or (is_p > 0 and oos_p < 0):
            issues.append({"code": "OOS_INSTABILITY", "severity": "P1",
                           "finding": f"IS pnl {is_p:,.0f} vs OOS pnl {oos_p:,.0f} "
                           f"({oos_t} OOS trades) - performance does not hold "
                           f"out-of-sample"})
        elif is_p > 0 and oos_p < 0.5 * is_p * (oos_t / max(1, is_res.get("trades", 1))):
            issues.append({"code": "OOS_DECAY", "severity": "P2",
                           "finding": "OOS pnl materially below IS after trade-count "
                           "scaling"})
        else:
            notes.append(f"OOS consistent: IS {is_p:,.0f} / OOS {oos_p:,.0f} "
                         f"({oos_t} trades)")

    # ---- parameter sensitivity (only when a grid is declared) -----------------
    if strategy.param_grid and strategy.default_params is not None:
        for pname, values in strategy.param_grid.items():
            pnls: List[float] = []
            for v in values:
                params = dict(strategy.default_params)
                params[pname] = v
                pnls.append(float(run_metrics(strategy, df, params)["pnl"]))
            med = float(np.median(np.abs(pnls))) or 1e-9
            deltas = [abs(pnls[i + 1] - pnls[i]) / med for i in range(len(pnls) - 1)]
            worst = max(deltas) if deltas else 0.0
            if worst > PARAM_CLIFF_RATIO:
                issues.append({"code": "PARAM_CLIFF", "severity": "P2",
                               "finding": f"parameter '{pname}' shows a cliff: max "
                               f"adjacent |pnl| swing {worst:.1f}x median across "
                               f"{values} (isolated peak / overfit risk)"})
            else:
                notes.append(f"parameter '{pname}' sweep stable "
                             f"(max swing {worst:.1f}x median)")
    else:
        notes.append("parameter sensitivity NOT VERIFIED (needs run(df, params) + "
                     "param_grid)")

    if rc_note:
        notes.append(rc_note)
    status = "FAIL" if any(i["severity"] == "P0" for i in issues) else \
             ("CONDITIONAL PASS" if any(i["severity"] == "P1" for i in issues) else "PASS")
    return {"status": status, "issues": issues, "notes": notes}
