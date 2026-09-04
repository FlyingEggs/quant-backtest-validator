"""Look-ahead section (V2).

Two tiers, stated honestly:
  * Code-level strategy (signal column exposed) -> lag sensitivity + period expansion
    run as evidence (v1 primitives).
  * Black-box strategy (no signal column)       -> NOT VERIFIED: a lag/expansion test
    needs the actual signal column; refusing to fake it is the point.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from validator import core
from validator.types import DataSpec, Strategy


def check(strategy: Strategy, df: pd.DataFrame, spec: DataSpec,
          confirmation) -> Dict:
    issues, notes = [], []
    if strategy.signal_col is None or strategy.bt_mechanism is None:
        return {"status": "NOT VERIFIED", "issues": [
            {"code": "SIGNAL_NOT_EXPOSED", "severity": "P4",
             "finding": "strategy did not expose its signal column - lag/expansion "
                        "checks skipped (provide signal_col + bt_mechanism for the "
                        "full mechanism suite)"}],
            "notes": ["black-box tier: look-ahead not assessable"]}

    col = strategy.signal_col
    bt = strategy.bt_mechanism
    lag = core.lag_sensitivity(df, col, bt, verbose=False)
    exp = core.period_expansion(df, col, bar_seconds=spec.bar_seconds, verbose=False)

    if lag["verdict"] == "LAG_DEPENDENT":
        issues.append({"code": "LAG_DEPENDENCE", "severity": "P1",
                       "finding": f"pnl {lag['base_pnl']:,.0f} -> "
                       f"{lag['shifted_pnl'] if lag['shifted_pnl'] is not None else 'n/a'} "
                       f"after +{lag['lag_bars']} bar signal lag - review signal "
                       f"construction/timestamps (evidence, not proof)"})
    if exp["verdict"] == "SUSPECT" and confirmation not in ("shifted", "completed"):
        issues.append({"code": "PERIOD_EXPANSION", "severity": "P0",
                       "finding": f"longest constant run {exp['longest_run_bars']} bars "
                       f"({exp['longest_run_hours']}h) - state must be explicitly "
                       f"confirmed as shifted/completed"})
    elif exp["verdict"] == "SUSPECT":
        issues.append({"code": "PERIOD_EXPANSION_CONFIRMED", "severity": "P3",
                       "finding": "expansion SUSPECT resolved by explicit confirmation"})
    notes.append(f"lag={lag['verdict']}, expansion={exp['verdict']}")
    notes.append("code-level timestamp verification NOT PERFORMED - mechanical "
                 "evidence only; a future-function PROOF requires the data/indicator/"
                 "signal/order/fill availability timeline (code review)")

    status = "FAIL" if any(i["severity"] == "P0" for i in issues) else \
             ("CONDITIONAL PASS" if any(i["severity"] == "P1" for i in issues) else "PASS")
    return {"status": status, "issues": issues, "notes": notes}
