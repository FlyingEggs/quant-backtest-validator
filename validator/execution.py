"""Execution section (V2) — entry semantics + fill-timing perturbation evidence."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from validator import core
from validator.types import DataSpec, Strategy, run_metrics


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

    status = "FAIL" if any(i["severity"] == "P0" for i in issues) else \
             ("CONDITIONAL PASS" if any(i["severity"] == "P1" for i in issues) else "PASS")
    return {"status": status, "issues": issues, "notes": notes}


def _isfinite(x) -> bool:
    import math
    return isinstance(x, float) and math.isfinite(x)
