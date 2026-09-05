"""Data integrity checks (V2) — structural sanity of the frame before any audit."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")


def check(df: pd.DataFrame, spec=None) -> Dict:
    issues: List[Dict] = []
    notes: List[str] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append({"code": "DATA_INDEX", "severity": "P0",
                       "finding": "index is not a DatetimeIndex - bar alignment cannot be trusted"})
    else:
        if not df.index.is_monotonic_increasing:
            issues.append({"code": "DATA_INDEX", "severity": "P0",
                           "finding": "index is not monotonically increasing"})
        dups = int(df.index.duplicated().sum())
        if dups:
            issues.append({"code": "DATA_DUP", "severity": "P1",
                           "finding": f"{dups} duplicate timestamps present"})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append({"code": "DATA_COLS", "severity": "P0",
                       "finding": f"missing required columns: {missing}"})
        return {"status": "FAIL", "issues": issues, "notes": notes}

    for c in REQUIRED_COLS:
        bad = int((df[c] <= 0).sum())
        if bad:
            issues.append({"code": "DATA_NONPOS", "severity": "P0",
                           "finding": f"column {c}: {bad} non-positive values"})
    if "high" in df.columns and "low" in df.columns:
        inv = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
        if inv:
            issues.append({"code": "DATA_HL", "severity": "P1",
                           "finding": f"{inv} rows where high < max(open, close)"})
        inv2 = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
        if inv2:
            issues.append({"code": "DATA_HL", "severity": "P1",
                           "finding": f"{inv2} rows where low > min(open, close)"})

    nan_cells = int(df[list(REQUIRED_COLS)].isna().sum().sum())
    if nan_cells:
        issues.append({"code": "DATA_NAN", "severity": "P1",
                       "finding": f"{nan_cells} NaN cells in OHLC"})

    status = "FAIL" if any(i["severity"] == "P0" for i in issues) else \
             ("CONDITIONAL PASS" if any(i["severity"] == "P1" for i in issues) else "PASS")
    notes.append(f"{len(df)} bars checked")
    return {"status": status, "issues": issues, "notes": notes}
