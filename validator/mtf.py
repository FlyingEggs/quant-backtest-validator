"""V3 — MTF Temporal Availability Engine.

Not a "does 4h data exist" checker. For a signal column on a LOW frame and a HIGH
frame, it reconstructs two counterfactuals per decision time t_dec (= low-bar close):

    legal(t)  = value of the last HIGH bar whose close_time <= t_dec
                (only closed bars are usable -> legal)
    naive(t)  = value of the last HIGH bar whose open   <= t_dec
                (includes the bar still forming at t_dec -> look-ahead)

If the strategy column equals `naive` at times where naive != legal, it used
information that was not yet available -> MTF_LEAK (P0).
If it equals `legal` throughout -> PASS (aligns with last-completed high bar).
If it matches neither -> the column cannot be attributed to this frame/transform
(NOT VERIFIED - refused, not faked).

config['mtf'] = {
    "col": <signal column on the low frame>,
    "frame": <name in DataSpec.timeframes>,
    "frame_seconds": <high bar length in seconds>,
    "transform": "identity" | "sign_diff" | callable(high_series)->series,
}
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from validator.types import DataSpec


def _transform_high(high_value: pd.Series, transform) -> pd.Series:
    if transform is None or transform == "identity":
        return high_value
    if transform == "sign_diff":
        # sign of the diff with NaN (missing close) -> 0. Compute on the ndarray
        # (numpy ufuncs on a Series return ndarray; on the ndarray everything is
        # typed) then rebuild the Series so index/alignment survive.
        sign = np.sign(high_value.diff().to_numpy())
        return pd.Series(np.where(np.isnan(sign), 0.0, sign),
                         index=high_value.index)
    if callable(transform):
        return transform(high_value)
    raise ValueError(f"unknown mtf transform: {transform!r}")


def temporal_availability(df: pd.DataFrame, col: str,
                          high_df: pd.DataFrame, frame_seconds: int,
                          low_seconds: int, transform=None) -> Dict:
    """Core engine. Returns the attribution report (pure, unit-testable)."""
    if col not in df.columns:
        return {"verdict": "NOT VERIFIED", "reason": f"column '{col}' not on low frame",
                "issues": []}
    high_idx = pd.DatetimeIndex(high_df.index)
    if not isinstance(high_idx, pd.DatetimeIndex):
        return {"verdict": "NOT VERIFIED", "reason": "high frame index is not datetime",
                "issues": []}
    value_col = high_df.columns[0] if "close" not in high_df.columns else "close"
    high_val = _transform_high(high_df[value_col], transform)
    high_open = high_idx.values.astype("datetime64[ns]")
    high_close = high_open + np.timedelta64(frame_seconds, "s")

    low_idx = pd.DatetimeIndex(df.index)
    t_dec = low_idx.values.astype("datetime64[ns]") + np.timedelta64(low_seconds, "s")
    sig = df[col].to_numpy(dtype=float)

    # last high bar with open <= t_dec (may still be forming)
    n_naive = np.searchsorted(high_open, t_dec, side="right") - 1
    # last high bar with close <= t_dec (definitely closed)
    n_legal = np.searchsorted(high_close, t_dec, side="right") - 1

    naive_val = np.full(len(df), np.nan)
    legal_val = np.full(len(df), np.nan)
    for i in range(len(df)):
        if n_naive[i] >= 0:
            naive_val[i] = high_val.iloc[int(n_naive[i])]
        if n_legal[i] >= 0:
            legal_val[i] = high_val.iloc[int(n_legal[i])]

    diff_ok = np.isclose(naive_val, legal_val, rtol=0, atol=1e-12) | \
        (np.isnan(naive_val) & np.isnan(legal_val))
    usable = ~np.isnan(sig)
    forming = usable & ~diff_ok                       # naive != legal (incl. no legal yet)
    leak = forming & np.isclose(sig, naive_val, rtol=0, atol=1e-12)
    legal_match = forming & ~np.isnan(legal_val) & \
        np.isclose(sig, legal_val, rtol=0, atol=1e-12)
    leak_n = int(leak.sum())
    forming_n = int(forming.sum())
    leak_frac = leak_n / forming_n if forming_n else 0.0
    other = int((forming & ~leak & ~legal_match).sum())

    # FP guard: a random column matches the forming value ~half the time by chance.
    # Only a *systematic* alignment (>= 90% of forming bars) counts as leakage.
    if leak_n > 0 and leak_frac < 0.9:
        return {"verdict": "NOT VERIFIED", "leak_rows": leak_n,
                "matched_legal_rows": int(legal_match.sum()),
                "unattributed_rows": other, "leak_frac": round(leak_frac, 3),
                "issues": [],
                "reason": f"{leak_n}/{forming_n} forming-bar rows matched the naive "
                          f"value ({leak_frac:.0%}) - partial matches are consistent "
                          f"with chance; exact column provenance required"}
    if leak_frac >= 0.9:
        return {"verdict": "FAIL", "leak_rows": leak_n,
                "matched_legal_rows": int(legal_match.sum()),
                "unattributed_rows": other, "leak_frac": round(leak_frac, 3),
                "issues": [{"code": "MTF_LEAK", "severity": "P0",
                            "finding": f"{leak_n}/{forming_n} decision bars used a "
                            f"HIGH-frame value whose bar had not closed yet "
                            f"(naive==column where legal != naive, {leak_frac:.0%} "
                            f"of forming bars) - not-yet-available information in "
                            f"the signal"}]}
    if other == 0:
        return {"verdict": "PASS", "leak_rows": 0,
                "matched_legal_rows": int(legal_match.sum()),
                "unattributed_rows": 0, "leak_frac": 0.0, "issues": [],
                "note": "column matches the last-COMPLETED high-frame bar at every "
                        "decision time (legal availability)"}
    return {"verdict": "NOT VERIFIED", "leak_rows": 0,
            "matched_legal_rows": int(legal_match.sum()),
            "unattributed_rows": other, "leak_frac": round(leak_frac, 3),
            "issues": [],
            "reason": f"{other} decision bars match neither legal nor naive values - "
                      f"column not attributable to this frame/transform"}


def check(df: pd.DataFrame, spec: DataSpec, config: Dict) -> Dict:
    """Section wrapper for the audit pipeline. No config -> NOT VERIFIED (honest)."""
    m = config.get("mtf")
    if not m:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "MTF_MODULE", "severity": "P4",
                            "finding": "no MTF binding supplied (config['mtf'] + "
                                       "DataSpec.timeframes) - temporal availability "
                                       "not assessed"}],
                "notes": ["supply config['mtf'] and higher-timeframe frames to verify"]}
    name = m.get("frame")
    if name not in spec.timeframes:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "MTF_BINDING", "severity": "P4",
                            "finding": f"mtf binding references frame '{name}' which is "
                                       f"not in DataSpec.timeframes"}],
                "notes": []}
    rep = temporal_availability(df, m["col"], spec.timeframes[name],
                                int(m.get("frame_seconds", 0)),
                                spec.bar_seconds, transform=m.get("transform"))
    tr = m.get("transform")
    if callable(tr):
        # a custom callable transform may embed future access (e.g. shift(-1)) and
        # the engine cannot see inside it - the attribution ran on a DECLARED
        # column, which is not the same as a verified one.
        note = ("custom callable transform: causality NOT mechanically verified "
                "(DECLARED) - provide an identity/sign_diff binding or a causal "
                "contract for a verified verdict")
        if rep["verdict"] == "FAIL":
            return {"status": "FAIL", "issues": rep["issues"],
                    "notes": [note] + [rep]}
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "MTF_TRANSFORM_DECLARED", "severity": "P3",
                            "finding": note}],
                "notes": [rep.get("reason", "custom transform")] + [rep]}
    if rep["verdict"] == "FAIL":
        return {"status": "FAIL", "issues": rep["issues"], "notes": rep}
    if rep["verdict"] == "PASS":
        return {"status": "PASS", "issues": [], "notes": [rep.get("note", "")] + [rep]}
    return {"status": "NOT VERIFIED", "issues": [],
            "notes": [rep.get("reason", "not attributable")] + [rep]}
