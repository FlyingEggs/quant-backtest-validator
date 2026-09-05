"""Adversarial audit of validator/wf.py (V3.3) — hunting real bugs at the
IS/OOS boundary and in the aggregate metrics."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy
from validator.wf import (POLICIES, filter_trades, walk_forward_audit,
                          parameter_freeze_audit)

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def df_hourly(n=200, start="2026-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="300s")
    c = 100.0 + np.arange(n) * 0.0
    o = c - 0.1
    return pd.DataFrame({"open": o, "high": o + 0.2, "low": o - 0.2,
                         "close": c + 0.1}, index=idx)


class OneBoundaryTradeStrategy:
    """Deterministic strategy emitting EXACTLY ONE trade whose entry bar is the
    first bar of the OOS window (signal on the last IS bar, fill at OOS open)."""

    def __init__(self, boundary_bar: int):
        self.boundary_bar = boundary_bar
        self.supports_from_bar = True
        self.entry_semantics = "next_open"
        self.name = "boundary-trade"
        self.default_params = {}
        self.param_grid = None

    def run(self, df, params):
        n = len(df)
        from_bar = int(params.get("_from_bar", 0))
        ts = df.index.to_numpy()
        o = df["open"].to_numpy()
        trades = []
        if self.boundary_bar >= from_bar and self.boundary_bar + 1 < n:
            trades.append({"side": "long", "qty": 1.0,
                           "signal_ts": ts[self.boundary_bar],
                           "entry_ts": ts[self.boundary_bar] + np.timedelta64(300, "s"),
                           "exit_ts": ts[self.boundary_bar] + np.timedelta64(600, "s"),
                           "entry_price": float(o[self.boundary_bar + 1]),
                           "exit_price": float(o[self.boundary_bar + 1]) + 1.0})
        return {"pnl": 1.0 * len(trades), "trades": len(trades),
                "trades_log": trades}


class TestBoundaryFillOffByOne(unittest.TestCase):
    """Bug hunt A: a trade that SIGNALS on the last IS bar and FILLS at the first
    OOS bar must be counted in OOS (entry in window), never dropped by both."""

    def test_boundary_fill_survives_in_os_window(self):
        n = 200
        df = df_hourly(n)
        # OOS window: bars [150, 180); signal on bar 149 -> entry at bar 150
        strat = OneBoundaryTradeStrategy(boundary_bar=149)
        cfg = {"oos": {"policy": "ENTRY_IN_WINDOW", "n_windows": 1,
                       "oos_bars": 30, "min_is_bars": 150, "min_oos_trades": 1}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        w = rep["windows"][0]
        # the single trade enters at the OOS open -> must be in OOS, not dropped
        self.assertEqual(w["oos_trades"], 1,
                         "boundary-fill trade vanished between IS and OOS runs "
                         "(off-by-one in _from_bar)")

    def test_full_policy_boundary_same(self):
        n = 200
        df = df_hourly(n)
        strat = OneBoundaryTradeStrategy(boundary_bar=149)
        cfg = {"oos": {"policy": "FULL_TRADE_IN_WINDOW", "n_windows": 1,
                       "oos_bars": 30, "min_is_bars": 150, "min_oos_trades": 1}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        self.assertEqual(rep["windows"][0]["oos_trades"], 1)


class TestCrossBoundaryAggregate(unittest.TestCase):
    """Bug hunt B: the reported cross-boundary total must be the sum of the
    per-window crossing counts, not a hard-coded zero."""

    def test_cross_boundary_total_matches_rows(self):
        n = 300
        df = df_hourly(n)
        strat = OneBoundaryTradeStrategy(boundary_bar=99)
        cfg = {"oos": {"policy": "FULL_TRADE_IN_WINDOW", "n_windows": 2,
                       "oos_bars": 30, "min_is_bars": 100, "min_oos_trades": 0}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        per_window = sum(r["cross_boundary"] for r in rep["windows"])
        self.assertEqual(rep["cross_boundary_total"], per_window)


class TestPolicyEdges(unittest.TestCase):
    """Boundary-condition correctness of filter_trades."""

    def _t(self, e_off, x_off, base=pd.Timestamp("2026-01-02 00:00")):
        return {"side": "long", "qty": 1.0,
                "entry_ts": base + pd.Timedelta(seconds=e_off),
                "exit_ts": base + pd.Timedelta(seconds=x_off),
                "entry_price": 100.0, "exit_price": 101.0}

    def setUp(self):
        self.lo = pd.Timestamp("2026-01-02 00:00").timestamp()
        self.hi = pd.Timestamp("2026-01-03 00:00").timestamp()

    def test_entry_exactly_on_upper_edge_excluded(self):
        # entry == hi -> not inside this window
        f = filter_trades([self._t(86400, 86400 + 60)], self.lo, self.hi,
                          "ENTRY_IN_WINDOW")
        self.assertEqual(len(f["kept"]), 0)

    def test_entry_exactly_on_lower_edge_included(self):
        f = filter_trades([self._t(0, 60)], self.lo, self.hi, "ENTRY_IN_WINDOW")
        self.assertEqual(len(f["kept"]), 1)

    def test_exit_exactly_on_upper_edge_excluded_for_full(self):
        f = filter_trades([self._t(60, 86400)], self.lo, self.hi,
                          "FULL_TRADE_IN_WINDOW")
        self.assertEqual(len(f["kept"]), 0)


class TestParamFreezeEdges(unittest.TestCase):
    """Documented boundary: a strategy declaring a tunable param_grid that has NO
    effect at all is flagged as PARAM_FREEZE (by design - a dead parameter is itself
    suspicious). Test documents the behaviour, not a silent pass."""

    def test_dead_param_grid_is_flagged_not_silent(self):
        df = df_hourly(120)

        def run(df, params):          # params intentionally unused (constant pnl)
            return {"pnl": 5.0, "trades": 10}

        strat = Strategy(name="dead-grid", run=run, default_params={"k": 1},
                         param_grid={"k": [1, 100]}, entry_semantics="next_open")
        pf = parameter_freeze_audit(strat, df, {})
        self.assertEqual(pf["refit_probe"], "FAIL")     # by design (needs review)
        self.assertTrue(any(i["code"] == "PARAM_FREEZE" for i in pf["issues"]))


if __name__ == "__main__":
    unittest.main()
