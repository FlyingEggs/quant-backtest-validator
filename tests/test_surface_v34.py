"""V3.4 — Parameter Surface tests: plateau vs island vs ridge, plus clustering."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy
from validator.robustness import check as rcheck
from validator.surface import surface_audit, cluster_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def surf_strategy(pnl_fn):
    """pnl_fn(x_val, y_val) -> pnl; grid declared on the strategy."""

    def run(frame, params):
        x = float(params["x"])
        y = float(params["y"])
        return {"pnl": pnl_fn(x, y), "trades": 100}

    return Strategy(name="surface", run=run, default_params={"x": 0, "y": 0},
                    param_grid={"x": [-2, -1, 0, 1, 2],
                                "y": [-2, -1, 0, 1, 2]},
                    entry_semantics="next_open")


def cfg(xs, ys):
    return {"x": "x", "y": "y", "x_values": xs, "y_values": ys}


class TestSurfaceClassification(unittest.TestCase):

    def test_plateau(self):
        # broad smooth bowl: many cells near best -> PLATEAU, never an island
        def p(x, y):
            return 100.0 - 0.1 * (x * x + y * y)
        strat = surf_strategy(p)
        rep = surface_audit(strat, D.regime_trend_df(),
                            {"surface": cfg([-2, -1, 0, 1, 2],
                                            [-2, -1, 0, 1, 2])})
        self.assertEqual(rep["verdict"], "PLATEAU")
        self.assertGreaterEqual(rep["plateau_frac"], 0.6)
        self.assertFalse(any(i["code"] == "PARAM_ISLAND" for i in rep["issues"]))

    def test_island(self):
        # single spike at center on a flat 10 background
        def p(x, y):
            return 1000.0 if (x == 0 and y == 0) else 10.0
        strat = surf_strategy(p)
        rep = surface_audit(strat, D.regime_trend_df(),
                            {"surface": cfg([-2, -1, 0, 1, 2],
                                            [-2, -1, 0, 1, 2])})
        self.assertEqual(rep["verdict"], "ISLAND")
        self.assertTrue(rep["isolated_best"])
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("PARAM_ISLAND", codes)
        self.assertEqual([i["severity"] for i in rep["issues"]
                          if i["code"] == "PARAM_ISLAND"], ["P1"])

    def test_ridge(self):
        # performance depends only on y (all x share the same value)
        def p(x, y):
            return 5.0 * (y + 3)
        strat = surf_strategy(p)
        rep = surface_audit(strat, D.regime_trend_df(),
                            {"surface": cfg([-2, -1, 0, 1, 2],
                                            [-2, -1, 0, 1, 2])})
        self.assertEqual(rep["verdict"], "RIDGE")
        self.assertTrue(rep["ridge_along_y"] or rep["ridge_along_x"])

    def test_integration_robustness_issue(self):
        def p(x, y):
            return 1000.0 if (x == 0 and y == 0) else 10.0
        strat = surf_strategy(p)
        sec = rcheck(strat, D.regime_trend_df(), SPEC,
                     {"surface": cfg([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]),
                      "seed": 1})
        codes = {i["code"] for i in sec["issues"]}
        self.assertIn("PARAM_ISLAND", codes)
        notes = " ".join(sec["notes"])
        self.assertIn("parameter surface", notes)


class TestClusterAudit(unittest.TestCase):

    def test_clustered_days_flagged(self):
        base = pd.Timestamp("2026-01-01 00:00")
        logs = [{"entry_ts": base + pd.Timedelta(hours=i)} for i in range(8)]
        rep = cluster_audit(logs)                  # same day
        self.assertEqual(rep["verdict"], "CLUSTERED")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("TRADE_CLUSTERING", codes)
        self.assertEqual(rep["active_days"], 1)

    def test_spread_days_clean(self):
        base = pd.Timestamp("2026-01-01 00:00")
        logs = [{"entry_ts": base + pd.Timedelta(days=d)} for d in range(8)]
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["active_days"], 8)

    def test_no_logs_not_verified(self):
        rep = cluster_audit([])
        self.assertEqual(rep["verdict"], "NOT VERIFIED")


if __name__ == "__main__":
    unittest.main()
