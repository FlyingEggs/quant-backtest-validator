"""V3.4 adversarial edges — genuine island vs plateau strategies at the thresholds,
and report-layer rendering of surface/clustering."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit, audit_text
from validator.surface import surface_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")
GRID = [-2, -1, 0, 1, 2]


def sstrategy(pnl_fn):
    def run(frame, params):
        return {"pnl": pnl_fn(float(params["x"]), float(params["y"])), "trades": 100}
    return Strategy(name="s", run=run, default_params={"x": 0, "y": 0},
                    param_grid={"x": GRID, "y": GRID}, entry_semantics="next_open")


class TestIslandThresholds(unittest.TestCase):
    """Boundary: neighbour exactly at 70% of best must NOT flip to island (FP guard);
    just below must."""

    def test_neighbour_at_threshold_not_island(self):
        # best at (0,0)=100; one neighbour exactly 70 -> not isolated -> NOT island
        def p(x, y):
            if (x, y) == (0.0, 0.0):
                return 100.0
            if (x, y) in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                return 70.0
            return 5.0
        rep = surface_audit(sstrategy(p), D.regime_trend_df(),
                            {"surface": {"x": "x", "y": "y",
                                         "x_values": GRID, "y_values": GRID}})
        self.assertFalse(rep["isolated_best"])           # 70 == 70% threshold counts ok
        self.assertNotEqual(rep["verdict"], "ISLAND")

    def test_neighbour_just_below_threshold_is_island(self):
        def p(x, y):
            if (x, y) == (0.0, 0.0):
                return 100.0
            if (x, y) in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                return 69.9
            return 5.0
        rep = surface_audit(sstrategy(p), D.regime_trend_df(),
                            {"surface": {"x": "x", "y": "y",
                                         "x_values": GRID, "y_values": GRID}})
        self.assertTrue(rep["isolated_best"])
        self.assertEqual(rep["verdict"], "ISLAND")

    def test_plateau_at_corner_not_island(self):
        # best at corner (2,2), neighbours still ~best -> plateau, never island
        def p(x, y):
            return 100.0 - 0.1 * ((x - 2.0) ** 2 + (y - 2.0) ** 2)
        rep = surface_audit(sstrategy(p), D.regime_trend_df(),
                            {"surface": {"x": "x", "y": "y",
                                         "x_values": GRID, "y_values": GRID}})
        self.assertIn(rep["verdict"], ("PLATEAU", "RIDGE"))   # never ISLAND
        codes = {i["code"] for i in rep["issues"]}
        self.assertNotIn("PARAM_ISLAND", codes)


class TestReportRendering(unittest.TestCase):
    """surface/cluster appear as their own lines in the audit text report."""

    def test_surface_line_in_report(self):
        df = D.regime_trend_df()

        def p(x, y):
            return 1000.0 if (x == 0 and y == 0) else 10.0
        strat = sstrategy(p)

        def run(frame, params):
            return {"pnl": p(float(params["x"]), float(params["y"])), "trades": 100}
        strat.run = run
        cfg = {"scope": ["Data Integrity", "Execution", "Statistics", "Robustness"],
               "surface": {"x": "x", "y": "y", "x_values": GRID, "y_values": GRID},
               "seed": 1}
        txt = audit_text(strat, df, SPEC, cfg)
        self.assertIn("Parameter Surface : ISLAND", txt)
        self.assertIn("PARAM_ISLAND", txt)


if __name__ == "__main__":
    unittest.main()
