"""V3.4 integrity — attack the surface/cluster engines on data-quality edges.

Code-auditor round: a single exploding grid cell or a degenerate grid must never
yield a clean-looking classification (PLATEAU / ISLAND are *confirmation* labels),
and cluster day-bucketing must not misread epoch-*seconds* as nanoseconds.

    inf cell  -> previously reported ISLAND (param-mining charge on a blown-up eval)
    nan cell  -> previously NOISY (silent, with RuntimeWarning)
    raising cell -> previously crashed the whole audit
    1x1 / 1xN grid -> previously "PLATEAU 100%" (no 2D evidence at all)
    int epoch seconds -> previously all collapsed to 1970-01-01 (false CLUSTERED)
"""

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
GRID = [-2, -1, 0, 1, 2]


def surf_strategy(pnl_fn, xs=GRID, ys=GRID):
    """pnl_fn(x_val, y_val) -> pnl; a single exploding cell can be injected."""

    def run(frame, params):
        return {"pnl": pnl_fn(float(params["x"]), float(params["y"])),
                "trades": 100}

    return Strategy(name="surface", run=run, default_params={"x": 0, "y": 0},
                    param_grid={"x": xs, "y": ys},
                    entry_semantics="next_open")


def cfg(xs=GRID, ys=GRID):
    return {"surface": {"x": "x", "y": "y", "x_values": xs, "y_values": ys}}


def smooth(x, y):
    """A clean bowl - must stay PLATEAU everywhere a cell is finite."""
    return 100.0 - (x * x + y * y)


class TestNonFiniteCells(unittest.TestCase):
    """One blown-up evaluation must be reported as a data failure, never silently
    reclassified and never mistaken for a parameter island."""

    def test_inf_cell_is_not_island(self):
        def p(x, y):
            if x == 1.0 and y == 1.0:
                return float("inf")
            return smooth(x, y)
        rep = surface_audit(surf_strategy(p), D.regime_trend_df(), cfg())
        self.assertEqual(rep["verdict"], "NON_FINITE_PNL")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("PARAM_NONFINITE_PNL", codes)
        self.assertNotIn("PARAM_ISLAND", codes)   # not a parameter-mining charge
        self.assertEqual(next(i["severity"] for i in rep["issues"]
                              if i["code"] == "PARAM_NONFINITE_PNL"), "P1")
        self.assertIn("(1, 1)", rep["reason"])    # the bad cell is located
        self.assertEqual(rep["n_bad"], 1)         # never double-counted

    def test_nan_cell_is_not_silent(self):
        def p(x, y):
            if x == -1.0 and y == 2.0:
                return float("nan")
            return smooth(x, y)
        rep = surface_audit(surf_strategy(p), D.regime_trend_df(), cfg())
        self.assertEqual(rep["verdict"], "NON_FINITE_PNL")
        self.assertEqual(rep["n_bad"], 1)
        codes = {i["code"] for i in rep["issues"]}
        self.assertNotIn("PARAM_NOISY", codes)     # never a silent reclassification

    def test_raising_cell_does_not_crash(self):
        def p(x, y):
            if x == 0.0 and y == 1.0:
                raise ValueError("pnl division exploded")
            return smooth(x, y)
        rep = surface_audit(surf_strategy(p), D.regime_trend_df(), cfg())
        self.assertEqual(rep["verdict"], "NON_FINITE_PNL")   # audit survives
        self.assertEqual(rep["n_bad"], 1)                     # no double count
        self.assertIn("(0, 1)", rep["reason"])

    def test_all_bad_cells_counted(self):
        def p(x, y):
            if x in (2.0, -2.0) and y == -2.0:
                return float("nan")
            return smooth(x, y)
        rep = surface_audit(surf_strategy(p), D.regime_trend_df(), cfg())
        self.assertEqual(rep["verdict"], "NON_FINITE_PNL")
        self.assertEqual(rep["n_bad"], 2)


class TestDegenerateGrids(unittest.TestCase):
    """No 2D structure -> no confirmation label. A single point / a line cannot
    support 'PLATEAU 100% of the surface'."""

    def test_single_point_grid_not_plateau(self):
        rep = surface_audit(surf_strategy(lambda x, y: 50.0, [0], [0]),
                            D.regime_trend_df(), cfg([0], [0]))
        self.assertEqual(rep["verdict"], "DEGENERATE_GRID")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("PARAM_DEGENERATE_GRID", codes)
        self.assertNotIn("PARAM_PLATEAU", codes)

    def test_line_grid_not_plateau(self):
        rep = surface_audit(surf_strategy(lambda x, y: 50.0 - abs(y), [0], GRID),
                            D.regime_trend_df(), cfg([0], GRID))
        self.assertEqual(rep["verdict"], "DEGENERATE_GRID")
        self.assertNotIn("PARAM_PLATEAU", {i["code"] for i in rep["issues"]})

    def test_clean_2d_plateau_still_classified(self):
        rep = surface_audit(surf_strategy(smooth), D.regime_trend_df(), cfg())
        self.assertEqual(rep["verdict"], "PLATEAU")   # fix must not hurt real grids
        self.assertEqual(rep["plateau_frac"], 1.0)


class TestSurfaceRobustnessIntegration(unittest.TestCase):
    """The issue must propagate through Robustness as CONDITIONAL, never a silent
    PASS and never a wrong island charge."""

    def test_nonfinite_via_robustness(self):
        def p(x, y):
            if x == 1.0 and y == 1.0:
                return float("inf")
            return smooth(x, y)
        rep = rcheck(surf_strategy(p), D.regime_trend_df(), SPEC, cfg())
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("PARAM_NONFINITE_PNL", codes)
        self.assertNotIn("PARAM_ISLAND", codes)
        self.assertIn("CONDITIONAL PASS", rep["status"])


class TestClusterTimestampIntegrity(unittest.TestCase):
    """Day bucketing must normalise int seconds/ms/ns the same way the execution
    timeline does - a week of distinct days is never 'CLUSTERED'."""

    def _logs(self, ts_list):
        return [{"entry_ts": ts} for ts in ts_list]

    def test_epoch_seconds_spread_days(self):
        logs = self._logs([1757000000 + 86400 * k for k in range(7)])
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")       # was CLUSTERED / 1 day
        self.assertEqual(rep["active_days"], 7)

    def test_epoch_milliseconds_spread_days(self):
        logs = self._logs([1757000000000 + 86400000 * k for k in range(7)])
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["active_days"], 7)

    def test_epoch_nanoseconds_spread_days(self):
        base = pd.Timestamp("2025-09-05").value
        logs = self._logs([base + int(86400e9) * k for k in range(7)])
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["active_days"], 7)

    def test_iso_strings_spread_days(self):
        logs = self._logs([f"2025-09-0{k + 1} 10:00:00" for k in range(7)])
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["active_days"], 7)

    def test_datetime_spread_days_still_clean(self):
        logs = self._logs([pd.Timestamp(2025, 9, k + 1, 10, 0) for k in range(7)])
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["active_days"], 7)

    def test_concentrated_days_still_clustered(self):
        logs = self._logs([pd.Timestamp(2025, 9, 1, 10, 0)] * 5)
        rep = cluster_audit(logs)
        self.assertEqual(rep["verdict"], "CLUSTERED")  # regression: still catches
        self.assertEqual(rep["active_days"], 1)

    def test_missing_timestamps_not_verified(self):
        rep = cluster_audit([{"pnl": 1.0}, {"pnl": 2.0}])
        self.assertEqual(rep["verdict"], "NOT VERIFIED")

    def test_nonfinite_timestamp_not_verified(self):
        rep = cluster_audit(self._logs([float("nan"), float("nan")]))
        self.assertEqual(rep["verdict"], "NOT VERIFIED")  # never a silent bucket


if __name__ == "__main__":
    unittest.main()
