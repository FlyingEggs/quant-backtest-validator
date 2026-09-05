"""Boundary round 2 — attack robustness param oscillation, empty audit scope, and
empty walk-forward windows."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator.robustness import check as rcheck
from validator.wf import walk_forward_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


class TestParamOscillation(unittest.TestCase):

    def _sec(self, pnls):
        df = D.regime_trend_df()

        def run(frame, params):
            k = params["k"]
            return {"pnl": float(pnls[int(k)]), "trades": 100}

        strat = Strategy(name="osc", run=run, default_params={"k": 0},
                         param_grid={"k": [str(i) for i in range(len(pnls))]},
                         entry_semantics="next_open")
        # k values are strings mapped to pnl index via int(k)
        def run2(frame, params):
            return {"pnl": float(pnls[int(float(params["k"]))]), "trades": 100}
        strat.run = run2
        return strat

    def test_alternating_sign_series_flagged(self):
        """[-100, +100, -100, +100]: every adjacent delta is exactly 2.0x median ->
        escapes the >2.0 cliff rule, but oscillates in sign => must be flagged."""
        strat = self._sec([-100.0, 100.0, -100.0, 100.0])
        sec = rcheck(strat, D.regime_trend_df(), SPEC, {"seed": 1})
        codes = {i["code"] for i in sec["issues"]}
        self.assertIn("PARAM_OSCILLATION", codes)

    def test_monotone_robust_not_flagged(self):
        strat = self._sec([10.0, 11.0, 12.0, 13.0])
        sec = rcheck(strat, D.regime_trend_df(), SPEC, {"seed": 1})
        codes = {i["code"] for i in sec["issues"]}
        self.assertNotIn("PARAM_OSCILLATION", codes)


class TestEmptyScope(unittest.TestCase):
    """An empty/unknown scope must never yield a PASS."""

    def test_empty_scope_raises(self):
        df = D.regime_trend_df()

        def run(frame, params):
            return {"pnl": 1.0, "trades": 1}
        strat = Strategy(name="s", run=run, entry_semantics="next_open")
        with self.assertRaises(ValueError):
            audit(strat, df, SPEC, {"scope": []})

    def test_unknown_only_scope_raises(self):
        df = D.regime_trend_df()

        def run(frame, params):
            return {"pnl": 1.0, "trades": 1}
        strat = Strategy(name="s", run=run, entry_semantics="next_open")
        with self.assertRaises(ValueError):
            audit(strat, df, SPEC, {"scope": ["NOPE", "MTF-ish"]})


class TestEmptyWalkForward(unittest.TestCase):

    def test_too_short_sample_not_clean(self):
        """Sample shorter than min_is_bars+oos_bars: zero windows must surface as an
        issue, not an empty clean walk-forward."""
        df = D.regime_trend_df().iloc[:200]

        def run(frame, params):
            return {"pnl": 1.0, "trades": 10}
        strat = Strategy(name="s", run=run, entry_semantics="next_open")
        cfg = {"oos": {"policy": "ENTRY_IN_WINDOW", "n_windows": 2,
                       "oos_bars": 200, "min_is_bars": 1000}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        self.assertEqual(rep["windows"], [])
        self.assertTrue(any(i["code"] == "WF_NO_WINDOWS" for i in rep["issues"]))

    def test_exact_fit_last_window_runs(self):
        """Window whose OOS runs exactly to the sample end must be present."""
        n = 160
        df = D.regime_trend_df().iloc[:n]

        def run(frame, params):
            return {"pnl": 1.0, "trades": 5}
        strat = Strategy(name="s", run=run, entry_semantics="next_open")
        cfg = {"oos": {"policy": "ENTRY_IN_WINDOW", "n_windows": 3,
                       "oos_bars": 30, "min_is_bars": 100}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        self.assertEqual(len(rep["windows"]), 2)      # [100,130) and [130,160)


if __name__ == "__main__":
    unittest.main()
