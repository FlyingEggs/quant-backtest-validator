"""Unit tests for the validator reference implementation.

Run from the repository root:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import (full_audit, lag_sensitivity, period_expansion,
                       fill_timing_sensitivity, randomized_control,
                       return_independence)
from examples import demo as D


class TestGateChecks(unittest.TestCase):

    def test_honest_next_open_passes_full_audit(self):
        df = D.regime_trend_df()
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation="completed",
                         n_shuffles=60, seed=11, verbose=False)
        self.assertTrue(aud["passed"])
        self.assertEqual(aud["verdict"], "PASS")
        self.assertEqual(aud["problems"], [])

    def test_same_bar_lag_clean_but_fill_fails(self):
        df = D.same_bar_leak_df()
        lag = lag_sensitivity(df, "sig", D.same_bar_bt, verbose=False)
        self.assertEqual(lag["verdict"], "PASS")
        fill = fill_timing_sensitivity(df, D.same_bar_bt, verbose=False)
        self.assertEqual(fill["verdict"], "FAIL")
        # profit collapse is ~total: it was all intraday (only the last bar's fill
        # survives the ffill edge, a negligible residue)
        self.assertLess(fill["lagged_pnl"], fill["base_pnl"] * 0.001)

    def test_period_expansion_is_hard_gate(self):
        df = D.daily_signal_df()
        rep = period_expansion(df, "sig", bar_seconds=300, verbose=False)
        self.assertEqual(rep["verdict"], "SUSPECT")
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation=None, n_shuffles=40, seed=7,
                         verbose=False)
        self.assertFalse(aud["passed"])
        self.assertTrue(any("expansion" in p for p in aud["problems"]))
        aud2 = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                          expansion_confirmation="completed", n_shuffles=40,
                          seed=7, verbose=False)
        self.assertTrue(aud2["passed"])

    def test_missing_column_raises(self):
        df = D.regime_trend_df().drop(columns=["sig"])
        with self.assertRaises(KeyError):
            period_expansion(df, "sig", verbose=False)


class TestRandomizedControl(unittest.TestCase):

    def test_edge_confirmed_for_trend(self):
        df = D.regime_trend_df()
        rep = randomized_control(df, "sig", D.next_open_hold(5),
                                 n_shuffles=60, seed=11, verbose=False)
        self.assertEqual(rep["verdict"], "EDGE_CONFIRMED")
        self.assertLess(rep["p_value"], 0.05)

    def test_no_edge_for_noise(self):
        df = D.noise_df()
        rep = randomized_control(df, "sig", D.next_open_hold(5),
                                 n_shuffles=60, seed=12, verbose=False)
        self.assertEqual(rep["verdict"], "NO_EDGE")
        self.assertGreaterEqual(rep["p_value"], 0.5)

    def test_reproducible_with_seed(self):
        df = D.regime_trend_df()
        a = randomized_control(df, "sig", D.next_open_hold(5), n_shuffles=40,
                               seed=3, verbose=False)
        b = randomized_control(df, "sig", D.next_open_hold(5), n_shuffles=40,
                               seed=3, verbose=False)
        self.assertEqual(a["p_value"], b["p_value"])
        self.assertEqual(a["seed"], 3)

    def test_bad_shuffles_rejected(self):
        df = D.noise_df()
        with self.assertRaises(ValueError):
            randomized_control(df, "sig", D.next_open_hold(5), n_shuffles=0,
                               verbose=False)


class TestReturnIndependence(unittest.TestCase):

    def test_ar1_autocorrelated(self):
        rep = return_independence(D.ar1_rets(), verbose=False)
        self.assertEqual(rep["verdict"], "AUTOCORRELATED")
        self.assertLess(rep["n_eff"], rep["n"] / 2)
        self.assertLess(rep["lb_p"], 0.05)

    def test_white_noise_independent(self):
        x = np.random.default_rng(99).normal(0, 1, 400)
        rep = return_independence(x, verbose=False)
        self.assertEqual(rep["verdict"], "INDEPENDENT")
        self.assertGreater(rep["n_eff"], 0.8 * rep["n"])

    def test_small_sample_insufficient(self):
        rep = return_independence([1.0] * 5, verbose=False)
        self.assertEqual(rep["verdict"], "INSUFFICIENT")
        self.assertIsNone(rep["n_eff"])


if __name__ == "__main__":
    unittest.main()
