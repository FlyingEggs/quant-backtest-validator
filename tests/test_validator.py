"""Unit tests for the validator reference implementation (v1.1).

Covers the three-tier verdict (PASS / CONDITIONAL PASS / FAIL), the P0-P4 issue log,
the linear-rho N_eff, and the reshuffled-null randomized control.

Run from the repository root:

    python3 -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import (full_audit, lag_sensitivity, period_expansion,
                       fill_timing_sensitivity, randomized_control,
                       return_independence, save_report, to_jsonable)
from examples import demo as D


def p(issues, code):
    return [i for i in issues if i["code"] == code]


class TestThreeTierVerdict(unittest.TestCase):
    """PASS / CONDITIONAL PASS / FAIL semantics."""

    def test_honest_next_open_passes(self):
        df = D.regime_trend_df()
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation="completed",
                         n_shuffles=60, seed=11, verbose=False)
        self.assertEqual(aud["verdict"], "PASS")
        self.assertTrue(aud["passed"])
        self.assertEqual(p(aud["issues"], "LAG_DEPENDENCE"), [])
        self.assertEqual(p(aud["issues"], "EXECUTION_FILL"), [])

    def test_lag_clean_for_slow_signal(self):
        df = D.regime_trend_df()
        rep = lag_sensitivity(df, "sig", D.next_open_hold(5), verbose=False)
        self.assertEqual(rep["verdict"], "STABLE")

    def test_short_horizon_is_conditional_not_fail(self):
        """A legitimately 1-bar-horizon signal collapses under lag AND fill (same
        perturbation) - it must be CONDITIONAL PASS (review), never auto-FAIL."""
        df = D.markov_short_df()
        aud = full_audit(df, "sig", D.markov_bt, bar_seconds=300,
                         n_shuffles=60, seed=5, verbose=False)
        self.assertEqual(aud["verdict"], "CONDITIONAL PASS")
        self.assertEqual(p(aud["issues"], "LAG_DEPENDENCE")[0]["severity"], "P1")
        self.assertEqual(p(aud["issues"], "EXECUTION_FILL_REVIEW")[0]["severity"], "P1")
        self.assertEqual(p(aud["issues"], "EXECUTION_FILL"), [])   # no P0

    def test_same_bar_fill_is_p0_fail(self):
        """Retention <10% after a 1-bar fill shift => P0 execution look-ahead."""
        df = D.same_bar_leak_df()
        aud = full_audit(df, "sig", D.same_bar_bt, bar_seconds=300,
                         expansion_confirmation="completed",
                         n_shuffles=40, seed=3, verbose=False)
        self.assertEqual(aud["verdict"], "FAIL")
        fill_issue = p(aud["issues"], "EXECUTION_FILL")
        self.assertEqual(len(fill_issue), 1)
        self.assertEqual(fill_issue[0]["severity"], "P0")

    def test_period_expansion_is_p0_gate(self):
        df = D.daily_signal_df()
        rep = period_expansion(df, "sig", bar_seconds=300, verbose=False)
        self.assertEqual(rep["verdict"], "SUSPECT")
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation=None, n_shuffles=40, seed=7,
                         verbose=False)
        self.assertEqual(aud["verdict"], "FAIL")
        self.assertEqual(p(aud["issues"], "PERIOD_EXPANSION")[0]["severity"], "P0")
        aud2 = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                          expansion_confirmation="completed", n_shuffles=40,
                          seed=7, verbose=False)
        self.assertEqual(aud2["verdict"], "PASS")


class TestLagSensitivity(unittest.TestCase):

    def test_nan_head_not_filled(self):
        """Shifted column must keep NaN at the head (no artificial warm-up value)."""
        df = D.regime_trend_df()
        shifted = df.copy()
        shifted["sig"] = df["sig"].shift(1)
        self.assertTrue(np.isnan(shifted["sig"].iloc[0]))
        self.assertFalse(np.isnan(df["sig"].iloc[0]))

    def test_no_trades_insufficient(self):
        df = D.regime_trend_df().copy()
        df["sig"] = 0.0
        rep = lag_sensitivity(df, "sig", D.next_open_hold(5), verbose=False)
        self.assertEqual(rep["verdict"], "INSUFFICIENT")


class TestFillTiming(unittest.TestCase):

    def test_same_bar_collapse(self):
        df = D.same_bar_leak_df()
        lag = lag_sensitivity(df, "sig", D.same_bar_bt, verbose=False)
        self.assertEqual(lag["verdict"], "STABLE")       # signal column itself is clean
        fill = fill_timing_sensitivity(df, D.same_bar_bt, verbose=False)
        self.assertEqual(fill["verdict"], "FAIL")
        self.assertLess(abs(fill["shifted_pnl"]), abs(fill["base_pnl"]) * 0.001)

    def test_marker_injected(self):
        df = D.same_bar_leak_df()
        shifted = df.copy()
        shifted["open"] = df["open"].shift(-1).ffill()
        shifted["__fill_shifted__"] = 1.0
        self.assertIn("__fill_shifted__", shifted.columns)   # contract: marker present

    def test_honest_fill_pass(self):
        df = D.regime_trend_df()
        fill = fill_timing_sensitivity(df, D.next_open_hold(5), verbose=False)
        self.assertEqual(fill["verdict"], "PASS")


class TestRandomizedControl(unittest.TestCase):

    def test_trend_beats_shuffled_null(self):
        df = D.regime_trend_df()
        rep = randomized_control(df, "sig", D.next_open_hold(5),
                                 n_shuffles=60, seed=11, verbose=False)
        self.assertEqual(rep["verdict"], "BEATS_SHUFFLED_NULL")
        self.assertLess(rep["p_value"], 0.05)

    def test_noise_does_not(self):
        df = D.noise_df()
        rep = randomized_control(df, "sig", D.next_open_hold(5),
                                 n_shuffles=60, seed=12, verbose=False)
        self.assertEqual(rep["verdict"], "NO_EDGE_VS_SHUFFLED_NULL")
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
    """N_eff is the linear-rho ESS of the mean (Kass et al. 1998)."""

    def test_ar1_autocorrelated(self):
        rep = return_independence(D.ar1_rets(), verbose=False)
        self.assertEqual(rep["verdict"], "AUTOCORRELATED")
        self.assertLess(rep["n_eff"], rep["n"] / 3)      # 400 -> ~53, linear rho
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


class TestReportSerialization(unittest.TestCase):

    def test_report_is_json_serializable(self):
        df = D.regime_trend_df()
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation="completed",
                         n_shuffles=20, seed=11, verbose=False)
        clean = to_jsonable(aud)
        json.dumps(clean)                                # must not raise
        self.assertIn("verdict", clean)
        self.assertIn("issues", clean)

    def test_save_report_roundtrip(self):
        df = D.daily_signal_df()
        aud = full_audit(df, "sig", D.next_open_hold(5), bar_seconds=300,
                         expansion_confirmation=None, n_shuffles=20, seed=7,
                         verbose=False)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.json")
            save_report(aud, path)
            with open(path) as fh:
                loaded = json.load(fh)
        self.assertEqual(loaded["verdict"], aud["verdict"])
        self.assertEqual(loaded["issues"][0]["severity"], "P0")


if __name__ == "__main__":
    unittest.main()
