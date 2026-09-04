"""V3 — MTF Temporal Availability Engine tests.

Key scenario: a 1h HIGH frame over 5-min LOW bars.
  legal col  = sign of the PREVIOUS (completed) hour  -> PASS
  naive col  = sign of the CURRENT (forming) hour    -> MTF_LEAK (P0)
  random col = matches neither                       -> NOT VERIFIED (no fake pass)
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator import mtf
from validator.mtf import temporal_availability

START = "2026-01-01 00:00"


def build(hours: int = 60, seed: int = 3):
    """Return (low_df, high_df) with aligned 5m/1h bars. high['close'] random walk."""
    rng = np.random.default_rng(seed)
    high_open = pd.date_range(START, periods=hours, freq="3600s")
    hc = 100.0 + np.cumsum(rng.normal(0, 0.5, hours))
    high = pd.DataFrame({"close": hc}, index=high_open)

    n_low = hours * 12
    low_open = pd.date_range(START, periods=n_low, freq="300s")
    # fake OHLC so the low frame is structurally valid
    base = 100.0 + np.arange(n_low) * 0.001
    low = pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5,
                        "close": base + 0.1}, index=low_open)
    sh = np.sign(np.diff(hc, prepend=hc[0]))                  # per-hour sign series
    hour_idx = np.arange(n_low) // 12
    pos = np.arange(n_low) % 12
    legal = np.where((hour_idx == 0) & (pos < 11),
                     np.nan,                                   # no completed hour yet
                     np.where(pos < 11,
                              sh[np.maximum(hour_idx - 1, 0)],
                              sh[hour_idx]))                   # last closed hour
    naive = sh[hour_idx]                                       # current (forming) hour
    low["sig_legal"] = legal
    low["sig_naive"] = naive
    rng2 = np.random.default_rng(seed + 1)
    low["sig_random"] = np.where(rng2.random(n_low) < 0.5, 1.0, -1.0)
    return low, high


def binding(col):
    return {"mtf": {"col": col, "frame": "h1", "frame_seconds": 3600,
                    "transform": "sign_diff"}}


class TestMtfEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.low, cls.high = build()

    def _run(self, col):
        return temporal_availability(self.low, col, self.high, 3600, 300,
                                     transform="sign_diff")

    def test_legal_previous_hour_passes(self):
        rep = self._run("sig_legal")
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["leak_rows"], 0)

    def test_naive_current_hour_leaks(self):
        rep = self._run("sig_naive")
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertGreater(rep["leak_rows"], 0)
        self.assertTrue(rep["issues"])
        self.assertEqual(rep["issues"][0]["code"], "MTF_LEAK")
        self.assertEqual(rep["issues"][0]["severity"], "P0")

    def test_unattributed_column_not_verified(self):
        rep = self._run("sig_random")
        self.assertEqual(rep["verdict"], "NOT VERIFIED")
        self.assertEqual(rep["issues"], [])                    # FP guard: no MTF_LEAK
        self.assertLess(rep["leak_frac"], 0.9)                 # ~50% = chance, not leak

    def test_missing_col_not_verified(self):
        rep = temporal_availability(self.low, "nope", self.high, 3600, 300,
                                    transform="sign_diff")
        self.assertEqual(rep["verdict"], "NOT VERIFIED")


class TestMtfInAudit(unittest.TestCase):
    """Integration: MTF section within the 4-state audit pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.low, cls.high = build()

    def _audit_mtf(self, col):
        def run(df, params):
            return {"pnl": 1.0, "trades": 1}
        strat = Strategy(name="mtf-test", run=run, entry_semantics="next_open")
        spec = DataSpec(bar_seconds=300, source="synthetic",
                        timeframes={"h1": self.high})
        cfg = binding(col)
        cfg["scope"] = ["MTF"]
        return audit(strat, self.low, spec, cfg)

    def test_naive_leak_fails_overall(self):
        rep = self._audit_mtf("sig_naive")
        self.assertEqual(rep["sections"]["MTF"]["status"], "FAIL")
        self.assertEqual(rep["overall"], "FAIL")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("MTF_LEAK", codes)

    def test_legal_passes_within_scope(self):
        rep = self._audit_mtf("sig_legal")
        self.assertEqual(rep["sections"]["MTF"]["status"], "PASS")
        self.assertEqual(rep["overall"], "PASS")       # MTF-only scope fully verified

    def test_default_no_binding_not_verified(self):
        def run(df, params):
            return {"pnl": 1.0, "trades": 1}
        strat = Strategy(name="m", run=run, entry_semantics="next_open")
        rep = audit(strat, self.low, DataSpec(bar_seconds=300),
                    {"scope": ["MTF"]})
        self.assertEqual(rep["sections"]["MTF"]["status"], "NOT VERIFIED")
        self.assertEqual(rep["overall"], "INCOMPLETE")  # honest, never PASS


if __name__ == "__main__":
    unittest.main()
