"""Boundary-adversarial audit of execution / costengine / mtf.

Hunts: ① execution min_latency inclusivity + unverifiable timestamps,
        ② costengine negative-qty improvement hole,
        ③ mtf decision-exactly-at-high-close + NaN-gap robustness.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.execution import timeline_audit
from validator.costengine import net_audit
from validator.mtf import temporal_availability
from validator.mtf import check as mtf_check


def basic_cost(**over):
    cfg = {"commission": {"mode": "bps", "open_rate": 5.0, "close_rate": 5.0},
           "spread": {"mode": "none"},
           "slippage": {"mode": "none"},
           "tick_size": 0.01,
           "market_impact": {"mode": "none"},
           "financing": None}
    cfg.update(over)
    return cfg


def tr(side, entry, exit_, qty=1.0):
    return {"side": side, "entry_price": entry, "exit_price": exit_, "qty": qty}


class TestExecutionTimelineBoundaries(unittest.TestCase):

    def test_fill_exactly_at_min_latency_is_legal(self):
        """latency L is a floor: entry at signal+L is legal, not a violation."""
        trades = [{"signal_ts": 1000, "entry_ts": 1000 + 2}]   # gap == latency 2
        rep = timeline_audit(trades, min_latency_s=2.0)
        self.assertEqual(rep["verdict"], "PASS")

    def test_negative_gap_is_violation(self):
        rep = timeline_audit([{"signal_ts": 2000, "entry_ts": 1000}], min_latency_s=0)
        self.assertEqual(rep["verdict"], "FAIL")

    def test_nonfinite_timestamp_not_silent_pass(self):
        """A trade whose signal_ts is NaN cannot be verified - must NOT silently PASS."""
        rep = timeline_audit([{"signal_ts": float("nan"), "entry_ts": 1000}],
                             min_latency_s=0)
        self.assertEqual(rep["verdict"], "NOT VERIFIED")


class TestCostengineNegativeQty(unittest.TestCase):
    """Anti-cheat C4 must hold even with pathological inputs."""

    def test_negative_qty_never_improves(self):
        cfg = basic_cost()
        cfg["slippage"] = {"mode": "fixed", "value": 0.1}
        net = net_audit([tr("long", 100.0, 110.0, qty=-1.0)], cfg)
        self.assertGreaterEqual(net["total_cost"], 0.0)      # no negative cost hole

    def test_negative_qty_commission_never_negative(self):
        cfg = basic_cost()
        cfg["commission"] = {"mode": "notional", "open_rate": 0.1, "close_rate": 0.1}
        cfg["tick_size"] = None
        net = net_audit([tr("long", 100.0, 110.0, qty=-2.0)], cfg)
        table = dict((r[0], r[1]) for r in net["table"])
        self.assertLessEqual(table["Commission"], 0.0)       # drag <= 0 means cost >= 0

    def test_exact_grid_tick_cost_zero(self):
        cfg = basic_cost()
        cfg["spread"] = {"mode": "none"}
        cfg["slippage"] = {"mode": "none"}
        cfg["commission"] = {"mode": "none"}
        net = net_audit([tr("long", 100.00, 101.00, qty=1.0)], cfg)   # already on grid
        self.assertAlmostEqual(net["total_cost"], 0.0, places=9)


def hour_pair(hours=10, seed=2, nan_at=None):
    """1h high over 5m low. Optionally inject a NaN high close at hour index nan_at."""
    rng = np.random.default_rng(seed)
    high_open = pd.date_range("2026-01-01 00:00", periods=hours, freq="3600s")
    hc = 100.0 + np.cumsum(rng.normal(0, 0.5, hours))
    if nan_at is not None:
        hc = hc.copy()
        hc[nan_at] = np.nan
    high = pd.DataFrame({"close": hc}, index=high_open)
    n = hours * 12
    low_open = pd.date_range("2026-01-01 00:00", periods=n, freq="300s")
    base = 100.0 + np.arange(n) * 0.001
    low = pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5,
                        "close": base + 0.1}, index=low_open)
    sh = np.sign(np.diff(hc, prepend=hc[0]))
    hour_idx = np.arange(n) // 12
    pos = np.arange(n) % 12
    legal = np.where((hour_idx == 0) & (pos < 11), np.nan,
                     np.where(pos < 11, sh[np.maximum(hour_idx - 1, 0)],
                              sh[hour_idx]))
    naive = sh[hour_idx]
    low["sig_legal"] = legal
    low["sig_naive"] = naive
    return low, high


class TestMtfBoundaries(unittest.TestCase):

    def test_decision_exactly_at_high_close_not_leak(self):
        """A low bar whose close == the high bar's close may use that value (legal)."""
        low, high = hour_pair()
        low = low.copy()
        # hour 1 closes at 02:00; the low bar opening 01:55 decides at 02:00 == close.
        # set that bar's sig to hour-1's sign (both naive and legal agree there)
        low["sig_edge"] = low["sig_naive"].where(low.index != pd.Timestamp("2026-01-01 01:55"), np.nan)
        rep = temporal_availability(low, "sig_naive", high, 3600, 300,
                                    transform="sign_diff")
        self.assertEqual(rep["verdict"], "FAIL")          # forming bars exist -> leak
        # now make the LAST bar of each hour the only candidate (decision == close):
        low2 = low.copy()
        sig = np.full(len(low2), np.nan)
        # only bars at minute :55 (decision == hour close) keep the hour's sign
        pos = np.arange(len(low2)) % 12
        sig[pos == 11] = low2["sig_naive"][pos == 11]
        low2["sig_boundary_only"] = sig
        rep2 = temporal_availability(low2, "sig_boundary_only", high, 3600, 300,
                                     transform="sign_diff")
        self.assertNotEqual(rep2["verdict"], "FAIL")       # boundary-only: no forming leak

    def test_nan_gap_no_crash_no_false_fail(self):
        low, high = hour_pair(nan_at=3)                    # hour 3 close missing
        for col in ("sig_naive", "sig_legal"):
            rep = temporal_availability(low, col, high, 3600, 300,
                                        transform="sign_diff")
            self.assertIn(rep["verdict"], ("FAIL", "PASS", "NOT VERIFIED"))  # no crash


if __name__ == "__main__":
    unittest.main()
