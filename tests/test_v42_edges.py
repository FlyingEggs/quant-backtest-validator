"""V4.2 edge sweep — code-auditor round on the V4.1/V4.2 integrity checks.

Targets paths the round tests did not pin: tolerance boundaries (no FP on float
noise, no FN on real drift), exception/fallback branches (WF fit_error windows),
gap-statistics precision, price-reachability FP guards and off-bar timestamps,
short-side & zero-price return cross-checks, crash-localisation per section, and a
full-dimension forged strategy hitting every ledger check at once.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator import data_integrity as di
from validator.wf import walk_forward_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")
COST = {"commission": {"mode": "bps", "open_rate": 5.0, "close_rate": 5.0}}


def _audit(run, scope=("Costs",), df=None, spec=SPEC):
    frame = df if df is not None else D.regime_trend_df(n=300)
    return audit(Strategy(name="e", run=run, entry_semantics="next_open"),
                 frame, spec,
                 {"scope": list(scope), "cost": COST, "seed": 1})


def _leg(run, pnl, ret, side="long", ep=100.0):
    return {"side": side, "qty": 1.0, "entry_price": ep, "exit_price": ep + ret,
            "entry_ts": pd.Timestamp("1990-01-01"),
            "exit_ts": pd.Timestamp("1990-01-02")}


class TestLedgerTolerance(unittest.TestCase):

    def _status(self, pnl, ret):
        rep = _audit(lambda f, p: {"pnl": pnl, "trades": 1,
                                   "trades_log": [_leg(None, pnl, ret)]})
        return rep["sections"]["Costs"]["status"]

    def test_float_noise_within_tolerance_verified(self):
        self.assertEqual(self._status(1.0 + 1e-9, 1.0), "VERIFIED")

    def test_real_drift_fails(self):
        self.assertEqual(self._status(2.0, 1.0), "FAIL")


class TestWalkForwardFallbacks(unittest.TestCase):

    def test_fit_error_window_reported(self):
        def run(f, p):
            return {"pnl": 10.0 * float(p.get("k", 1.0)), "trades": 1,
                    "trades_log": []}

        def bad_fit(d):
            raise RuntimeError("fit failed")
        strat = Strategy(name="e", run=run, default_params={"k": 1.0},
                         entry_semantics="next_open", fit_is=bad_fit,
                         accepts_frozen=True)
        rep = walk_forward_audit(strat, D.regime_trend_df(n=1500),
                                 {"oos": {"n_windows": 2, "oos_bars": 200,
                                          "min_is_bars": 300}}, SPEC)
        self.assertTrue(rep["windows"])
        w = rep["windows"][0]
        self.assertIn("fit_error", w["params_source"])
        self.assertIsNone(w["frozen_params_hash"])
        # fallback to defaults still runs the window (no crash)
        self.assertIn(w["oos_pnl"], (10.0, 0.0))

    def test_frozen_with_supports_from_bar(self):
        def run(f, p):
            k = float(p.get("k", 1.0))
            ts = f.index
            i0 = max(0, int(len(f) * 0.5))
            i1 = min(len(f) - 1, i0 + 1)
            log = [{"side": "long", "qty": 1.0, "entry_price": 100.0,
                    "exit_price": 110.0, "entry_ts": ts[i0], "exit_ts": ts[i1]}]
            return {"pnl": 10.0, "trades": 1, "trades_log": log}
        strat = Strategy(name="s", run=run, default_params={"k": 1.0},
                         entry_semantics="next_open", supports_from_bar=True,
                         fit_is=lambda d: {"k": 3.0}, accepts_frozen=True)
        rep = walk_forward_audit(strat, D.regime_trend_df(n=1500),
                                 {"oos": {"n_windows": 2, "oos_bars": 200,
                                          "min_is_bars": 300}}, SPEC)
        w = rep["windows"][0]
        self.assertEqual(w["params_source"], "frozen_fit")
        self.assertIsNotNone(w["frozen_params_hash"])


class TestGapStatistics(unittest.TestCase):

    def _frame(self, steps):
        return pd.DataFrame({"open": [1.0] * len(steps),
                             "high": [1.0] * len(steps),
                             "low": [1.0] * len(steps),
                             "close": [1.0] * len(steps)},
                            index=[pd.Timestamp("2026-01-01 00:00:00")
                                   + pd.Timedelta(seconds=300 * i)
                                   for i in steps])

    def test_missing_bar_count_precise(self):
        df = self._frame([0, 1, 2, 3, 4, 11])          # 6-bar hole after index 4
        findings = [i["finding"] for i in di.check(df, DataSpec(bar_seconds=300))
                    ["issues"] if i["code"] == "DATA_NONUNIFORM"]
        self.assertEqual(len(findings), 1)
        self.assertIn("~6 missing", findings[0])

    def test_jitter_within_tolerance_not_flagged(self):
        df = self._frame([0, 1, 2, 3, 4])              # uniform
        df2 = df.copy()
        df2.index = [df.index[0] + pd.Timedelta(seconds=int(300 * 1.4 * k))
                     for k in range(5)]
        codes = [i["code"] for i in di.check(df2, DataSpec(bar_seconds=300))
                 ["issues"]]
        self.assertNotIn("DATA_NONUNIFORM", codes)     # 1.4x < 1.5x threshold


class TestPriceReachabilityEdges(unittest.TestCase):

    def _bar_df(self, base=3000.0, n=50):
        closes = base * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.001, n))
        df = D.frame(closes, 300)
        return df

    def test_price_just_outside_tolerance_not_flagged(self):
        df = self._bar_df()
        hi = float(df["high"].iloc[10])
        xp = hi * 1.005                      # 0.5% above high < 1% tolerance
        def run(f, p):
            return {"pnl": xp - float(df["open"].iloc[10]), "trades": 1,
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": float(df["open"].iloc[10]),
                                    "exit_price": xp, "entry_ts": df.index[10],
                                    "exit_ts": df.index[11],
                                    "signal_ts": df.index[9]}]}
        rep = _audit(run, ("Costs",), df=df)
        self.assertNotIn("EXEC_PRICE_UNREACHABLE",
                         [i["code"] for i in rep["sections"]["Costs"]["issues"]])

    def test_price_way_outside_flagged(self):
        df = self._bar_df()
        xp = float(df["high"].iloc[10]) * 1.02          # 2% above high
        def run(f, p):
            return {"pnl": xp, "trades": 1,
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": float(df["open"].iloc[10]),
                                    "exit_price": xp, "entry_ts": df.index[10],
                                    "exit_ts": df.index[11],
                                    "signal_ts": df.index[9]}]}
        rep = _audit(run, ("Costs",), df=df)
        self.assertIn("EXEC_PRICE_UNREACHABLE",
                      [i["code"] for i in rep["sections"]["Costs"]["issues"]])

    def test_off_bar_timestamp_not_false_positive(self):
        # timestamp not on a bar -> unverifiable -> no reachability claim either way
        df = self._bar_df()
        off = df.index[10] + pd.Timedelta(seconds=1)
        def run(f, p):
            return {"pnl": 40.0, "trades": 1,
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": 100.0, "exit_price": 140.0,
                                    "entry_ts": off, "exit_ts": off}]}
        rep = _audit(run, ("Costs",), df=df)
        self.assertNotIn("EXEC_PRICE_UNREACHABLE",
                         [i["code"] for i in rep["sections"]["Costs"]["issues"]])


class TestRetsLedgerEdges(unittest.TestCase):
    """via full audit on scope Statistics+Costs"""

    def _run_audit(self, run):
        return audit(Strategy(name="r", run=run, entry_semantics="next_open"),
                     D.regime_trend_df(n=300), SPEC,
                     {"scope": ["Statistics", "Costs"], "cost": COST, "seed": 1})

    def test_short_side_return_mismatch_caught(self):
        def run(f, p):
            # short: entry 105 -> exit 100 => +5/105 per unit; forged rets 0.05
            return {"pnl": 5.0, "trades": 1, "rets": [0.05],
                    "trades_log": [_leg(None, 5.0, -5.0, side="short", ep=105.0)]}
        rep = self._run_audit(run)
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("STAT_RETS_LEDGER_MISMATCH", codes)

    def test_zero_entry_price_does_not_crash(self):
        def run(f, p):
            return {"pnl": 1.0, "trades": 1, "rets": [0.5],
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": 0.0, "exit_price": 1.0}]}
        rep = self._run_audit(run)                      # division guard, no crash
        self.assertIn(rep["overall"], ("PASS", "CONDITIONAL PASS", "INCOMPLETE"))


class TestFullDimensionForgery(unittest.TestCase):
    """One strategy forging EVERY ledger dimension at once - every guard fires."""

    def test_all_forged_dimensions_caught_together(self):
        df = D.regime_trend_df(n=300)
        ts = df.index

        def run(f, p):
            return {"pnl": 999999.0, "trades": 1, "rets": [0.99],
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": 100.0, "exit_price": 140.0,
                                    "entry_ts": ts[10], "exit_ts": ts[11],
                                    "signal_ts": ts[9]}]}
        rep = audit(Strategy(name="all", run=run, entry_semantics="next_open"),
                    df, SPEC, {"scope": ["Statistics", "Costs"],
                               "cost": COST, "seed": 1})
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("TRADE_LEDGER_PNL_MISMATCH", codes)    # claim vs ledger
        self.assertIn("STAT_RETS_LEDGER_MISMATCH", codes)    # rets vs ledger
        self.assertIn("EXEC_PRICE_UNREACHABLE", codes)       # prices vs bars
        self.assertEqual(rep["overall"], "FAIL")


if __name__ == "__main__":
    unittest.main()
